#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cassert>
#include <filesystem>
#include <atomic>
#include <chrono>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <omp.h>

#ifdef _WIN32
#include <io.h>
#define ISATTY _isatty
#define FILENO _fileno
#else
#include <unistd.h>
#define ISATTY isatty
#define FILENO fileno
#endif

#include "../engines/tile_matrix_4cw/ldpc_tile_matrix_core.h"

namespace fs = std::filesystem;

// ==============================================================================
// Tenstorrent QuietBox 2: 40-Tile Face-Aligned Hardware-Accurate Lockstep Stepper
// Simulates the exact physical execution of the Tensix Blackhole hardware:
//   - 4-Face Tile Layout (Faces 0, 1, 2, 3)
//   - 40-Tile Face-Aligned Schedule:
//       32 Deg-6 tiles (32 checks/tile: 4 faces x 8 checks)
//       8 Deg-3 tiles (64 checks/tile: 4 faces x 16 checks)
//       4:1 Interleaving (4 Deg-6 tiles per 1 Deg-3 tile)
//   - Dual-Worker Gather / Scatter via SCRATCH_FACE_LUT (48 pairs = 96 offsets per worker)
//       Worker 0: Left Half (Faces 0 & 2)
//       Worker 1: Right Half (Faces 1 & 3)
//   - 32-Lane Systolic Vector Approximate-Min* on SFPU
//   - 40-Page Circular FIFO for Extrinsic Check Messages (CB_R_MSG)
// ==============================================================================

struct ParityCheckMatrix {
    uint32_t M = 0;
    uint32_t N = 0;
    uint32_t K = 0;
    uint32_t P = 0;
    uint32_t circulant_p = 0;
    std::vector<std::vector<uint32_t>> check_to_var;
    std::vector<std::vector<uint32_t>> var_to_check;
    std::vector<std::vector<uint32_t>> check_to_var_offsets;
    std::vector<uint32_t> info_var_offsets;
    std::vector<uint32_t> h_offsets_pairs;
};

struct DynamicTileDescriptor {
    std::vector<uint32_t> check_indices;
    uint8_t  check_degree;
    uint8_t  num_checks;
    uint16_t num_edges;
    uint16_t h_offset_start;
    uint16_t r_msg_page_offset;
};

static inline uint32_t make_pair_word(uint32_t ve, uint32_t vo) {
    uint32_t oe = get_tile_vnode_byte_offset(ve) & 0xFFFF;
    uint32_t oo = get_tile_vnode_byte_offset(vo) & 0xFFFF;
    return oe | (oo << 16);
}

bool parse_chinn_out(const std::string& filepath, ParityCheckMatrix& pcm) {
    std::ifstream infile(filepath);
    if (!infile.is_open()) {
        std::cerr << "[ERROR] Failed to open parity check matrix: " << filepath << std::endl;
        return false;
    }

    struct RawEdge { uint32_t r, c; float val; };
    std::vector<RawEdge> raw_lines;
    raw_lines.reserve(10000);

    uint32_t r, c;
    float val;
    while (infile >> r >> c >> val) {
        raw_lines.push_back({r, c, val});
    }

    if (raw_lines.empty()) return false;

    const auto& dim_line = raw_lines.back();
    pcm.M = dim_line.r;
    pcm.N = dim_line.c;
    pcm.K = pcm.N - pcm.M;
    pcm.circulant_p = pcm.M / 3;
    pcm.P = pcm.circulant_p;

    pcm.check_to_var.resize(pcm.M);
    pcm.var_to_check.resize(pcm.N);

    for (size_t i = 0; i + 1 < raw_lines.size(); ++i) {
        const auto& edge = raw_lines[i];
        if (std::abs(edge.val) > 1e-6f) {
            uint32_t m_idx = edge.r - 1;
            uint32_t v_idx = edge.c - 1;
            pcm.check_to_var[m_idx].push_back(v_idx);
            pcm.var_to_check[v_idx].push_back(m_idx);
        }
    }

    pcm.check_to_var_offsets.resize(pcm.M);
    for (uint32_t m = 0; m < pcm.M; ++m) {
        for (uint32_t v : pcm.check_to_var[m]) {
            pcm.check_to_var_offsets[m].push_back(get_tile_vnode_byte_offset(v));
        }
    }
    pcm.info_var_offsets.resize(pcm.K);
    for (uint32_t v = 0; v < pcm.K; ++v) {
        pcm.info_var_offsets[v] = get_tile_vnode_byte_offset(v);
    }

    std::cout << "[INFO] Matrix Loaded: " << filepath << " (M=" << pcm.M << ", N=" << pcm.N << ")" << std::endl;
    return true;
}

static void partition_checks_balanced(
    const std::vector<uint32_t>& check_list,
    const ParityCheckMatrix& pcm,
    uint32_t target_tile_count,
    uint32_t max_tile_size,
    std::vector<std::vector<uint32_t>>& out_tiles
) {
    std::unordered_map<uint32_t, std::vector<uint32_t>> var_to_c;
    for (uint32_t c : check_list) {
        for (uint32_t v : pcm.check_to_var[c]) {
            var_to_c[v].push_back(c);
        }
    }

    std::vector<std::pair<uint32_t, uint32_t>> check_conflicts;
    for (uint32_t c : check_list) {
        std::unordered_set<uint32_t> conflicts;
        for (uint32_t v : pcm.check_to_var[c]) {
            for (uint32_t other : var_to_c[v]) {
                if (other != c) conflicts.insert(other);
            }
        }
        check_conflicts.push_back({c, static_cast<uint32_t>(conflicts.size())});
    }

    std::sort(check_conflicts.begin(), check_conflicts.end(), 
              [](const auto& a, const auto& b) { 
                  if (a.second != b.second) return a.second > b.second;
                  return a.first < b.first; 
              });

    out_tiles.assign(target_tile_count, std::vector<uint32_t>());
    std::vector<std::unordered_set<uint32_t>> tile_vars(target_tile_count);

    for (const auto& item : check_conflicts) {
        uint32_t c = item.first;
        std::unordered_set<uint32_t> c_vars(pcm.check_to_var[c].begin(), pcm.check_to_var[c].end());

        int best_tile = -1;
        size_t min_size = max_tile_size + 1;

        for (uint32_t t = 0; t < target_tile_count; ++t) {
            if (out_tiles[t].size() < max_tile_size) {
                bool conflict = false;
                for (uint32_t v : c_vars) {
                    if (tile_vars[t].count(v)) {
                        conflict = true;
                        break;
                    }
                }
                if (!conflict && out_tiles[t].size() < min_size) {
                    min_size = out_tiles[t].size();
                    best_tile = static_cast<int>(t);
                }
            }
        }

        assert(best_tile != -1 && "Partitioning failed: check cannot be placed without collision!");
        out_tiles[best_tile].push_back(c);
        for (uint32_t v : c_vars) {
            tile_vars[best_tile].insert(v);
        }
    }
}

std::vector<DynamicTileDescriptor> build_circulant_aware_tile_schedule(ParityCheckMatrix& pcm) {
    std::vector<uint32_t> deg6_checks;
    std::vector<uint32_t> deg3_checks;

    for (uint32_t m = 0; m < pcm.M; ++m) {
        if (pcm.check_to_var[m].size() == 6) deg6_checks.push_back(m);
        else if (pcm.check_to_var[m].size() == 3) deg3_checks.push_back(m);
    }

    std::vector<std::vector<uint32_t>> deg6_tiles;
    partition_checks_balanced(deg6_checks, pcm, TILES_PER_DEG6_BLOCK * 2, CHECKS_PER_TILE_DEG6, deg6_tiles);

    std::vector<std::vector<uint32_t>> deg3_tiles;
    partition_checks_balanced(deg3_checks, pcm, TILES_PER_DEG3_BLOCK, CHECKS_PER_TILE_DEG3, deg3_tiles);

    // Proportionally interleave the 32 degree-6 tiles and 8 degree-3 tiles (Total: 40 tiles, 4:1 ratio)
    std::vector<DynamicTileDescriptor> schedule;
    schedule.reserve(TOTAL_TILES_PER_ITER);
    uint16_t current_h_offset = 0;
    uint16_t current_r_page   = 0;

    pcm.h_offsets_pairs.clear();

    auto add_tile_to_schedule = [&](const std::vector<uint32_t>& tile, uint8_t deg) {
        DynamicTileDescriptor desc{};
        desc.check_indices     = tile;
        desc.check_degree      = deg;
        desc.num_checks        = static_cast<uint8_t>(tile.size());
        desc.num_edges         = static_cast<uint16_t>(desc.num_checks * deg);
        desc.h_offset_start    = current_h_offset;
        desc.r_msg_page_offset = current_r_page;

        std::vector<uint32_t> w0_pairs(48);
        std::vector<uint32_t> w1_pairs(48);

        for (uint32_t p = 0; p < 48; ++p) {
            uint32_t d, kh, c0_even, c0_odd, c1_even, c1_odd;
            if (deg == 6) {
                if (p < 24) {
                    d = p / 4;
                    kh = p % 4;
                    // Face 0: checks 0..7 (Worker 0)
                    c0_even = 0 + kh;
                    c0_odd  = 4 + kh;
                    // Face 1: checks 8..15 (Worker 1)
                    c1_even = 8 + kh;
                    c1_odd  = 12 + kh;
                } else {
                    d = (p - 24) / 4;
                    kh = (p - 24) % 4;
                    // Face 2: checks 16..23 (Worker 0)
                    c0_even = 16 + kh;
                    c0_odd  = 20 + kh;
                    // Face 3: checks 24..31 (Worker 1)
                    c1_even = 24 + kh;
                    c1_odd  = 28 + kh;
                }
            } else { // deg == 3 (64 checks: Face 0: 0..15, Face 1: 16..31, Face 2: 32..47, Face 3: 48..63)
                if (p < 24) {
                    uint32_t blk = p / 12; // 0 or 1
                    d  = (p % 12) / 4;    // 0..2
                    kh = (p % 12) % 4;    // 0..3
                    c0_even = 0  + blk * 8 + kh;
                    c0_odd  = 0  + blk * 8 + 4 + kh;
                    c1_even = 16 + blk * 8 + kh;
                    c1_odd  = 16 + blk * 8 + 4 + kh;
                } else {
                    uint32_t blk = (p - 24) / 12; // 0 or 1
                    d  = ((p - 24) % 12) / 4;     // 0..2
                    kh = ((p - 24) % 12) % 4;     // 0..3
                    c0_even = 32 + blk * 8 + kh;
                    c0_odd  = 32 + blk * 8 + 4 + kh;
                    c1_even = 48 + blk * 8 + kh;
                    c1_odd  = 48 + blk * 8 + 4 + kh;
                }
            }

            uint32_t v0_even = (c0_even < tile.size()) ? pcm.check_to_var[tile[c0_even]][d] : 2559;
            uint32_t v0_odd  = (c0_odd  < tile.size()) ? pcm.check_to_var[tile[c0_odd]][d]  : 2559;
            w0_pairs[p] = make_pair_word(v0_even, v0_odd);

            uint32_t v1_even = (c1_even < tile.size()) ? pcm.check_to_var[tile[c1_even]][d] : 2559;
            uint32_t v1_odd  = (c1_odd  < tile.size()) ? pcm.check_to_var[tile[c1_odd]][d]  : 2559;
            w1_pairs[p] = make_pair_word(v1_even, v1_odd);
        }

        // Store Worker 0 pairs (48 pairs = 96 offsets)
        for (uint32_t p = 0; p < 48; ++p) pcm.h_offsets_pairs.push_back(w0_pairs[p]);
        // Store Worker 1 pairs (48 pairs = 96 offsets)
        for (uint32_t p = 0; p < 48; ++p) pcm.h_offsets_pairs.push_back(w1_pairs[p]);

        current_h_offset += 96 * sizeof(uint32_t);
        current_r_page   += 1;
        schedule.push_back(desc);
    };

    // Interleave 32 deg-6 tiles and 8 deg-3 tiles at a 4:1 ratio
    for (size_t i = 0; i < 8; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            add_tile_to_schedule(deg6_tiles[i * 4 + j], 6);
        }
        add_tile_to_schedule(deg3_tiles[i], 3);
    }

    return schedule;
}

inline float to_bf16(float f) {
    return bf16_to_fp32(fp32_to_bf16(f));
}

// Direct 16-bit bitfield truncation matching Tensix PACK unit (pack_tile) and SFPU registers
inline float to_bf16_trunc(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    u &= 0xFFFF0000;
    float res;
    std::memcpy(&res, &u, sizeof(res));
    return res;
}

inline uint16_t fp32_to_bf16_trunc(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return static_cast<uint16_t>(u >> 16);
}

// ------------------------------------------------------------------------------
// Dr. Jones' Branchless Approximate-Min* Formulation
// ------------------------------------------------------------------------------
// Model A: FP32 Precision (standard IEEE-754 24-bit mantissa)
inline float approx_min_star_fp32(float a, float b) {
    float min_v = std::min(a, b);
    float max_v = std::max(a, b);
    float diff  = max_v - min_v;

    static const float c_0_69315   = bf16_to_fp32(16177);
    static const float c_neg_0_4   = bf16_to_fp32(48844);
    static const float c_0_65715   = bf16_to_fp32(16168);
    static const float c_neg_0_328 = bf16_to_fp32(48807);

    float line1 = c_0_69315 + c_neg_0_4 * diff;
    float line2 = c_0_65715 + c_neg_0_328 * diff;
    float corr  = std::max(std::max(line1, line2), 0.0f);
    return std::max(min_v - corr, 0.0f);
}

// Model B: Tensix Hardware Bit-Exact Precision (Native bfloat16 SFPU emulation)
inline float approx_min_star_tensix(float a, float b) {
    float min_v = std::min(a, b);
    float max_v = std::max(a, b);
    float diff  = max_v - min_v;

    static const float c_0_69315   = bf16_to_fp32(16177); // 0.69315 (sFloat16b)
    static const float c_neg_0_4   = bf16_to_fp32(48844); // -0.4000 (sFloat16b)
    static const float c_0_65715   = bf16_to_fp32(16168); // 0.65715 (sFloat16b)
    static const float c_neg_0_328 = bf16_to_fp32(48807); // -0.3280 (sFloat16b)

    float line1 = c_0_69315 + c_neg_0_4 * diff;
    float line2 = c_0_65715 + c_neg_0_328 * diff;
    float corr  = std::max(std::max(line1, line2), 0.0f);
    return std::max(min_v - corr, 0.0f);
}

inline void init_vnodes_host(
    uint8_t* vnode_l1_ptr,
    float mu_llr,
    float sigma_llr,
    Xoshiro128Plus& rng
) {
    bool bench = (sigma_llr <= 0.0f);
    uint16_t mu_bf = fp32_to_bf16(mu_llr);

    // 1. Transmitted variable nodes v in [0..2047] (Tiles 0..7)
    for (uint32_t v = 0; v < 2048; ++v) {
        uint32_t off = get_tile_vnode_byte_offset(v);
        uint16_t* p = reinterpret_cast<uint16_t*>(vnode_l1_ptr + off);
        if (bench) {
            p[0] = mu_bf;
            p[1] = mu_bf;
            p[2] = mu_bf;
            p[3] = mu_bf;
        } else {
            float l0, l1, l2, l3;
            generate_awgn_pair(rng, mu_llr, sigma_llr, l0, l1);
            generate_awgn_pair(rng, mu_llr, sigma_llr, l2, l3);
            p[0] = fp32_to_bf16(l0);
            p[1] = fp32_to_bf16(l1);
            p[2] = fp32_to_bf16(l2);
            p[3] = fp32_to_bf16(l3);
        }
    }

    // 2. Punctured variable nodes v in [2048..2559] (Tiles 8 & 9: 4,096 bytes = 512 variables x 8 bytes)
    std::memset(vnode_l1_ptr + 8 * 2048, 0, 4096);
}

// ------------------------------------------------------------------------------
// Hardware-Accurate 4-Codeword 40-Tile Batch Decoder
// ------------------------------------------------------------------------------
struct BatchStats {
    uint32_t bit_errors[4] = {0, 0, 0, 0};
    uint32_t iters_taken[4] = {0, 0, 0, 0};
    bool converged[4] = {false, false, false, false};
};

inline BatchStats decode_tile_matrix_batch(
    const ParityCheckMatrix& pcm,
    const std::vector<DynamicTileDescriptor>& schedule,
    uint8_t* vnode_l1,
    std::vector<std::vector<uint16_t>>& r_msg_fifo,
    std::vector<uint16_t>& scratch_in,
    std::vector<uint16_t>& scratch_out,
    uint32_t max_iter,
    float post_clip_max = 0.0f,
    bool tensix_precision = true
) {
    BatchStats stats;
    uint8_t active_mask = 0x0F;
    const uint32_t* p_hoff_base = pcm.h_offsets_pairs.data();

    for (uint32_t iter = 0; iter < max_iter; ++iter) {
        for (size_t t = 0; t < schedule.size(); ++t) {
            uint32_t deg = schedule[t].check_degree;
            const uint32_t* tile_hoff_w0 = p_hoff_base + t * 96;
            const uint32_t* tile_hoff_w1 = tile_hoff_w0 + 48;

            // 1. Dual Gather Stage (Worker 0: Faces 0 & 2, Worker 1: Faces 1 & 3)
            for (int p = 0; p < 48; ++p) {
                // Worker 0
                uint32_t off0 = tile_hoff_w0[p];
                uint32_t off0_even = off0 & 0xFFFF;
                uint32_t off0_odd  = off0 >> 16;
                uint32_t s0_even = SCRATCH_FACE_LUT.w0_even[p];
                uint32_t s0_odd  = SCRATCH_FACE_LUT.w0_odd[p];

                *reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(scratch_in.data()) + s0_even) =
                    *reinterpret_cast<const uint64_t*>(vnode_l1 + off0_even);
                *reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(scratch_in.data()) + s0_odd) =
                    *reinterpret_cast<const uint64_t*>(vnode_l1 + off0_odd);

                // Worker 1
                uint32_t off1 = tile_hoff_w1[p];
                uint32_t off1_even = off1 & 0xFFFF;
                uint32_t off1_odd  = off1 >> 16;
                uint32_t s1_even = SCRATCH_FACE_LUT.w1_even[p];
                uint32_t s1_odd  = SCRATCH_FACE_LUT.w1_odd[p];

                *reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(scratch_in.data()) + s1_even) =
                    *reinterpret_cast<const uint64_t*>(vnode_l1 + off1_even);
                *reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(scratch_in.data()) + s1_odd) =
                    *reinterpret_cast<const uint64_t*>(vnode_l1 + off1_odd);
            }

            // 2. Systolic TRISC Check Node Compute (32 Lanes in Parallel across 4 Faces)
            uint32_t num_passes = (deg == 6) ? 4 : 8;
            const int deg6_face_bases[4] = {0, 8, 16, 24};
            const int deg3_face_bases[8] = {0, 3, 8, 11, 16, 19, 24, 27};

            for (uint32_t pass = 0; pass < num_passes; ++pass) {
                uint32_t base_r = (deg == 6) ? deg6_face_bases[pass] : deg3_face_bases[pass];

                for (uint32_t lane = 0; lane < 32; ++lane) {
                    uint32_t col = lane;
                    float q[8], signs[8], mags[8];
                    float total_sign = 1.0f;
                    uint32_t min_idx = 0;
                    float min_mag = 99999.0f;

                    for (uint32_t r = 0; r < deg; ++r) {
                        uint32_t offset = get_scratch_byte_offset(base_r + r, col);
                        float l_val = bf16_to_fp32(*reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(scratch_in.data()) + offset));
                        float r_old = (iter == 0) ? 0.0f : bf16_to_fp32(*reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(r_msg_fifo[t].data()) + offset));

                        q[r] = tensix_precision ? to_bf16(l_val - r_old) : (l_val - r_old);
                        signs[r] = (q[r] < 0.0f) ? -1.0f : 1.0f;
                        mags[r]  = std::fabs(q[r]);
                        total_sign *= signs[r];
                        if (mags[r] < min_mag) {
                            min_mag = mags[r];
                            min_idx = r;
                        }
                    }

                    // True Recursive Approximate-Min* Check Update
                    float m_others = (min_idx == 0) ? 1000.0f : mags[0];
                    if (tensix_precision) {
                        for (uint32_t r = 1; r < deg; ++r) {
                            float m = (min_idx == r) ? 1000.0f : mags[r];
                            m_others = approx_min_star_tensix(m_others, m);
                        }
                        float m_all = approx_min_star_tensix(m_others, min_mag);

                        // Scatter out to Tile 0 (R_new) and Tile 1 (L_new)
                        for (uint32_t r = 0; r < deg; ++r) {
                            uint32_t offset = get_scratch_byte_offset(base_r + r, col);
                            float edge_sign = total_sign * signs[r];
                            float edge_mag = (r == min_idx) ? m_others : m_all;
                            float r_raw = edge_sign * edge_mag;
                            float l_new = to_bf16(q[r] + r_raw);
                            float r_new = to_bf16(r_raw);

                            if (post_clip_max > 0.0f) {
                                if (l_new > post_clip_max) l_new = post_clip_max;
                                if (l_new < -post_clip_max) l_new = -post_clip_max;
                            }

                            *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(r_msg_fifo[t].data()) + offset) = fp32_to_bf16(r_new);
                            *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(scratch_out.data()) + offset)   = fp32_to_bf16(l_new);
                        }
                    } else {
                        for (uint32_t r = 1; r < deg; ++r) {
                            float m = (min_idx == r) ? 1000.0f : mags[r];
                            m_others = approx_min_star_fp32(m_others, m);
                        }
                        float m_all = approx_min_star_fp32(m_others, min_mag);

                        // Scatter out to Tile 0 (R_new) and Tile 1 (L_new)
                        for (uint32_t r = 0; r < deg; ++r) {
                            uint32_t offset = get_scratch_byte_offset(base_r + r, col);
                            float edge_sign = total_sign * signs[r];
                            float edge_mag = (r == min_idx) ? m_others : m_all;
                            float r_new = edge_sign * edge_mag;
                            float l_new = q[r] + r_new;

                            if (post_clip_max > 0.0f) {
                                if (l_new > post_clip_max) l_new = post_clip_max;
                                if (l_new < -post_clip_max) l_new = -post_clip_max;
                            }

                            *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(r_msg_fifo[t].data()) + offset) = fp32_to_bf16(r_new);
                            *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(scratch_out.data()) + offset)   = fp32_to_bf16(l_new);
                        }
                    }
                }
            }

            // 3. Dual Scatter Stage (Worker 0 & Worker 1 in parallel)
            for (int p = 0; p < 48; ++p) {
                // Worker 0
                uint32_t off0 = tile_hoff_w0[p];
                uint32_t off0_even = off0 & 0xFFFF;
                uint32_t off0_odd  = off0 >> 16;
                uint32_t s0_even = SCRATCH_FACE_LUT.w0_even[p];
                uint32_t s0_odd  = SCRATCH_FACE_LUT.w0_odd[p];

                *reinterpret_cast<uint64_t*>(vnode_l1 + off0_even) =
                    *reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(scratch_out.data()) + s0_even);
                *reinterpret_cast<uint64_t*>(vnode_l1 + off0_odd) =
                    *reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(scratch_out.data()) + s0_odd);

                // Worker 1
                uint32_t off1 = tile_hoff_w1[p];
                uint32_t off1_even = off1 & 0xFFFF;
                uint32_t off1_odd  = off1 >> 16;
                uint32_t s1_even = SCRATCH_FACE_LUT.w1_even[p];
                uint32_t s1_odd  = SCRATCH_FACE_LUT.w1_odd[p];

                *reinterpret_cast<uint64_t*>(vnode_l1 + off1_even) =
                    *reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(scratch_out.data()) + s1_even);
                *reinterpret_cast<uint64_t*>(vnode_l1 + off1_odd) =
                    *reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(scratch_out.data()) + s1_odd);
            }
        }

        // Parity Check Evaluation for Active Codewords (Early Check Exit)
        for (int c = 0; c < 4; ++c) {
            if (!(active_mask & (1 << c))) continue;

            bool satisfied = true;
            for (uint32_t m = 0; m < pcm.M; ++m) {
                int syn = 0;
                for (uint32_t off : pcm.check_to_var_offsets[m]) {
                    const uint16_t* p = reinterpret_cast<const uint16_t*>(vnode_l1 + off);
                    if (bf16_to_fp32(p[c]) < 0.0f) syn ^= 1;
                }
                if (syn != 0) {
                    satisfied = false;
                    break;
                }
            }

            if (satisfied) {
                stats.converged[c] = true;
                stats.iters_taken[c] = iter + 1;
                active_mask &= ~(1 << c);
            }
        }

        if (active_mask == 0) break;
    }

    // Residual Bit Error Check on Information Bits (v < K)
    for (int c = 0; c < 4; ++c) {
        if (!stats.converged[c]) {
            stats.iters_taken[c] = max_iter;
            for (uint32_t off : pcm.info_var_offsets) {
                const uint16_t* p = reinterpret_cast<const uint16_t*>(vnode_l1 + off);
                if (bf16_to_fp32(p[c]) < 0.0f) stats.bit_errors[c]++;
            }
        }
    }

    return stats;
}

// ------------------------------------------------------------------------------
// Multi-Threaded Host Monte Carlo Waterfall Engine (OpenMP Parallel)
// ------------------------------------------------------------------------------
int run_monte_carlo_simulation(
    int argc,
    char** argv,
    const ParityCheckMatrix& pcm,
    const std::vector<DynamicTileDescriptor>& schedule
) {
    double eb_n0_db = (argc > 2) ? std::stod(argv[2]) : 2.0;
    uint64_t max_blocks = (argc > 3) ? std::stoull(argv[3]) : 10000000ULL;
    uint64_t min_frame_errors = (argc > 4) ? std::stoull(argv[4]) : 25ULL;
    uint32_t max_iter = (argc > 5) ? std::stoul(argv[5]) : 200U;
    int num_threads = (argc > 6) ? std::stoi(argv[6]) : 16;
    float post_clip = (argc > 7) ? std::stof(argv[7]) : 0.0f;
    std::string csv_path = (argc > 8) ? argv[8] : "Results/results_40tile_lockstep_ebn0_2.00_10M.csv";

    if (num_threads <= 0) num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);

    bool tensix_precision = true;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--fp32" || std::string(argv[i]) == "--full-precision") {
            tensix_precision = false;
        } else if (std::string(argv[i]) == "--tensix-precision") {
            tensix_precision = true;
        }
    }

    std::cout << "\n=================================================================" << std::endl;
    std::cout << " TENSIX 40-TILE FACE-ALIGNED LOCKSTEP STEPPER: MONTE CARLO" << std::endl;
    std::cout << " Precision Mode:    " << (tensix_precision ? "Tensix Hardware Bit-Exact (bfloat16 mantissa truncation)" : "Full FP32 Precision") << std::endl;
    std::cout << " Eb/N0:             " << std::fixed << std::setprecision(2) << eb_n0_db << " dB" << std::endl;
    std::cout << " Max Blocks:        " << max_blocks << std::endl;
    std::cout << " Min Frame Errors:  " << min_frame_errors << std::endl;
    std::cout << " Max Iterations:    " << max_iter << std::endl;
    std::cout << " Post-Clipping:     " << (post_clip > 0.0f ? std::to_string(post_clip) : "None (Unclipped)") << std::endl;
    std::cout << " Host Threads:      " << num_threads << " (OpenMP Parallel)" << std::endl;
    std::cout << " CSV Target:        " << csv_path << std::endl;
    std::cout << "=================================================================\n" << std::endl;

    float code_rate = static_cast<float>(pcm.K) / (pcm.N - pcm.P); // 1024 / 2048 = 0.5
    float eb_n0_lin = std::pow(10.0f, static_cast<float>(eb_n0_db) / 10.0f);
    float es_n0 = code_rate * eb_n0_lin;
    float noise_var = 1.0f / (2.0f * es_n0);
    float noise_std = std::sqrt(noise_var);
    float mu_llr = 2.0f / noise_var;
    float sigma_llr = 2.0f / noise_std;

    std::atomic<uint64_t> total_codewords{0};
    std::atomic<uint64_t> total_bit_errors{0};
    std::atomic<uint64_t> total_frame_errors{0};
    std::atomic<uint64_t> total_cw_iterations{0};
    std::atomic<bool> stop_flag{false};

    auto t_start = std::chrono::high_resolution_clock::now();

    // Background Monitor Thread
    std::thread monitor([&]() {
        uint64_t last_fe = 0;
        uint64_t last_blocks = 0;
        auto last_print_time = std::chrono::high_resolution_clock::now();
        bool is_tty = (ISATTY(FILENO(stdout)) != 0);

        while (!stop_flag.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2000));
            if (stop_flag.load(std::memory_order_relaxed)) break;

            uint64_t cur_cws = total_codewords.load(std::memory_order_relaxed);
            uint64_t cur_fe = total_frame_errors.load(std::memory_order_relaxed);
            uint64_t cur_be = total_bit_errors.load(std::memory_order_relaxed);
            uint64_t cur_iters = total_cw_iterations.load(std::memory_order_relaxed);

            auto now = std::chrono::high_resolution_clock::now();
            double elapsed_s = std::chrono::duration<double>(now - t_start).count();
            if (elapsed_s < 0.1 || cur_cws == 0) continue;

            double fer = static_cast<double>(cur_fe) / static_cast<double>(cur_cws);
            double ber = static_cast<double>(cur_be) / (static_cast<double>(cur_cws) * pcm.K);
            double avg_iter = static_cast<double>(cur_iters) / static_cast<double>(cur_cws);
            double msps = (static_cast<double>(cur_cws) * (pcm.N - pcm.P)) / (elapsed_s * 1e6);

            if (is_tty) {
                std::cout << "\r[" << std::fixed << std::setprecision(2) << eb_n0_db << " dB] "
                          << "Blocks: " << cur_cws << " / " << max_blocks << " | "
                          << "FE: " << cur_fe << " / " << min_frame_errors << " | "
                          << "FER: " << std::scientific << std::setprecision(3) << fer << " | "
                          << "BER: " << std::scientific << std::setprecision(3) << ber << " | "
                          << "AvgIter: " << std::fixed << std::setprecision(2) << avg_iter << " | "
                          << "Speed: " << std::fixed << std::setprecision(2) << msps << " Msps"
                          << std::flush;
            } else {
                double time_since_last_print = std::chrono::duration<double>(now - last_print_time).count();
                if (cur_fe > last_fe || cur_cws - last_blocks >= 100000 || time_since_last_print >= 5.0) {
                    last_fe = cur_fe;
                    last_blocks = cur_cws;
                    last_print_time = now;
                    std::cout << "[" << std::fixed << std::setprecision(2) << eb_n0_db << " dB] "
                              << "Blocks: " << cur_cws << " / " << max_blocks << " | "
                              << "FE: " << cur_fe << " / " << min_frame_errors << " | "
                              << "FER: " << std::scientific << std::setprecision(3) << fer << " | "
                              << "BER: " << std::scientific << std::setprecision(3) << ber << " | "
                              << "AvgIter: " << std::fixed << std::setprecision(2) << avg_iter << " | "
                              << "Speed: " << std::fixed << std::setprecision(2) << msps << " Msps"
                              << std::endl;
                }
            }
        }
    });

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        Xoshiro128Plus rng;
        uint32_t seed = 1234567U + static_cast<uint32_t>(eb_n0_db * 100.0) * 10007U;
        rng.init(seed);
        for (int i = 0; i < tid; ++i) rng.jump();

        std::vector<uint8_t> vnode_l1(20480, 0);
        std::vector<std::vector<uint16_t>> r_msg_fifo(schedule.size(), std::vector<uint16_t>(1024, 0));
        std::vector<uint16_t> scratch_in(1024, 0);
        std::vector<uint16_t> scratch_out(1024, 0);

        uint32_t chunk_batches = (eb_n0_db < 1.7) ? 25 : std::min(250U, std::max(1U, static_cast<uint32_t>((max_blocks + 3) / 4 / num_threads + 1)));

        while (!stop_flag.load(std::memory_order_relaxed)) {
            uint32_t local_fe = 0;
            uint32_t local_be = 0;
            uint64_t local_cw_iters = 0;

            for (uint32_t b = 0; b < chunk_batches; ++b) {
                for (auto& fifo : r_msg_fifo) {
                    std::fill(fifo.begin(), fifo.end(), 0);
                }
                init_vnodes_host(vnode_l1.data(), mu_llr, sigma_llr, rng);
                BatchStats stats = decode_tile_matrix_batch(
                    pcm, schedule, vnode_l1.data(), r_msg_fifo, scratch_in, scratch_out, max_iter, post_clip, tensix_precision
                );

                for (int c = 0; c < 4; ++c) {
                    local_cw_iters += stats.iters_taken[c];
                    if (stats.bit_errors[c] > 0 || !stats.converged[c]) {
                        local_fe++;
                        local_be += stats.bit_errors[c];
                    }
                }
            }

            uint64_t cur_cws = total_codewords.fetch_add(chunk_batches * 4, std::memory_order_relaxed) + chunk_batches * 4;
            uint64_t cur_fe = total_frame_errors.fetch_add(local_fe, std::memory_order_relaxed) + local_fe;
            total_bit_errors.fetch_add(local_be, std::memory_order_relaxed);
            total_cw_iterations.fetch_add(local_cw_iters, std::memory_order_relaxed);

            if (cur_fe >= min_frame_errors || cur_cws >= max_blocks) {
                stop_flag.store(true, std::memory_order_relaxed);
            }
        }
    }

    stop_flag.store(true);
    if (monitor.joinable()) monitor.join();

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();

    uint64_t final_cws = total_codewords.load();
    uint64_t final_fe = total_frame_errors.load();
    uint64_t final_be = total_bit_errors.load();
    uint64_t final_iters = total_cw_iterations.load();

    double final_fer = (final_cws > 0) ? (static_cast<double>(final_fe) / static_cast<double>(final_cws)) : 0.0;
    double final_ber = (final_cws > 0) ? (static_cast<double>(final_be) / (static_cast<double>(final_cws) * pcm.K)) : 0.0;
    double mean_iter = (final_cws > 0) ? (static_cast<double>(final_iters) / static_cast<double>(final_cws)) : 0.0;
    double throughput_msps = (final_cws * (pcm.N - pcm.P)) / (total_sec * 1e6);

    std::cout << "\n\n=================================================================" << std::endl;
    std::cout << " MONTE CARLO SIMULATION COMPLETED" << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << " Eb/N0:             " << std::fixed << std::setprecision(2) << eb_n0_db << " dB" << std::endl;
    std::cout << " Total Blocks:      " << final_cws << std::endl;
    std::cout << " Total Frame Errs:  " << final_fe << std::endl;
    std::cout << " Total Bit Errs:    " << final_be << std::endl;
    std::cout << " FER:               " << std::scientific << std::setprecision(6) << final_fer << std::endl;
    std::cout << " BER:               " << std::scientific << std::setprecision(6) << final_ber << std::endl;
    std::cout << " Mean Iterations:   " << std::fixed << std::setprecision(3) << mean_iter << std::endl;
    std::cout << " Host Throughput:   " << std::fixed << std::setprecision(2) << throughput_msps << " Msps" << std::endl;
    std::cout << " Wall-Clock Time:   " << std::fixed << std::setprecision(2) << total_sec << " s" << std::endl;
    std::cout << "=================================================================" << std::endl;

    fs::create_directories(fs::path(csv_path).parent_path());
    bool file_exists = fs::exists(csv_path);
    std::ofstream csv(csv_path, std::ios::app);
    if (csv.is_open()) {
        if (!file_exists) {
            csv << "EbN0_dB,Total_Blocks,Total_Bit_Errors,Total_Block_Errors,BER,FER,Mean_Iterations,Throughput_Msps,Host_Wall_s\n";
        }
        csv << std::fixed << std::setprecision(2) << eb_n0_db << ","
            << final_cws << ","
            << final_be << ","
            << final_fe << ","
            << std::scientific << std::setprecision(6) << final_ber << ","
            << std::scientific << std::setprecision(6) << final_fer << ","
            << std::fixed << std::setprecision(3) << mean_iter << ","
            << std::fixed << std::setprecision(2) << throughput_msps << ","
            << std::fixed << std::setprecision(2) << total_sec << "\n";
        csv.close();
        std::cout << "[INFO] Results saved to: " << csv_path << std::endl;
    } else {
        std::cerr << "[ERROR] Could not open CSV path: " << csv_path << std::endl;
    }

    return 0;
}

// ==============================================================================
// Main Lockstep Stepper Runner
// ==============================================================================
int main(int argc, char** argv) {
    std::string matrix_path = "matrices/AR4JA_r45_4c_128c_r12.chinn.out";
    if (!fs::exists(matrix_path)) matrix_path = "../../matrices/AR4JA_r45_4c_128c_r12.chinn.out";
    if (!fs::exists(matrix_path)) matrix_path = "../matrices/AR4JA_r45_4c_128c_r12.chinn.out";

    ParityCheckMatrix pcm;
    if (!parse_chinn_out(matrix_path, pcm)) {
        return 1;
    }

    auto schedule = build_circulant_aware_tile_schedule(pcm);
    std::cout << "[INFO] Scheduled " << schedule.size() << " face-aligned tiles (32 Deg-6 + 8 Deg-3)." << std::endl;

    if (argc > 1 && std::string(argv[1]) == "--monte-carlo") {
        return run_monte_carlo_simulation(argc, argv, pcm, schedule);
    }

    std::string tensix_init_path  = "engines/tile_matrix_4cw/dump/tensix_initial_vnodes.bin";
    std::string tensix_final_path = "engines/tile_matrix_4cw/dump/tensix_final_vnodes.bin";
    if (!fs::exists(tensix_init_path) && fs::exists("dump/tensix_initial_vnodes.bin")) {
        tensix_init_path  = "dump/tensix_initial_vnodes.bin";
        tensix_final_path = "dump/tensix_final_vnodes.bin";
    }

    bool standalone_mode = false;
    double eb_n0_db = 2.5;
    uint32_t num_iterations = 16;
    uint32_t seed = 12345;

    if (argc > 1 && std::string(argv[1]) == "--standalone") {
        standalone_mode = true;
        if (argc > 2) eb_n0_db = std::stod(argv[2]);
        if (argc > 3) num_iterations = std::stoul(argv[3]);
        if (argc > 4) seed = std::stoul(argv[4]);
    } else {
        if (argc > 1) num_iterations = std::stoul(argv[1]);
        if (argc > 2) tensix_init_path = argv[2];
        if (argc > 3) tensix_final_path = argv[3];
    }

    bool tensix_precision = true;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--fp32" || std::string(argv[i]) == "--full-precision") {
            tensix_precision = false;
        } else if (std::string(argv[i]) == "--tensix-precision") {
            tensix_precision = true;
        }
    }

    std::cout << "=================================================================" << std::endl;
    std::cout << " TENSIX 40-TILE FACE-ALIGNED LOCKSTEP STEPPER" << std::endl;
    std::cout << " Precision Mode:    " << (tensix_precision ? "Tensix Hardware Bit-Exact (bfloat16 mantissa truncation)" : "Full FP32 Precision") << std::endl;
    if (standalone_mode) {
        std::cout << " Mode:              STANDALONE CONVERGENCE VERIFICATION" << std::endl;
        std::cout << " Eb/N0:             " << eb_n0_db << " dB" << std::endl;
        std::cout << " Max Iterations:    " << num_iterations << std::endl;
        std::cout << " PRNG Seed:         " << seed << std::endl;
    } else {
        std::cout << " Mode:              SILICON REPLAY CO-VERIFICATION" << std::endl;
        std::cout << " Target Iterations: " << num_iterations << std::endl;
        std::cout << " Initial Dump:      " << tensix_init_path << std::endl;
        std::cout << " Tensix Final Dump: " << tensix_final_path << std::endl;
    }
    std::cout << "=================================================================" << std::endl;

    std::vector<uint8_t> vnode_l1(20480, 0);

    if (standalone_mode) {
        float code_rate = static_cast<float>(pcm.K) / (pcm.N - pcm.P); // 1024 / 2048 = 0.5
        float eb_n0_lin = std::pow(10.0f, static_cast<float>(eb_n0_db) / 10.0f);
        float es_n0 = code_rate * eb_n0_lin;
        float noise_var = 1.0f / (2.0f * es_n0);
        float noise_std = std::sqrt(noise_var);
        float mu_llr = 2.0f / noise_var;
        float sigma_llr = 2.0f / noise_std;

        Xoshiro128Plus rng;
        rng.init(seed);
        init_vnodes_host(vnode_l1.data(), mu_llr, sigma_llr, rng);
        std::cout << "[INFO] Synthesized AWGN channel LLRs for 4 codewords at " << eb_n0_db << " dB (mu=" << mu_llr << ", sigma=" << sigma_llr << ")" << std::endl;
    } else {
        std::ifstream f_init(tensix_init_path, std::ios::binary);
        if (!f_init.is_open()) {
            std::cerr << "[ERROR] Cannot open initial dump: " << tensix_init_path << std::endl;
            return 1;
        }
        f_init.read(reinterpret_cast<char*>(vnode_l1.data()), 20480);
        f_init.close();
        std::cout << "[INFO] Loaded 20,480 bytes from " << tensix_init_path << std::endl;
    }

    // 40-Page Circular FIFO for Extrinsic Check Messages (CB_R_MSG)
    std::vector<std::vector<uint16_t>> r_msg_fifo(schedule.size(), std::vector<uint16_t>(1024, 0));

    // Scratchpad Tile Buffers
    std::vector<uint16_t> scratch_in(1024, 0);
    std::vector<uint16_t> scratch_out(1024, 0);

    const uint32_t* p_hoff_base = pcm.h_offsets_pairs.data();

    // Print Initial Error State before decoding
    {
        uint32_t init_info_errs[4] = {0, 0, 0, 0};
        uint32_t init_total_neg[4] = {0, 0, 0, 0};
        for (uint32_t v = 0; v < 2048; ++v) {
            uint32_t off = get_tile_vnode_byte_offset(v);
            const uint16_t* p = reinterpret_cast<const uint16_t*>(vnode_l1.data() + off);
            for (int c = 0; c < 4; ++c) {
                if (bf16_to_fp32(p[c]) < 0.0f) {
                    init_total_neg[c]++;
                    if (v < 1024) init_info_errs[c]++;
                }
            }
        }
        std::cout << "[Initial State] Info BE: [" << init_info_errs[0] << ", " << init_info_errs[1] << ", " << init_info_errs[2] << ", " << init_info_errs[3] << "] | "
                  << "Tx Neg: [" << init_total_neg[0] << ", " << init_total_neg[1] << ", " << init_total_neg[2] << ", " << init_total_neg[3] << "]"
                  << std::endl;
    }

    // --------------------------------------------------------------------------
    // Iteration Loop: Executes the exact 40-tile face-aligned hardware pipeline
    // --------------------------------------------------------------------------
    for (uint32_t iter = 0; iter < num_iterations; ++iter) {
        for (size_t t = 0; t < schedule.size(); ++t) {
            uint32_t deg = schedule[t].check_degree;
            const uint32_t* tile_hoff_w0 = p_hoff_base + t * 96;
            const uint32_t* tile_hoff_w1 = tile_hoff_w0 + 48;

            // 1. Dual Gather Stage (Worker 0: Faces 0 & 2, Worker 1: Faces 1 & 3)
            for (int p = 0; p < 48; ++p) {
                // Worker 0
                uint32_t off0 = tile_hoff_w0[p];
                uint32_t off0_even = off0 & 0xFFFF;
                uint32_t off0_odd  = off0 >> 16;
                uint32_t s0_even = SCRATCH_FACE_LUT.w0_even[p];
                uint32_t s0_odd  = SCRATCH_FACE_LUT.w0_odd[p];

                *reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(scratch_in.data()) + s0_even) =
                    *reinterpret_cast<const uint64_t*>(vnode_l1.data() + off0_even);
                *reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(scratch_in.data()) + s0_odd) =
                    *reinterpret_cast<const uint64_t*>(vnode_l1.data() + off0_odd);

                // Worker 1
                uint32_t off1 = tile_hoff_w1[p];
                uint32_t off1_even = off1 & 0xFFFF;
                uint32_t off1_odd  = off1 >> 16;
                uint32_t s1_even = SCRATCH_FACE_LUT.w1_even[p];
                uint32_t s1_odd  = SCRATCH_FACE_LUT.w1_odd[p];

                *reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(scratch_in.data()) + s1_even) =
                    *reinterpret_cast<const uint64_t*>(vnode_l1.data() + off1_even);
                *reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(scratch_in.data()) + s1_odd) =
                    *reinterpret_cast<const uint64_t*>(vnode_l1.data() + off1_odd);
            }

            // 2. Systolic TRISC Check Node Compute (32 Lanes across 4 Faces)
            uint32_t num_passes = (deg == 6) ? 4 : 8;
            const int deg6_face_bases[4] = {0, 8, 16, 24};
            const int deg3_face_bases[8] = {0, 3, 8, 11, 16, 19, 24, 27};

            for (uint32_t pass = 0; pass < num_passes; ++pass) {
                uint32_t base_r = (deg == 6) ? deg6_face_bases[pass] : deg3_face_bases[pass];

                for (uint32_t lane = 0; lane < 32; ++lane) {
                    uint32_t col = lane;
                    float q[8], signs[8], mags[8];
                    float total_sign = 1.0f;
                    uint32_t min_idx = 0;
                    float min_mag = 99999.0f;

                    for (uint32_t r = 0; r < deg; ++r) {
                        uint32_t offset = get_scratch_byte_offset(base_r + r, col);
                        float l_val = bf16_to_fp32(*reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(scratch_in.data()) + offset));
                        float r_old = (iter == 0) ? 0.0f : bf16_to_fp32(*reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(r_msg_fifo[t].data()) + offset));

                        q[r] = tensix_precision ? to_bf16(l_val - r_old) : (l_val - r_old);
                        signs[r] = (q[r] < 0.0f) ? -1.0f : 1.0f;
                        mags[r]  = std::fabs(q[r]);
                        total_sign *= signs[r];
                        if (mags[r] < min_mag) {
                            min_mag = mags[r];
                            min_idx = r;
                        }
                    }

                    // True Recursive Approximate-Min* Check Update
                    float m_others = (min_idx == 0) ? 1000.0f : mags[0];
                    if (tensix_precision) {
                        for (uint32_t r = 1; r < deg; ++r) {
                            float m = (min_idx == r) ? 1000.0f : mags[r];
                            m_others = approx_min_star_tensix(m_others, m);
                        }
                        float m_all = approx_min_star_tensix(m_others, min_mag);

                        // Scatter out to Tile 0 (R_new) and Tile 1 (L_new)
                        for (uint32_t r = 0; r < deg; ++r) {
                            uint32_t offset = get_scratch_byte_offset(base_r + r, col);
                            float edge_sign = total_sign * signs[r];
                            float edge_mag = (r == min_idx) ? m_others : m_all;
                            float r_raw = edge_sign * edge_mag;
                            float l_new = to_bf16(q[r] + r_raw);
                            float r_new = to_bf16(r_raw);

                            *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(r_msg_fifo[t].data()) + offset) = fp32_to_bf16(r_new);
                            *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(scratch_out.data()) + offset)   = fp32_to_bf16(l_new);
                        }
                    } else {
                        for (uint32_t r = 1; r < deg; ++r) {
                            float m = (min_idx == r) ? 1000.0f : mags[r];
                            m_others = approx_min_star_fp32(m_others, m);
                        }
                        float m_all = approx_min_star_fp32(m_others, min_mag);

                        // Scatter out to Tile 0 (R_new) and Tile 1 (L_new)
                        for (uint32_t r = 0; r < deg; ++r) {
                            uint32_t offset = get_scratch_byte_offset(base_r + r, col);
                            float edge_sign = total_sign * signs[r];
                            float edge_mag = (r == min_idx) ? m_others : m_all;
                            float r_new = edge_sign * edge_mag;
                            float l_new = q[r] + r_new;

                            *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(r_msg_fifo[t].data()) + offset) = fp32_to_bf16(r_new);
                            *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(scratch_out.data()) + offset)   = fp32_to_bf16(l_new);
                        }
                    }
                }
            }

            // 3. Dual Scatter Stage (Worker 0 & Worker 1 in parallel)
            for (int p = 0; p < 48; ++p) {
                // Worker 0
                uint32_t off0 = tile_hoff_w0[p];
                uint32_t off0_even = off0 & 0xFFFF;
                uint32_t off0_odd  = off0 >> 16;
                uint32_t s0_even = SCRATCH_FACE_LUT.w0_even[p];
                uint32_t s0_odd  = SCRATCH_FACE_LUT.w0_odd[p];

                *reinterpret_cast<uint64_t*>(vnode_l1.data() + off0_even) =
                    *reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(scratch_out.data()) + s0_even);
                *reinterpret_cast<uint64_t*>(vnode_l1.data() + off0_odd) =
                    *reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(scratch_out.data()) + s0_odd);

                // Worker 1
                uint32_t off1 = tile_hoff_w1[p];
                uint32_t off1_even = off1 & 0xFFFF;
                uint32_t off1_odd  = off1 >> 16;
                uint32_t s1_even = SCRATCH_FACE_LUT.w1_even[p];
                uint32_t s1_odd  = SCRATCH_FACE_LUT.w1_odd[p];

                *reinterpret_cast<uint64_t*>(vnode_l1.data() + off1_even) =
                    *reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(scratch_out.data()) + s1_even);
                *reinterpret_cast<uint64_t*>(vnode_l1.data() + off1_odd) =
                    *reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(scratch_out.data()) + s1_odd);
            }
        }

        // Iteration Statistics
        uint32_t info_errs[4] = {0, 0, 0, 0};
        uint32_t total_neg[4] = {0, 0, 0, 0};
        for (uint32_t v = 0; v < 2048; ++v) {
            uint32_t off = get_tile_vnode_byte_offset(v);
            const uint16_t* p = reinterpret_cast<const uint16_t*>(vnode_l1.data() + off);
            for (int c = 0; c < 4; ++c) {
                if (bf16_to_fp32(p[c]) < 0.0f) {
                    total_neg[c]++;
                    if (v < 1024) info_errs[c]++;
                }
            }
        }

        uint32_t unsat_checks[4] = {0, 0, 0, 0};
        for (uint32_t m = 0; m < pcm.M; ++m) {
            int syn[4] = {0, 0, 0, 0};
            for (uint32_t v : pcm.check_to_var[m]) {
                uint32_t off = get_tile_vnode_byte_offset(v);
                const uint16_t* p = reinterpret_cast<const uint16_t*>(vnode_l1.data() + off);
                for (int c = 0; c < 4; ++c) {
                    if (bf16_to_fp32(p[c]) < 0.0f) syn[c] ^= 1;
                }
            }
            for (int c = 0; c < 4; ++c) {
                if (syn[c] != 0) unsat_checks[c]++;
            }
        }

        std::cout << "[Iter " << std::setw(2) << (iter + 1) << "] "
                  << "Info BE: [" << info_errs[0] << ", " << info_errs[1] << ", " << info_errs[2] << ", " << info_errs[3] << "] | "
                  << "Tx Neg: [" << total_neg[0] << ", " << total_neg[1] << ", " << total_neg[2] << ", " << total_neg[3] << "] | "
                  << "Unsat Checks: [" << unsat_checks[0] << ", " << unsat_checks[1] << ", " << unsat_checks[2] << ", " << unsat_checks[3] << "]"
                  << std::endl;

        if (info_errs[0] == 0 && info_errs[1] == 0 && info_errs[2] == 0 && info_errs[3] == 0 &&
            unsat_checks[0] == 0 && unsat_checks[1] == 0 && unsat_checks[2] == 0 && unsat_checks[3] == 0) {
            std::cout << ">>> [CONVERGENCE] All 4 codewords reached 0 bit errors and 0 syndrome failures at iteration " << (iter + 1) << "!" << std::endl;
            if (standalone_mode) break;
        }
    }

    std::string golden_out_path = "dump/golden_lockstep_final_vnodes.bin";
    fs::create_directories("dump");
    std::ofstream f_out(golden_out_path, std::ios::binary);
    f_out.write(reinterpret_cast<const char*>(vnode_l1.data()), 20480);
    f_out.close();
    std::cout << "\n[SUCCESS] Wrote 40-Tile Face-Aligned Lockstep Golden Reference: " << golden_out_path << std::endl;

    if (!standalone_mode) {
        std::cout << "\n=================================================================" << std::endl;
        std::cout << " RUNNING LOCKSTEP DIFFERENTIAL COMPARISON: CPU vs TENSIX SILICON" << std::endl;
        std::cout << "=================================================================" << std::endl;

        std::string cmd = "/home/ttuser/.tenstorrent-venv/bin/python tools/diff_tensix_vs_emulator.py " + golden_out_path + " " + tensix_final_path;
        int ret = std::system(cmd.c_str());
        (void)ret;
    }

    return 0;
}
