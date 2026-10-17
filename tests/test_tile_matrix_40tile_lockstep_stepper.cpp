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
// Supports both Certified 2-Piece Baseline and Enhanced 3-Piece Convex Hull
// ------------------------------------------------------------------------------
#if defined(LDPC_USE_3PIECE_AMIN_STAR)
static bool g_use_3piece_amin = true;
#else
static bool g_use_3piece_amin = false;
#endif

// Model A: FP32 Precision (standard IEEE-754 24-bit mantissa)
inline float approx_min_star_fp32(float a, float b) {
    float min_v = std::min(a, b);
    float max_v = std::max(a, b);
    float diff  = max_v - min_v;

    if (g_use_3piece_amin) {
        float line1 = 0.67862f - 0.39622f * diff;
        float line2 = 0.48710f - 0.18297f * diff;
        float line3 = 0.17614f - 0.04029f * diff;
        float corr  = std::max(std::max(std::max(line1, line2), line3), 0.0f);
        return std::max(min_v - corr, 0.0f);
    } else {
        static const float c_0_69315   = bf16_to_fp32(16177);
        static const float c_neg_0_4   = bf16_to_fp32(48844);
        static const float c_0_65715   = bf16_to_fp32(16168);
        static const float c_neg_0_328 = bf16_to_fp32(48807);

        float line1 = c_0_69315 + c_neg_0_4 * diff;
        float line2 = c_0_65715 + c_neg_0_328 * diff;
        float corr  = std::max(std::max(line1, line2), 0.0f);
        return std::max(min_v - corr, 0.0f);
    }
}

// Model B: Tensix Hardware Bit-Exact Precision (Native bfloat16 SFPU emulation)
inline float approx_min_star_tensix(float a, float b) {
    float min_v = std::min(a, b);
    float max_v = std::max(a, b);
    float diff  = max_v - min_v;

    if (g_use_3piece_amin) {
        static const float c_0_67862     = bf16_to_fp32(16173); // 0x3F2D
        static const float c_neg_0_39622 = bf16_to_fp32(48842); // 0xBECA
        static const float c_0_48710     = bf16_to_fp32(16121); // 0x3EF9
        static const float c_neg_0_18297 = bf16_to_fp32(48699); // 0xBE3B
        static const float c_0_17614     = bf16_to_fp32(15924); // 0x3E34
        static const float c_neg_0_04029 = bf16_to_fp32(48421); // 0xBD25

        float line1 = c_0_67862 + c_neg_0_39622 * diff;
        float line2 = c_0_48710 + c_neg_0_18297 * diff;
        float line3 = c_0_17614 + c_neg_0_04029 * diff;
        float corr  = std::max(std::max(std::max(line1, line2), line3), 0.0f);
        return std::max(min_v - corr, 0.0f);
    } else {
        static const float c_0_69315   = bf16_to_fp32(16177); // 0.69315 (sFloat16b)
        static const float c_neg_0_4   = bf16_to_fp32(48844); // -0.4000 (sFloat16b)
        static const float c_0_65715   = bf16_to_fp32(16168); // 0.65715 (sFloat16b)
        static const float c_neg_0_328 = bf16_to_fp32(48807); // -0.3280 (sFloat16b)

        float line1 = c_0_69315 + c_neg_0_4 * diff;
        float line2 = c_0_65715 + c_neg_0_328 * diff;
        float corr  = std::max(std::max(line1, line2), 0.0f);
        return std::max(min_v - corr, 0.0f);
    }
}

// ------------------------------------------------------------------------------
// Bit-Exact Tensix Vector Box-Muller AWGN Generator
// Replicates the 32-lane vector SFPU xoshiro128+ and Cartesian Box-Muller
// transformations from engines/tile_matrix_4cw/compute_trisc.cpp with 100% bit-accuracy.
// ------------------------------------------------------------------------------
namespace TensixBM {

inline uint32_t arx_hash(uint32_t x, uint32_t c) {
    x += c;
    x ^= (x >> 16);
    x += (x << 11);
    x ^= (x >> 7);
    x += (x << 5);
    x ^= (x >> 13);
    x += (x << 17);
    x ^= (x >> 11);
    return x;
}

struct VectorXoshiro128Plus {
    uint32_t s0, s1, s2, s3;
    inline void init(uint32_t seed_val) {
        s0 = arx_hash(seed_val, 0x9E3779B9U);
        s1 = arx_hash(seed_val, 0x85EBCA6BU);
        s2 = arx_hash(seed_val, 0xC2B2AE35U);
        s3 = arx_hash(seed_val, 0x27D4EB2FU);
        if ((s0 | s1 | s2 | s3) == 0) {
            s0 = 0x13579BDFU;
        }
    }
    inline uint32_t next() {
        uint32_t res = s0 + s3;
        uint32_t t = s1 << 9;
        s2 ^= s0;
        s3 ^= s1;
        s1 ^= s2;
        s0 ^= s3;
        s2 ^= t;
        s3 = (s3 << 11) | (s3 >> 21);
        return res;
    }
};

inline float sfpu_fast_ln(float a) {
    if (a < 1e-7f) a = 1e-7f;
    uint32_t u;
    std::memcpy(&u, &a, sizeof(u));
    int32_t exp_val = static_cast<int32_t>((u >> 23) & 0xFF) - 127;
    uint32_t m_bits = (u & 0x007FFFFF) | 0x3F800000;
    float m;
    std::memcpy(&m, &m_bits, sizeof(m));
    float z = m - 1.0f;

    // In TT-Metal SFPI, sFloat16b(v) truncates float directly to 16 bits (float_as_uint >> 16)
    float p = to_bf16_trunc(-0.0528f);
    p = p * z + to_bf16_trunc(0.1264f);
    p = p * z - to_bf16_trunc(0.1678f);
    p = p * z + to_bf16_trunc(0.2057f);
    p = p * z - to_bf16_trunc(0.2490f);
    p = p * z + to_bf16_trunc(0.3328f);
    p = p * z - 0.5f;
    p = p * z + 1.0f;
    float ln_m = p * z;

    float e_flt = static_cast<float>(exp_val);
    return ln_m + e_flt * 0.69314718f;
}

inline float sfpu_fast_sqrt(float val) {
    if (val < 1e-7f) val = 1e-7f;
    uint32_t u;
    std::memcpy(&u, &val, sizeof(u));
    uint32_t magic = 0x5F375A86U;
    uint32_t approx_u = magic - (u >> 1);
    float approx;
    std::memcpy(&approx, &approx_u, sizeof(approx));
    float neg_half_val = val * -0.5f;
    approx = ((approx * approx) * neg_half_val + 1.5f) * approx;
    approx = ((approx * approx) * neg_half_val + 1.5f) * approx;
    return approx * val;
}

inline void generate_channel_vnodes(
    uint8_t* vnode_l1,
    float mu_llr,
    float sigma_llr,
    uint32_t invocation_id,
    uint32_t batch,
    float l_ch_max = 7.0f
) {
    for (uint32_t t = 0; t < 8; ++t) {
        VectorXoshiro128Plus rngs[32];
        uint32_t seed_base = 0x12345678U + (invocation_id * 1013904223U) + batch * 31337U + t * 7919U;
        for (uint32_t lane = 0; lane < 32; ++lane) {
            uint32_t lane_seed = seed_base + (2 * lane);
            rngs[lane].init(lane_seed);
        }

        float dst_reg[32][32];
        for (int r = 0; r < 32; ++r) {
            for (int lane = 0; lane < 32; ++lane) {
                uint32_t raw = rngs[lane].next();
                uint32_t u_bits = ((raw >> 9) & 0x007FFFFF) | 0x3F800000;
                float u;
                std::memcpy(&u, &u_bits, sizeof(u));
                u -= 1.0f;
                dst_reg[r][lane] = to_bf16_trunc(std::max(u, 1e-7f));
            }
        }

        for (int p = 0; p < 16; ++p) {
            for (int lane = 0; lane < 32; ++lane) {
                float u1 = dst_reg[2 * p + 0][lane];
                float ln_u1 = sfpu_fast_ln(u1);
                float r = sfpu_fast_sqrt(ln_u1 * -2.0f);

                float u2 = dst_reg[2 * p + 1][lane];
                float theta = u2 * 6.2831853f - 3.14159265f;
                float s_th = theta * theta;

                float cos_val = to_bf16_trunc(0.0000248f);
                cos_val = cos_val * s_th - to_bf16_trunc(0.0013889f);
                cos_val = cos_val * s_th + to_bf16_trunc(0.0416667f);
                cos_val = cos_val * s_th - 0.5f;
                cos_val = cos_val * s_th + 1.0f;
                float llr0 = mu_llr + sigma_llr * (r * cos_val);
                if (l_ch_max > 0.0f) {
                    llr0 = std::min(std::max(llr0, -l_ch_max), l_ch_max);
                }
                dst_reg[2 * p + 0][lane] = to_bf16_trunc(llr0);

                float sin_val = to_bf16_trunc(0.00000276f);
                sin_val = sin_val * s_th - to_bf16_trunc(0.0001984f);
                sin_val = sin_val * s_th + to_bf16_trunc(0.0083333f);
                sin_val = sin_val * s_th - to_bf16_trunc(0.1666667f);
                sin_val = sin_val * s_th + 1.0f;
                float llr1 = mu_llr + sigma_llr * (r * (sin_val * theta));
                if (l_ch_max > 0.0f) {
                    llr1 = std::min(std::max(llr1, -l_ch_max), l_ch_max);
                }
                dst_reg[2 * p + 1][lane] = to_bf16_trunc(llr1);
            }
        }

        uint32_t tile_offset = t * 2048;
        for (int r = 0; r < 32; ++r) {
            for (int lane = 0; lane < 32; ++lane) {
                uint16_t bf = fp32_to_bf16_trunc(dst_reg[r][lane]);
                uint32_t byte_off = tile_offset + get_scratch_byte_offset(r, lane);
                *reinterpret_cast<uint16_t*>(vnode_l1 + byte_off) = bf;
            }
        }
    }

    std::memset(vnode_l1 + 8 * 2048, 0, 4096);
}

} // namespace TensixBM

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

enum class ScheduleMode {
    SYNC,
    PIPELINED,
    INTER_BATCH
};
static ScheduleMode g_schedule_mode = ScheduleMode::PIPELINED;

struct DualBatchStats {
    BatchStats a;
    BatchStats b;
};

// ------------------------------------------------------------------------------
// Modular Single-Tile Processing Kernels (Shared across Sync and Pipelined modes)
// ------------------------------------------------------------------------------
inline void step_tile_gather(
    std::vector<uint16_t>& scratch_in,
    const uint8_t* vnode_l1,
    const uint32_t* tile_hoff_w0,
    const uint32_t* tile_hoff_w1
) {
    for (int p = 0; p < 48; ++p) {
        // Worker 0: Faces 0 & 2
        uint32_t off0 = tile_hoff_w0[p];
        uint32_t off0_even = off0 & 0xFFFF;
        uint32_t off0_odd  = off0 >> 16;
        uint32_t s0_even = SCRATCH_FACE_LUT.w0_even[p];
        uint32_t s0_odd  = SCRATCH_FACE_LUT.w0_odd[p];

        *reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(scratch_in.data()) + s0_even) =
            *reinterpret_cast<const uint64_t*>(vnode_l1 + off0_even);
        *reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(scratch_in.data()) + s0_odd) =
            *reinterpret_cast<const uint64_t*>(vnode_l1 + off0_odd);

        // Worker 1: Faces 1 & 3
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
}

inline void step_tile_compute(
    const std::vector<uint16_t>& scratch_in,
    std::vector<uint16_t>& scratch_out,
    std::vector<uint16_t>& r_msg_tile,
    uint32_t deg,
    uint32_t iter,
    float post_clip_max,
    bool tensix_precision,
    bool output_delta = false
) {
    const int face_bases[4] = {0, 8, 16, 24};

    for (int f = 0; f < 4; ++f) {
        int base_r = face_bases[f];

        for (int sub = 0; sub < 2; ++sub) {
            int p0 = base_r + 0 + sub;
            int p1 = base_r + 2 + sub;
            int p2 = base_r + 4 + sub;

            float q0[32], q1[32], q2[32];
            float r_old0_arr[32], r_old1_arr[32], r_old2_arr[32];
            for (uint32_t lane = 0; lane < 32; ++lane) {
                uint32_t off0 = get_scratch_byte_offset(p0, lane);
                uint32_t off1 = get_scratch_byte_offset(p1, lane);
                uint32_t off2 = get_scratch_byte_offset(p2, lane);

                float l0 = bf16_to_fp32(*reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(scratch_in.data()) + off0));
                float l1 = bf16_to_fp32(*reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(scratch_in.data()) + off1));
                float l2 = bf16_to_fp32(*reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(scratch_in.data()) + off2));

                float r_old0 = (iter == 0) ? 0.0f : bf16_to_fp32(*reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(r_msg_tile.data()) + off0));
                float r_old1 = (iter == 0) ? 0.0f : bf16_to_fp32(*reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(r_msg_tile.data()) + off1));
                float r_old2 = (iter == 0) ? 0.0f : bf16_to_fp32(*reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(r_msg_tile.data()) + off2));
                r_old0_arr[lane] = r_old0;
                r_old1_arr[lane] = r_old1;
                r_old2_arr[lane] = r_old2;

                q0[lane] = tensix_precision ? to_bf16(l0 - r_old0) : (l0 - r_old0);
                q1[lane] = tensix_precision ? to_bf16(l1 - r_old1) : (l1 - r_old1);
                q2[lane] = tensix_precision ? to_bf16(l2 - r_old2) : (l2 - r_old2);
            }

            if (deg == 6) {
                for (uint32_t lane = 0; lane < 32; ++lane) {
                    uint32_t s_i = (lane < 16) ? (lane + 16) : (lane - 16);

                    float m0 = std::fabs(q0[lane]), m0_swap = std::fabs(q0[s_i]);
                    float m1 = std::fabs(q1[lane]), m1_swap = std::fabs(q1[s_i]);
                    float m2 = std::fabs(q2[lane]), m2_swap = std::fabs(q2[s_i]);

                    float s0 = (q0[lane] < 0.0f) ? -1.0f : 1.0f;
                    float s1 = (q1[lane] < 0.0f) ? -1.0f : 1.0f;
                    float s2 = (q2[lane] < 0.0f) ? -1.0f : 1.0f;

                    float s0_swap = (q0[s_i] < 0.0f) ? -1.0f : 1.0f;
                    float s1_swap = (q1[s_i] < 0.0f) ? -1.0f : 1.0f;
                    float s2_swap = (q2[s_i] < 0.0f) ? -1.0f : 1.0f;

                    float s_tot = (s0 * s0_swap) * (s1 * s1_swap) * (s2 * s2_swap);

                    auto ams = [&](float a, float b) {
                        return tensix_precision ? approx_min_star_tensix(a, b) : approx_min_star_fp32(a, b);
                    };

                    float m01 = tensix_precision ? to_bf16(ams(m0, m0_swap)) : ams(m0, m0_swap);
                    float m23 = tensix_precision ? to_bf16(ams(m1, m1_swap)) : ams(m1, m1_swap);
                    float m45 = tensix_precision ? to_bf16(ams(m2, m2_swap)) : ams(m2, m2_swap);

                    float r_mag_0 = ams(ams(m23, m45), m0_swap);
                    float r_mag_1 = ams(ams(m01, m45), m1_swap);
                    float r_mag_2 = ams(ams(m01, m23), m2_swap);

                    float r_out_0 = tensix_precision ? to_bf16((s_tot * s0) * r_mag_0) : ((s_tot * s0) * r_mag_0);
                    float r_out_1 = tensix_precision ? to_bf16((s_tot * s1) * r_mag_1) : ((s_tot * s1) * r_mag_1);
                    float r_out_2 = tensix_precision ? to_bf16((s_tot * s2) * r_mag_2) : ((s_tot * s2) * r_mag_2);

                    float val_out_0, val_out_1, val_out_2;
                    if (output_delta) {
                        val_out_0 = tensix_precision ? to_bf16(r_out_0 - r_old0_arr[lane]) : (r_out_0 - r_old0_arr[lane]);
                        val_out_1 = tensix_precision ? to_bf16(r_out_1 - r_old1_arr[lane]) : (r_out_1 - r_old1_arr[lane]);
                        val_out_2 = tensix_precision ? to_bf16(r_out_2 - r_old2_arr[lane]) : (r_out_2 - r_old2_arr[lane]);
                    } else {
                        val_out_0 = tensix_precision ? to_bf16(q0[lane] + r_out_0) : (q0[lane] + r_out_0);
                        val_out_1 = tensix_precision ? to_bf16(q1[lane] + r_out_1) : (q1[lane] + r_out_1);
                        val_out_2 = tensix_precision ? to_bf16(q2[lane] + r_out_2) : (q2[lane] + r_out_2);

                        if (post_clip_max > 0.0f) {
                            if (val_out_0 > post_clip_max) val_out_0 = post_clip_max;
                            if (val_out_0 < -post_clip_max) val_out_0 = -post_clip_max;
                            if (val_out_1 > post_clip_max) val_out_1 = post_clip_max;
                            if (val_out_1 < -post_clip_max) val_out_1 = -post_clip_max;
                            if (val_out_2 > post_clip_max) val_out_2 = post_clip_max;
                            if (val_out_2 < -post_clip_max) val_out_2 = -post_clip_max;
                        }
                    }

                    uint32_t off0 = get_scratch_byte_offset(p0, lane);
                    uint32_t off1 = get_scratch_byte_offset(p1, lane);
                    uint32_t off2 = get_scratch_byte_offset(p2, lane);

                    *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(r_msg_tile.data()) + off0) = fp32_to_bf16(r_out_0);
                    *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(scratch_out.data()) + off0)   = fp32_to_bf16(val_out_0);

                    *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(r_msg_tile.data()) + off1) = fp32_to_bf16(r_out_1);
                    *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(scratch_out.data()) + off1)   = fp32_to_bf16(val_out_1);

                    *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(r_msg_tile.data()) + off2) = fp32_to_bf16(r_out_2);
                    *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(scratch_out.data()) + off2)   = fp32_to_bf16(val_out_2);
                }
            } else { // deg == 3
                for (uint32_t lane = 0; lane < 32; ++lane) {
                    uint32_t s_i = (lane < 16) ? (lane + 16) : (lane - 16);

                    float m0 = std::fabs(q0[lane]), m0_swap = std::fabs(q0[s_i]);
                    float m1 = std::fabs(q1[lane]), m1_swap = std::fabs(q1[s_i]);
                    float m2 = std::fabs(q2[lane]), m2_swap = std::fabs(q2[s_i]);

                    float s0 = (q0[lane] < 0.0f) ? -1.0f : 1.0f;
                    float s1 = (q1[lane] < 0.0f) ? -1.0f : 1.0f;
                    float s2 = (q2[lane] < 0.0f) ? -1.0f : 1.0f;

                    float s0_swap = (q0[s_i] < 0.0f) ? -1.0f : 1.0f;
                    float s1_swap = (q1[s_i] < 0.0f) ? -1.0f : 1.0f;
                    float s2_swap = (q2[s_i] < 0.0f) ? -1.0f : 1.0f;

                    float m_e2_blk0 = (lane < 16) ? m1 : m1_swap;
                    float s_e2_blk0 = (lane < 16) ? s1 : s1_swap;
                    float m_e0_blk1 = (lane < 16) ? m1_swap : m1;
                    float s_e0_blk1 = (lane < 16) ? s1_swap : s1;

                    auto ams = [&](float a, float b) {
                        return tensix_precision ? approx_min_star_tensix(a, b) : approx_min_star_fp32(a, b);
                    };

                    float s_tot_blk0 = (s0 * s0_swap) * s_e2_blk0;
                    float r_mag_0    = ams(m0_swap, m_e2_blk0);
                    float r_out_0    = tensix_precision ? to_bf16((s_tot_blk0 * s0) * r_mag_0) : ((s_tot_blk0 * s0) * r_mag_0);

                    float r_mag_e2_blk0 = ams(m0, m0_swap);
                    float s_out_e2_blk0 = s_tot_blk0 * s_e2_blk0;

                    float s_tot_blk1 = (s2 * s2_swap) * s_e0_blk1;
                    float r_mag_2    = ams(m2_swap, m_e0_blk1);
                    float r_out_2    = tensix_precision ? to_bf16((s_tot_blk1 * s2) * r_mag_2) : ((s_tot_blk1 * s2) * r_mag_2);

                    float r_mag_e0_blk1 = ams(m2, m2_swap);
                    float s_out_e0_blk1 = s_tot_blk1 * s_e0_blk1;

                    float r_out_1_raw = (lane < 16) ? (s_out_e2_blk0 * r_mag_e2_blk0)
                                                    : (s_out_e0_blk1 * r_mag_e0_blk1);
                    float r_out_1 = tensix_precision ? to_bf16(r_out_1_raw) : r_out_1_raw;

                    float val_out_0, val_out_1, val_out_2;
                    if (output_delta) {
                        val_out_0 = tensix_precision ? to_bf16(r_out_0 - r_old0_arr[lane]) : (r_out_0 - r_old0_arr[lane]);
                        val_out_1 = tensix_precision ? to_bf16(r_out_1 - r_old1_arr[lane]) : (r_out_1 - r_old1_arr[lane]);
                        val_out_2 = tensix_precision ? to_bf16(r_out_2 - r_old2_arr[lane]) : (r_out_2 - r_old2_arr[lane]);
                    } else {
                        val_out_0 = tensix_precision ? to_bf16(q0[lane] + r_out_0) : (q0[lane] + r_out_0);
                        val_out_1 = tensix_precision ? to_bf16(q1[lane] + r_out_1) : (q1[lane] + r_out_1);
                        val_out_2 = tensix_precision ? to_bf16(q2[lane] + r_out_2) : (q2[lane] + r_out_2);

                        if (post_clip_max > 0.0f) {
                            if (val_out_0 > post_clip_max) val_out_0 = post_clip_max;
                            if (val_out_0 < -post_clip_max) val_out_0 = -post_clip_max;
                            if (val_out_1 > post_clip_max) val_out_1 = post_clip_max;
                            if (val_out_1 < -post_clip_max) val_out_1 = -post_clip_max;
                            if (val_out_2 > post_clip_max) val_out_2 = post_clip_max;
                            if (val_out_2 < -post_clip_max) val_out_2 = -post_clip_max;
                        }
                    }

                    uint32_t off0 = get_scratch_byte_offset(p0, lane);
                    uint32_t off1 = get_scratch_byte_offset(p1, lane);
                    uint32_t off2 = get_scratch_byte_offset(p2, lane);

                    *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(r_msg_tile.data()) + off0) = fp32_to_bf16(r_out_0);
                    *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(scratch_out.data()) + off0)   = fp32_to_bf16(val_out_0);

                    *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(r_msg_tile.data()) + off1) = fp32_to_bf16(r_out_1);
                    *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(scratch_out.data()) + off1)   = fp32_to_bf16(val_out_1);

                    *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(r_msg_tile.data()) + off2) = fp32_to_bf16(r_out_2);
                    *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(scratch_out.data()) + off2)   = fp32_to_bf16(val_out_2);
                }
            }
        }

        // Zero out unused padding registers (regs 6, 7 of this face)
        for (int sub = 0; sub < 2; ++sub) {
            uint32_t reg = base_r + 6 + sub;
            for (uint32_t lane = 0; lane < 32; ++lane) {
                uint32_t offset = get_scratch_byte_offset(reg, lane);
                *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(r_msg_tile.data()) + offset) = 0;
                *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(scratch_out.data()) + offset)   = 0;
            }
        }
    }
}

inline void step_tile_scatter(
    const std::vector<uint16_t>& scratch_out,
    uint8_t* vnode_l1,
    const uint32_t* tile_hoff_w0,
    const uint32_t* tile_hoff_w1,
    uint8_t active_mask = 0x0F,
    bool accum_delta = false,
    float post_clip_max = 0.0f
) {
    if (!accum_delta) {
        if (active_mask == 0x0F) {
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
        } else {
            uint64_t mask64 = ((active_mask & 1) ? 0x000000000000FFFFULL : 0ULL) |
                              ((active_mask & 2) ? 0x00000000FFFF0000ULL : 0ULL) |
                              ((active_mask & 4) ? 0x0000FFFF00000000ULL : 0ULL) |
                              ((active_mask & 8) ? 0xFFFF000000000000ULL : 0ULL);
            uint64_t inv_mask64 = ~mask64;

            for (int p = 0; p < 48; ++p) {
                // Worker 0
                uint32_t off0 = tile_hoff_w0[p];
                uint32_t off0_even = off0 & 0xFFFF;
                uint32_t off0_odd  = off0 >> 16;
                uint32_t s0_even = SCRATCH_FACE_LUT.w0_even[p];
                uint32_t s0_odd  = SCRATCH_FACE_LUT.w0_odd[p];

                uint64_t src0_even = *reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(scratch_out.data()) + s0_even);
                uint64_t dst0_even = *reinterpret_cast<uint64_t*>(vnode_l1 + off0_even);
                *reinterpret_cast<uint64_t*>(vnode_l1 + off0_even) = (src0_even & mask64) | (dst0_even & inv_mask64);

                uint64_t src0_odd = *reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(scratch_out.data()) + s0_odd);
                uint64_t dst0_odd = *reinterpret_cast<uint64_t*>(vnode_l1 + off0_odd);
                *reinterpret_cast<uint64_t*>(vnode_l1 + off0_odd) = (src0_odd & mask64) | (dst0_odd & inv_mask64);

                // Worker 1
                uint32_t off1 = tile_hoff_w1[p];
                uint32_t off1_even = off1 & 0xFFFF;
                uint32_t off1_odd  = off1 >> 16;
                uint32_t s1_even = SCRATCH_FACE_LUT.w1_even[p];
                uint32_t s1_odd  = SCRATCH_FACE_LUT.w1_odd[p];

                uint64_t src1_even = *reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(scratch_out.data()) + s1_even);
                uint64_t dst1_even = *reinterpret_cast<uint64_t*>(vnode_l1 + off1_even);
                *reinterpret_cast<uint64_t*>(vnode_l1 + off1_even) = (src1_even & mask64) | (dst1_even & inv_mask64);

                uint64_t src1_odd = *reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(scratch_out.data()) + s1_odd);
                uint64_t dst1_odd = *reinterpret_cast<uint64_t*>(vnode_l1 + off1_odd);
                *reinterpret_cast<uint64_t*>(vnode_l1 + off1_odd) = (src1_odd & mask64) | (dst1_odd & inv_mask64);
            }
        }
    } else {
        // Delta-R accumulation mode (Resolves RAW overlap hazard in pipelined DAE)
        auto accum_word = [&](uint32_t vnode_byte_off, uint32_t scratch_byte_off) {
            uint16_t* p_dst = reinterpret_cast<uint16_t*>(vnode_l1 + vnode_byte_off);
            const uint16_t* p_del = reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(scratch_out.data()) + scratch_byte_off);
            for (int c = 0; c < 4; ++c) {
                if (active_mask & (1 << c)) {
                    float cur = bf16_to_fp32(p_dst[c]);
                    float del = bf16_to_fp32(p_del[c]);
                    float updated = cur + del;
                    if (post_clip_max > 0.0f) {
                        if (updated > post_clip_max) updated = post_clip_max;
                        else if (updated < -post_clip_max) updated = -post_clip_max;
                    }
                    p_dst[c] = fp32_to_bf16(updated);
                }
            }
        };

        for (int p = 0; p < 48; ++p) {
            // Worker 0
            uint32_t off0 = tile_hoff_w0[p];
            accum_word(off0 & 0xFFFF, SCRATCH_FACE_LUT.w0_even[p]);
            accum_word(off0 >> 16,    SCRATCH_FACE_LUT.w0_odd[p]);

            // Worker 1
            uint32_t off1 = tile_hoff_w1[p];
            accum_word(off1 & 0xFFFF, SCRATCH_FACE_LUT.w1_even[p]);
            accum_word(off1 >> 16,    SCRATCH_FACE_LUT.w1_odd[p]);
        }
    }
}

// ------------------------------------------------------------------------------
// 1. Synchronous Stop-and-Wait Batch Decoder (Diagnostic / Legacy Baseline)
// ------------------------------------------------------------------------------
inline BatchStats decode_tile_matrix_batch_sync(
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

            step_tile_gather(scratch_in, vnode_l1, tile_hoff_w0, tile_hoff_w1);
            step_tile_compute(scratch_in, scratch_out, r_msg_fifo[t], deg, iter, post_clip_max, tensix_precision);
            step_tile_scatter(scratch_out, vnode_l1, tile_hoff_w0, tile_hoff_w1, active_mask);
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
// 2. High-Throughput Pipelined Ping-Pong DAE Batch Decoder (Production Engine)
// Continuous Circular Modulo-40 Ring with 3-Step Startup Priming & Zero Draining
// ------------------------------------------------------------------------------
inline BatchStats decode_tile_matrix_batch_pipelined(
    const ParityCheckMatrix& pcm,
    const std::vector<DynamicTileDescriptor>& schedule,
    uint8_t* vnode_l1,
    std::vector<std::vector<uint16_t>>& r_msg_fifo,
    std::vector<uint16_t> scratch_in[2],
    std::vector<uint16_t> scratch_out[2],
    uint32_t max_iter,
    float post_clip_max = 0.0f,
    bool tensix_precision = true
) {
    BatchStats stats;
    uint8_t active_mask = 0x0F;
    const uint32_t* p_hoff_base = pcm.h_offsets_pairs.data();
    uint32_t num_tiles = schedule.size();
    uint32_t max_steps = max_iter * num_tiles;

    // Loop over global step k across the continuous circular ring
    for (uint32_t k = 0; k < max_steps + 2; ++k) {
        uint32_t b_in   = k % 2;
        uint32_t b_comp = (k - 1) % 2;
        uint32_t b_scat = (k - 2) % 2;

        uint32_t t_gath = k % num_tiles;
        uint32_t t_comp = (k >= 1) ? ((k - 1) % num_tiles) : 0;
        uint32_t t_scat = (k >= 2) ? ((k - 2) % num_tiles) : 0;

        // 1. Scatter Stage: Scatter Tile t_scat (Enabled for k >= 2)
        // Must execute before Gather(k) so that Tile k sees updates from Tile k-2!
        if (k >= 2) {
            const uint32_t* tile_hoff_w0 = p_hoff_base + t_scat * 96;
            const uint32_t* tile_hoff_w1 = tile_hoff_w0 + 48;
            step_tile_scatter(scratch_out[b_scat], vnode_l1, tile_hoff_w0, tile_hoff_w1, active_mask, true, post_clip_max);
        }

        // 2. Gather Stage: Gather Tile t_gath (Enabled for k < max_steps)
        if (k < max_steps) {
            const uint32_t* tile_hoff_w0 = p_hoff_base + t_gath * 96;
            const uint32_t* tile_hoff_w1 = tile_hoff_w0 + 48;
            step_tile_gather(scratch_in[b_in], vnode_l1, tile_hoff_w0, tile_hoff_w1);
        }

        // 3. Compute Stage: Compute Tile t_comp (Enabled for 1 <= k <= max_steps)
        if (k >= 1 && k <= max_steps) {
            uint32_t deg = schedule[t_comp].check_degree;
            uint32_t iter = (k - 1) / num_tiles;
            step_tile_compute(scratch_in[b_comp], scratch_out[b_comp], r_msg_fifo[t_comp], deg, iter, post_clip_max, tensix_precision, true);
        }

        // 4. Parity Convergence Evaluation (when Tile 39 finishes scatter, i.e. t_scat == num_tiles - 1)
        if (k >= 2 && t_scat == num_tiles - 1) {
            uint32_t iter_done = (k - 2) / num_tiles;
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
                    stats.iters_taken[c] = iter_done + 1;
                    active_mask &= ~(1 << c);
                }
            }

            if (active_mask == 0) break;
        }
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
// 3. Inter-Batch Ping-Pong DAE Batch Decoder (Section 7.1)
// Alternates execution steps between Batch A (Ping) and Batch B (Pong).
// Zero RAW hazard between adjacent steps enables pure 64-bit direct stores (*d_e64 = src_even64).
// Processes 8 codewords concurrently (4 in Batch A + 4 in Batch B).
// ------------------------------------------------------------------------------
inline DualBatchStats decode_tile_matrix_batch_inter_batch(
    const ParityCheckMatrix& pcm,
    const std::vector<DynamicTileDescriptor>& schedule,
    uint8_t* vnode_l1_a,
    uint8_t* vnode_l1_b,
    std::vector<std::vector<uint16_t>>& r_msg_fifo_a,
    std::vector<std::vector<uint16_t>>& r_msg_fifo_b,
    std::vector<uint16_t> scratch_in[2],
    std::vector<uint16_t> scratch_out[2],
    uint32_t max_iter,
    float post_clip_max = 0.0f,
    bool tensix_precision = true
) {
    DualBatchStats stats;
    uint8_t active_mask[2] = {0x0F, 0x0F}; // active_mask[0] for Batch A, active_mask[1] for Batch B
    uint8_t* vnodes[2] = {vnode_l1_a, vnode_l1_b};
    std::vector<std::vector<uint16_t>>* r_fifos[2] = {&r_msg_fifo_a, &r_msg_fifo_b};
    const uint32_t* p_hoff_base = pcm.h_offsets_pairs.data();
    uint32_t num_tiles = schedule.size();
    uint32_t max_steps = max_iter * num_tiles;

    // Loop over global step k across the interleaved dual-batch schedule
    // Total steps: 2 * max_steps + 2 (covers priming and draining for both batches)
    for (uint32_t k = 0; k < 2 * max_steps + 2; ++k) {
        uint32_t b_dae = k % 2;
        uint32_t s_dae = k / 2;

        // 1. DAE Scatter Stage: Scatter Tile (s_dae - 1) of Batch b_dae
        if (s_dae >= 1 && (s_dae - 1) < max_steps) {
            uint32_t s_scat = s_dae - 1;
            uint32_t t_scat = s_scat % num_tiles;
            const uint32_t* tile_hoff_w0 = p_hoff_base + t_scat * 96;
            const uint32_t* tile_hoff_w1 = tile_hoff_w0 + 48;

            if (active_mask[b_dae] != 0) {
                step_tile_scatter(scratch_out[b_dae], vnodes[b_dae], tile_hoff_w0, tile_hoff_w1, active_mask[b_dae], false /* direct store */, post_clip_max);
            }

            // Parity Convergence Evaluation when Tile 39 finishes scatter
            if (t_scat == num_tiles - 1 && active_mask[b_dae] != 0) {
                uint32_t iter_done = s_scat / num_tiles;
                for (int c = 0; c < 4; ++c) {
                    if (!(active_mask[b_dae] & (1 << c))) continue;

                    bool satisfied = true;
                    for (uint32_t m = 0; m < pcm.M; ++m) {
                        int syn = 0;
                        for (uint32_t off : pcm.check_to_var_offsets[m]) {
                            const uint16_t* p = reinterpret_cast<const uint16_t*>(vnodes[b_dae] + off);
                            if (bf16_to_fp32(p[c]) < 0.0f) syn ^= 1;
                        }
                        if (syn != 0) {
                            satisfied = false;
                            break;
                        }
                    }

                    if (satisfied) {
                        if (b_dae == 0) {
                            stats.a.converged[c] = true;
                            stats.a.iters_taken[c] = iter_done + 1;
                        } else {
                            stats.b.converged[c] = true;
                            stats.b.iters_taken[c] = iter_done + 1;
                        }
                        active_mask[b_dae] &= ~(1 << c);
                    }
                }
            }
        }

        // 2. DAE Gather Stage: Gather Tile s_dae of Batch b_dae
        if (s_dae < max_steps) {
            if (active_mask[b_dae] != 0) {
                uint32_t t_gath = s_dae % num_tiles;
                const uint32_t* tile_hoff_w0 = p_hoff_base + t_gath * 96;
                const uint32_t* tile_hoff_w1 = tile_hoff_w0 + 48;
                step_tile_gather(scratch_in[b_dae], vnodes[b_dae], tile_hoff_w0, tile_hoff_w1);
            }
        }

        // 3. TRISC Compute Stage: Compute Tile s_comp of Batch b_comp
        if (k >= 1) {
            uint32_t b_comp = (k - 1) % 2;
            uint32_t s_comp = (k - 1) / 2;
            if (s_comp < max_steps && active_mask[b_comp] != 0) {
                uint32_t t_comp = s_comp % num_tiles;
                uint32_t iter_comp = s_comp / num_tiles;
                uint32_t deg = schedule[t_comp].check_degree;
                step_tile_compute(scratch_in[b_comp], scratch_out[b_comp], (*r_fifos[b_comp])[t_comp], deg, iter_comp, post_clip_max, tensix_precision, false /* direct L_new */);
            }
        }

        // Early termination check: if both batches have converged, stop early
        if (active_mask[0] == 0 && active_mask[1] == 0) break;
    }

    // Residual Bit Error Check on Information Bits (v < K)
    for (int c = 0; c < 4; ++c) {
        if (!stats.a.converged[c]) {
            stats.a.iters_taken[c] = max_iter;
            for (uint32_t off : pcm.info_var_offsets) {
                const uint16_t* p = reinterpret_cast<const uint16_t*>(vnode_l1_a + off);
                if (bf16_to_fp32(p[c]) < 0.0f) stats.a.bit_errors[c]++;
            }
        }
        if (!stats.b.converged[c]) {
            stats.b.iters_taken[c] = max_iter;
            for (uint32_t off : pcm.info_var_offsets) {
                const uint16_t* p = reinterpret_cast<const uint16_t*>(vnode_l1_b + off);
                if (bf16_to_fp32(p[c]) < 0.0f) stats.b.bit_errors[c]++;
            }
        }
    }

    return stats;
}

// ------------------------------------------------------------------------------
// Top-Level Batch Dispatcher
// ------------------------------------------------------------------------------
inline BatchStats decode_tile_matrix_batch(
    const ParityCheckMatrix& pcm,
    const std::vector<DynamicTileDescriptor>& schedule,
    uint8_t* vnode_l1,
    std::vector<std::vector<uint16_t>>& r_msg_fifo,
    std::vector<uint16_t> scratch_in[2],
    std::vector<uint16_t> scratch_out[2],
    uint32_t max_iter,
    float post_clip_max = 0.0f,
    bool tensix_precision = true
) {
    if (g_schedule_mode == ScheduleMode::SYNC) {
        return decode_tile_matrix_batch_sync(
            pcm, schedule, vnode_l1, r_msg_fifo, scratch_in[0], scratch_out[0], max_iter, post_clip_max, tensix_precision
        );
    } else {
        return decode_tile_matrix_batch_pipelined(
            pcm, schedule, vnode_l1, r_msg_fifo, scratch_in, scratch_out, max_iter, post_clip_max, tensix_precision
        );
    }
}

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
    std::vector<uint16_t> s_in[2] = {scratch_in, std::vector<uint16_t>(1024, 0)};
    std::vector<uint16_t> s_out[2] = {scratch_out, std::vector<uint16_t>(1024, 0)};
    BatchStats st = decode_tile_matrix_batch(pcm, schedule, vnode_l1, r_msg_fifo, s_in, s_out, max_iter, post_clip_max, tensix_precision);
    scratch_in = s_in[0];
    scratch_out = s_out[0];
    return st;
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
    int mc_idx = 1;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--monte-carlo") {
            mc_idx = i;
            break;
        }
    }

    double eb_n0_db = (argc > mc_idx + 1 && argv[mc_idx + 1][0] != '-') ? std::stod(argv[mc_idx + 1]) : 2.0;
    uint64_t max_blocks = (argc > mc_idx + 2 && argv[mc_idx + 2][0] != '-') ? std::stoull(argv[mc_idx + 2]) : 10000000ULL;
    uint64_t min_frame_errors = (argc > mc_idx + 3 && argv[mc_idx + 3][0] != '-') ? std::stoull(argv[mc_idx + 3]) : 25ULL;
    uint32_t max_iter = (argc > mc_idx + 4 && argv[mc_idx + 4][0] != '-') ? std::stoul(argv[mc_idx + 4]) : 200U;
    int num_threads = (argc > mc_idx + 5 && argv[mc_idx + 5][0] != '-') ? std::stoi(argv[mc_idx + 5]) : 16;
    float post_clip = (argc > mc_idx + 6 && argv[mc_idx + 6][0] != '-') ? std::stof(argv[mc_idx + 6]) : 0.0f;
    std::string csv_path = (argc > mc_idx + 7 && argv[mc_idx + 7][0] != '-') ? argv[mc_idx + 7] : "Results/results_40tile_lockstep_ebn0_2.00_10M.csv";

    if (num_threads <= 0) num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);

    bool tensix_precision = true;
    bool use_tensix_noise = false;
    uint32_t tensix_inv = 1;
    uint32_t tensix_start_batch = 0;
    float l_ch_max = 7.0f;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--fp32" || arg == "--full-precision") {
            tensix_precision = false;
        } else if (arg == "--tensix-precision") {
            tensix_precision = true;
        } else if (arg == "--pipelined" || arg == "--intra-batch") {
            g_schedule_mode = ScheduleMode::PIPELINED;
        } else if (arg == "--inter-batch" || arg == "--inter-batch-ping-pong" || arg == "--pipelined-inter-batch") {
            g_schedule_mode = ScheduleMode::INTER_BATCH;
        } else if (arg == "--sync" || arg == "--stop-and-wait" || arg == "--lockstep") {
            g_schedule_mode = ScheduleMode::SYNC;
        } else if (arg == "--tensix-noise") {
            use_tensix_noise = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') tensix_inv = std::stoul(argv[++i]);
            if (i + 1 < argc && argv[i + 1][0] != '-') tensix_start_batch = std::stoul(argv[++i]);
        } else if (arg == "--use-3piece" || arg == "--amin-3p") {
            g_use_3piece_amin = true;
        } else if (arg == "--use-2piece" || arg == "--amin-2p") {
            g_use_3piece_amin = false;
        } else if (arg == "--no-ch-clip" || arg == "--unclipped") {
            l_ch_max = 0.0f;
        } else if (arg == "--ch-clip" || arg == "--l-ch-max") {
            if (i + 1 < argc) l_ch_max = std::stof(argv[++i]);
        }
    }

    std::string sched_str = "Intra-Batch Pipelined Ping-Pong DAE (Delta-R Accumulation, 4 Codewords)";
    if (g_schedule_mode == ScheduleMode::INTER_BATCH) {
        sched_str = "Inter-Batch Ping-Pong DAE (Zero-Hazard Fast Direct Store, 8 Codewords)";
    } else if (g_schedule_mode == ScheduleMode::SYNC) {
        sched_str = "Synchronous Stop-and-Wait (Diagnostic Reference, 4 Codewords)";
    }

    std::cout << "\n=================================================================" << std::endl;
    std::cout << " TENSIX 40-TILE FACE-ALIGNED LOCKSTEP STEPPER: MONTE CARLO" << std::endl;
    std::cout << " Precision Mode:    " << (tensix_precision ? "Tensix Hardware Bit-Exact (bfloat16 mantissa truncation)" : "Full FP32 Precision") << std::endl;
    std::cout << " Schedule Mode:     " << sched_str << std::endl;
    std::cout << " Approx-Min* Model: " << (g_use_3piece_amin ? "Optimal 3-Piece Convex Hull (Phase 3 Enhanced)" : "Certified 2-Piece Baseline (Dr. Jones MILCOM 2003)") << std::endl;
    std::cout << " Noise Generator:   " << (use_tensix_noise ? ("Tensix Vector Box-Muller (Invocation: " + std::to_string(tensix_inv) + ", Start Batch: " + std::to_string(tensix_start_batch) + ")") : "Host Scalar Xoshiro128+") << std::endl;
    if (use_tensix_noise) {
        std::cout << " Channel Clamping:  " << (l_ch_max > 0.0f ? ("|L_ch| <= " + std::to_string(l_ch_max)) : "None (Unclipped)") << std::endl;
    }
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

    std::atomic<uint32_t> global_batch_counter{0};

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        Xoshiro128Plus rng;
        uint32_t seed = 1234567U + static_cast<uint32_t>(eb_n0_db * 100.0) * 10007U;
        rng.init(seed);
        for (int i = 0; i < tid; ++i) rng.jump();

        std::vector<uint8_t> vnode_l1_a(20480, 0);
        std::vector<uint8_t> vnode_l1_b(20480, 0);
        std::vector<std::vector<uint16_t>> r_msg_fifo_a(schedule.size(), std::vector<uint16_t>(1024, 0));
        std::vector<std::vector<uint16_t>> r_msg_fifo_b(schedule.size(), std::vector<uint16_t>(1024, 0));
        std::vector<uint16_t> scratch_in[2] = {std::vector<uint16_t>(1024, 0), std::vector<uint16_t>(1024, 0)};
        std::vector<uint16_t> scratch_out[2] = {std::vector<uint16_t>(1024, 0), std::vector<uint16_t>(1024, 0)};

        uint32_t cws_per_step = (g_schedule_mode == ScheduleMode::INTER_BATCH) ? 8 : 4;
        uint32_t chunk_batches = (eb_n0_db < 1.7) ? 25 : std::min(250U, std::max(1U, static_cast<uint32_t>((max_blocks + cws_per_step - 1) / cws_per_step / num_threads + 1)));

        while (!stop_flag.load(std::memory_order_relaxed)) {
            uint32_t local_fe = 0;
            uint32_t local_be = 0;
            uint64_t local_cw_iters = 0;

            for (uint32_t b = 0; b < chunk_batches; ++b) {
                if (g_schedule_mode == ScheduleMode::INTER_BATCH) {
                    for (auto& fifo : r_msg_fifo_a) std::fill(fifo.begin(), fifo.end(), 0);
                    for (auto& fifo : r_msg_fifo_b) std::fill(fifo.begin(), fifo.end(), 0);

                    if (use_tensix_noise) {
                        uint32_t b_idx_a = tensix_start_batch + global_batch_counter.fetch_add(1, std::memory_order_relaxed);
                        uint32_t b_idx_b = tensix_start_batch + global_batch_counter.fetch_add(1, std::memory_order_relaxed);
                        TensixBM::generate_channel_vnodes(vnode_l1_a.data(), mu_llr, sigma_llr, tensix_inv, b_idx_a, l_ch_max);
                        TensixBM::generate_channel_vnodes(vnode_l1_b.data(), mu_llr, sigma_llr, tensix_inv, b_idx_b, l_ch_max);
                    } else {
                        init_vnodes_host(vnode_l1_a.data(), mu_llr, sigma_llr, rng);
                        init_vnodes_host(vnode_l1_b.data(), mu_llr, sigma_llr, rng);
                    }

                    DualBatchStats stats = decode_tile_matrix_batch_inter_batch(
                        pcm, schedule, vnode_l1_a.data(), vnode_l1_b.data(),
                        r_msg_fifo_a, r_msg_fifo_b, scratch_in, scratch_out,
                        max_iter, post_clip, tensix_precision
                    );

                    for (int c = 0; c < 4; ++c) {
                        local_cw_iters += stats.a.iters_taken[c];
                        if (stats.a.bit_errors[c] > 0 || !stats.a.converged[c]) {
                            local_fe++;
                            local_be += stats.a.bit_errors[c];
                        }
                    }
                    for (int c = 0; c < 4; ++c) {
                        local_cw_iters += stats.b.iters_taken[c];
                        if (stats.b.bit_errors[c] > 0 || !stats.b.converged[c]) {
                            local_fe++;
                            local_be += stats.b.bit_errors[c];
                        }
                    }
                } else {
                    for (auto& fifo : r_msg_fifo_a) std::fill(fifo.begin(), fifo.end(), 0);

                    if (use_tensix_noise) {
                        uint32_t b_idx = tensix_start_batch + global_batch_counter.fetch_add(1, std::memory_order_relaxed);
                        TensixBM::generate_channel_vnodes(vnode_l1_a.data(), mu_llr, sigma_llr, tensix_inv, b_idx, l_ch_max);
                    } else {
                        init_vnodes_host(vnode_l1_a.data(), mu_llr, sigma_llr, rng);
                    }

                    BatchStats stats = decode_tile_matrix_batch(
                        pcm, schedule, vnode_l1_a.data(), r_msg_fifo_a, scratch_in, scratch_out, max_iter, post_clip, tensix_precision
                    );

                    for (int c = 0; c < 4; ++c) {
                        local_cw_iters += stats.iters_taken[c];
                        if (stats.bit_errors[c] > 0 || !stats.converged[c]) {
                            local_fe++;
                            local_be += stats.bit_errors[c];
                        }
                    }
                }
            }

            uint64_t added_cws = chunk_batches * cws_per_step;
            uint64_t cur_cws = total_codewords.fetch_add(added_cws, std::memory_order_relaxed) + added_cws;
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

    if (fs::path(csv_path).has_parent_path()) {
        fs::create_directories(fs::path(csv_path).parent_path());
    }
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

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "\n=================================================================\n"
                      << " TENSIX 40-TILE FACE-ALIGNED STEPPER: USAGE & OPTIONS\n"
                      << "=================================================================\n\n"
                      << "SCHEDULE MODES (How Tensix Hardware Stepping is Executed):\n"
                      << "  --pipelined          [DEFAULT] Intra-Batch Pipelined Ping-Pong DAE Engine.\n"
                      << "                       Overlaps Gather(k), Compute(k-1), Scatter(k-2) using\n"
                      << "                       double-buffered L1 scratchpads and Delta-R accumulation.\n"
                      << "                       Replicates physical Blackhole DAE execution (~3,580 cycles/tile).\n"
                      << "  --inter-batch        Inter-Batch Ping-Pong DAE Engine (Section 7.1).\n"
                      << "                       Overlaps Gather/Scatter of Batch A with Compute of Batch B\n"
                      << "                       using ping-pong scratchpads. Zero RAW hazard between adjacent steps,\n"
                      << "                       enabling pure 64-bit direct stores without RMW addition.\n"
                      << "                       Processes 8 codewords concurrently (4 in Batch A + 4 in Batch B).\n"
                      << "  --inter-batch-ping-pong  Alias for --inter-batch.\n"
                      << "  --pipelined-inter-batch  Alias for --inter-batch.\n"
                      << "  --intra-batch        Alias for --pipelined.\n"
                      << "  --sync               Synchronous Stop-and-Wait Engine.\n"
                      << "                       Executes Gather -> Compute -> Scatter strictly per tile.\n"
                      << "                       Single-buffered canonical Gauss-Seidel diagnostic reference.\n"
                      << "  --lockstep           Backwards-compatible alias for --sync.\n"
                      << "  --stop-and-wait      Alias for --sync.\n\n"
                      << "EXECUTION MODES:\n"
                      << "  --monte-carlo <ebn0> <blocks> <min_fe> <max_iter> <threads> <post_clip> <csv_out>\n"
                      << "                       Multi-threaded host Monte Carlo simulation.\n"
                      << "  --standalone <ebn0> <max_iter> <seed>\n"
                      << "                       Single-batch AWGN test across all 4 companion codewords.\n"
                      << "  --tensix-noise [inv] [batch] [ebn0] [max_iter]\n"
                      << "                       Bit-exact hardware Box-Muller noise replay.\n"
                      << "  --passthrough [iters]\n"
                      << "                       Validates gather/scatter data movement without compute.\n"
                      << "  --test-r-dataflow [iters]\n"
                      << "                       Validates 40-page circular FIFO check message flow.\n\n"
                      << "PRECISION & CLIPPING OPTIONS:\n"
                      << "  --fp32 / --full-precision    Use 32-bit floating point precision (default: Tensix bfloat16).\n"
                      << "  --post-clip <val>            Clip updated variable LLRs: [-val, +val].\n"
                      << "  --ch-clip <val>              Clip initial channel observations: [-val, +val] (default: 7.0).\n"
                      << "  --no-ch-clip / --unclipped   Disable channel LLR clipping.\n"
                      << "=================================================================\n\n";
            return 0;
        }
    }

    bool is_monte_carlo = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--monte-carlo") {
            is_monte_carlo = true;
            break;
        }
    }
    if (is_monte_carlo) {
        return run_monte_carlo_simulation(argc, argv, pcm, schedule);
    }

    std::string tensix_init_path  = "engines/tile_matrix_4cw/dump/tensix_initial_vnodes.bin";
    std::string tensix_final_path = "engines/tile_matrix_4cw/dump/tensix_final_vnodes.bin";
    if (!fs::exists(tensix_init_path) && fs::exists("dump/tensix_initial_vnodes.bin")) {
        tensix_init_path  = "dump/tensix_initial_vnodes.bin";
        tensix_final_path = "dump/tensix_final_vnodes.bin";
    }

    bool standalone_mode = false;
    bool passthrough_mode = false;
    bool r_dataflow_mode = false;
    double eb_n0_db = 2.5;
    uint32_t num_iterations = 16;
    uint32_t seed = 12345;
    uint32_t num_tiles_to_run = schedule.size();
    float post_clip_max = 0.0f;

    bool tensix_noise_mode = false;
    uint32_t tensix_noise_inv = 1;
    uint32_t tensix_noise_batch = 0;
    float l_ch_max = 7.0f;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--passthrough") {
            passthrough_mode = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') num_iterations = std::stoul(argv[++i]);
        } else if (arg == "--test-r-dataflow") {
            r_dataflow_mode = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') num_iterations = std::stoul(argv[++i]);
        } else if (arg == "--tiles" && i + 1 < argc) {
            num_tiles_to_run = std::stoul(argv[++i]);
        } else if (arg == "--pipelined" || arg == "--intra-batch") {
            g_schedule_mode = ScheduleMode::PIPELINED;
        } else if (arg == "--inter-batch" || arg == "--inter-batch-ping-pong" || arg == "--pipelined-inter-batch") {
            g_schedule_mode = ScheduleMode::INTER_BATCH;
        } else if (arg == "--sync" || arg == "--stop-and-wait" || arg == "--lockstep") {
            g_schedule_mode = ScheduleMode::SYNC;
        } else if (arg == "--tensix-noise") {
            tensix_noise_mode = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') tensix_noise_inv = std::stoul(argv[++i]);
            if (i + 1 < argc && argv[i + 1][0] != '-') tensix_noise_batch = std::stoul(argv[++i]);
            if (i + 1 < argc && argv[i + 1][0] != '-') eb_n0_db = std::stod(argv[++i]);
            if (i + 1 < argc && argv[i + 1][0] != '-') num_iterations = std::stoul(argv[++i]);
        } else if (arg == "--no-ch-clip" || arg == "--unclipped") {
            l_ch_max = 0.0f;
        } else if (arg == "--ch-clip" || arg == "--l-ch-max") {
            if (i + 1 < argc) l_ch_max = std::stof(argv[++i]);
        } else if (arg == "--ebn0" && i + 1 < argc) {
            eb_n0_db = std::stod(argv[++i]);
        } else if (arg == "--max-iter" && i + 1 < argc) {
            num_iterations = std::stoul(argv[++i]);
        } else if (arg == "--post-clip" && i + 1 < argc) {
            post_clip_max = std::stof(argv[++i]);
        }
    }

    if (argc > 1 && std::string(argv[1]) == "--standalone") {
        standalone_mode = true;
        if (argc > 2) eb_n0_db = std::stod(argv[2]);
        if (argc > 3) num_iterations = std::stoul(argv[3]);
        if (argc > 4) seed = std::stoul(argv[4]);
    } else if (!passthrough_mode && !r_dataflow_mode && !tensix_noise_mode) {
        if (argc > 1 && argv[1][0] != '-') num_iterations = std::stoul(argv[1]);
        if (argc > 2 && argv[2][0] != '-') tensix_init_path = argv[2];
        if (argc > 3 && argv[3][0] != '-') tensix_final_path = argv[3];
    }

    bool tensix_precision = true;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--fp32" || std::string(argv[i]) == "--full-precision") {
            tensix_precision = false;
        } else if (std::string(argv[i]) == "--tensix-precision") {
            tensix_precision = true;
        } else if (std::string(argv[i]) == "--use-3piece" || std::string(argv[i]) == "--amin-3p") {
            g_use_3piece_amin = true;
        } else if (std::string(argv[i]) == "--use-2piece" || std::string(argv[i]) == "--amin-2p") {
            g_use_3piece_amin = false;
        }
    }

    std::string sched_str_main = "Intra-Batch Pipelined Ping-Pong DAE (Delta-R Accumulation, 4 Codewords)";
    if (g_schedule_mode == ScheduleMode::INTER_BATCH) {
        sched_str_main = "Inter-Batch Ping-Pong DAE (Zero-Hazard Fast Direct Store, 8 Codewords)";
    } else if (g_schedule_mode == ScheduleMode::SYNC) {
        sched_str_main = "Synchronous Stop-and-Wait (Diagnostic Reference, 4 Codewords)";
    }

    std::cout << "=================================================================" << std::endl;
    std::cout << " TENSIX 40-TILE FACE-ALIGNED LOCKSTEP STEPPER" << std::endl;
    std::cout << " Precision Mode:    " << (tensix_precision ? "Tensix Hardware Bit-Exact (bfloat16 mantissa truncation)" : "Full FP32 Precision") << std::endl;
    std::cout << " Schedule Mode:     " << sched_str_main << std::endl;
    std::cout << " Approx-Min* Model: " << (g_use_3piece_amin ? "Optimal 3-Piece Convex Hull (Phase 3 Enhanced)" : "Certified 2-Piece Baseline (Dr. Jones MILCOM 2003)") << std::endl;
    if (passthrough_mode) {
        std::cout << " Mode:              IDENTITY PASS-THROUGH (R_out=R_in, L_out=L_in)" << std::endl;
        std::cout << " Vnode Pattern:     OFFSET COUNTING NUMBERS (1024..3583)" << std::endl;
        std::cout << " R Edge Values:     ZERO (Unmodified)" << std::endl;
        std::cout << " Target Iterations: " << num_iterations << std::endl;
        std::cout << " Tiles to Run:      " << num_tiles_to_run << " / " << schedule.size() << std::endl;
    } else if (r_dataflow_mode) {
        std::cout << " Mode:              R-DATAFLOW VERIFICATION ((V - R_in) + R_out)" << std::endl;
        std::cout << " Vnode Pattern:     64.0 + (v % 32) (100% exact in bfloat16)" << std::endl;
        std::cout << " R Edge Values:     1.0 + r (Edge index r in [0..deg-1])" << std::endl;
        std::cout << " Target Iterations: " << num_iterations << std::endl;
        std::cout << " Tiles to Run:      " << num_tiles_to_run << " / " << schedule.size() << std::endl;
    } else if (tensix_noise_mode) {
        std::cout << " Mode:              TENSIX BIT-EXACT NOISE REPLAY" << std::endl;
        std::cout << " Invocation ID:     " << tensix_noise_inv << std::endl;
        std::cout << " Batch Index:       " << tensix_noise_batch << std::endl;
        std::cout << " Eb/N0:             " << eb_n0_db << " dB" << std::endl;
        std::cout << " Channel Clamping:  " << (l_ch_max > 0.0f ? ("|L_ch| <= " + std::to_string(l_ch_max)) : "None (Unclipped)") << std::endl;
        std::cout << " Target Iterations: " << num_iterations << std::endl;
        std::cout << " Post-Clipping:     " << (post_clip_max > 0.0f ? std::to_string(post_clip_max) : "None (Unclipped)") << std::endl;
    } else if (standalone_mode) {
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
    std::vector<uint8_t> iter1_vnodes(20480, 0);

    if (passthrough_mode) {
        for (uint32_t v = 0; v < 2560; ++v) {
            uint32_t off = get_tile_vnode_byte_offset(v);
            uint16_t* p = reinterpret_cast<uint16_t*>(vnode_l1.data() + off);
            uint16_t val = static_cast<uint16_t>(1024 + v);
            p[0] = val;
            p[1] = val;
            p[2] = val;
            p[3] = val;
        }
        fs::create_directories("dump");
        std::ofstream f_init("dump/golden_initial_vnodes.bin", std::ios::binary);
        f_init.write(reinterpret_cast<const char*>(vnode_l1.data()), 20480);
        f_init.close();
        std::cout << "[INFO] Initialized 2,560 Vnodes with offset counting numbers (1024..3583) across all 4 codewords." << std::endl;
        std::cout << "[INFO] Dumped initial golden Vnodes to dump/golden_initial_vnodes.bin" << std::endl;
    } else if (r_dataflow_mode) {
        for (uint32_t v = 0; v < 2560; ++v) {
            uint32_t off = get_tile_vnode_byte_offset(v);
            uint16_t* p = reinterpret_cast<uint16_t*>(vnode_l1.data() + off);
            float sign = ((v / 2) % 2 == 1) ? -1.0f : 1.0f;
            float mag  = 2.0f + 0.25f * static_cast<float>(v % 8);
            float fval = sign * mag;
            uint16_t val = fp32_to_bf16(fval);
            p[0] = val;
            p[1] = val;
            p[2] = val;
            p[3] = val;
        }
        fs::create_directories("dump");
        std::ofstream f_init("dump/golden_initial_vnodes.bin", std::ios::binary);
        f_init.write(reinterpret_cast<const char*>(vnode_l1.data()), 20480);
        f_init.close();
        std::cout << "[INFO] Initialized 2,560 Vnodes with probing pattern (-1)^(v/2) * (2.0 + 0.25*(v%8)) across all 4 codewords." << std::endl;
        std::cout << "[INFO] Dumped initial golden Vnodes to dump/golden_initial_vnodes.bin" << std::endl;
    } else if (tensix_noise_mode) {
        float code_rate = static_cast<float>(pcm.K) / (pcm.N - pcm.P); // 1024 / 2048 = 0.5
        float eb_n0_lin = std::pow(10.0f, static_cast<float>(eb_n0_db) / 10.0f);
        float es_n0 = code_rate * eb_n0_lin;
        float noise_var = 1.0f / (2.0f * es_n0);
        float noise_std = std::sqrt(noise_var);
        float mu_llr = 2.0f / noise_var;
        float sigma_llr = 2.0f / noise_std;

        TensixBM::generate_channel_vnodes(vnode_l1.data(), mu_llr, sigma_llr, tensix_noise_inv, tensix_noise_batch, l_ch_max);
        std::cout << "[INFO] Synthesized Bit-Exact Tensix Vector Box-Muller Channel LLRs (mu=" << mu_llr << ", sigma=" << sigma_llr << ")" << std::endl;
    } else if (standalone_mode) {
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

    // Scratchpad Tile Buffers (Double-Buffered for Pipelined Ping-Pong)
    std::vector<uint16_t> scratch_in[2] = {std::vector<uint16_t>(1024, 0), std::vector<uint16_t>(1024, 0)};
    std::vector<uint16_t> scratch_out[2] = {std::vector<uint16_t>(1024, 0), std::vector<uint16_t>(1024, 0)};

    const uint32_t* p_hoff_base = pcm.h_offsets_pairs.data();

    // Print Initial Error State before decoding (only for AWGN / BPSK modes)
    if (!passthrough_mode) {
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
    // Execution Loop: Pipelined Ping-Pong DAE or Synchronous Stop-and-Wait
    // --------------------------------------------------------------------------
    uint8_t active_mask = 0x0F;

    if (g_schedule_mode == ScheduleMode::INTER_BATCH && !passthrough_mode && !r_dataflow_mode) {
        std::vector<uint8_t> vnode_l1_a = vnode_l1;
        std::vector<uint8_t> vnode_l1_b(20480, 0);

        if (standalone_mode) {
            float code_rate = static_cast<float>(pcm.K) / (pcm.N - pcm.P);
            float eb_n0_lin = std::pow(10.0f, static_cast<float>(eb_n0_db) / 10.0f);
            float es_n0 = code_rate * eb_n0_lin;
            float noise_var = 1.0f / (2.0f * es_n0);
            float noise_std = std::sqrt(noise_var);
            float mu_llr = 2.0f / noise_var;
            float sigma_llr = 2.0f / noise_std;

            Xoshiro128Plus rng_b;
            rng_b.init(seed);
            rng_b.jump();
            init_vnodes_host(vnode_l1_b.data(), mu_llr, sigma_llr, rng_b);

            uint32_t init_info_errs_b[4] = {0, 0, 0, 0};
            uint32_t init_total_neg_b[4] = {0, 0, 0, 0};
            for (uint32_t v = 0; v < 2048; ++v) {
                uint32_t off = get_tile_vnode_byte_offset(v);
                const uint16_t* p = reinterpret_cast<const uint16_t*>(vnode_l1_b.data() + off);
                for (int c = 0; c < 4; ++c) {
                    if (bf16_to_fp32(p[c]) < 0.0f) {
                        init_total_neg_b[c]++;
                        if (v < 1024) init_info_errs_b[c]++;
                    }
                }
            }
            std::cout << "[Initial State Batch B] Info BE: [" << init_info_errs_b[0] << ", " << init_info_errs_b[1] << ", " << init_info_errs_b[2] << ", " << init_info_errs_b[3] << "] | "
                      << "Tx Neg: [" << init_total_neg_b[0] << ", " << init_total_neg_b[1] << ", " << init_total_neg_b[2] << ", " << init_total_neg_b[3] << "]"
                      << std::endl;
        }

        std::vector<std::vector<uint16_t>> r_msg_fifo_a(schedule.size(), std::vector<uint16_t>(1024, 0));
        std::vector<std::vector<uint16_t>> r_msg_fifo_b(schedule.size(), std::vector<uint16_t>(1024, 0));
        std::vector<uint16_t> scratch_in_dual[2] = {std::vector<uint16_t>(1024, 0), std::vector<uint16_t>(1024, 0)};
        std::vector<uint16_t> scratch_out_dual[2] = {std::vector<uint16_t>(1024, 0), std::vector<uint16_t>(1024, 0)};

        uint8_t dual_active_mask[2] = {0x0F, 0x0F};
        uint8_t* vnodes[2] = {vnode_l1_a.data(), vnode_l1_b.data()};
        std::vector<std::vector<uint16_t>>* r_fifos[2] = {&r_msg_fifo_a, &r_msg_fifo_b};
        uint32_t num_tiles = schedule.size();
        uint32_t max_steps = num_iterations * num_tiles;

        for (uint32_t k = 0; k < 2 * max_steps + 2; ++k) {
            uint32_t b_dae = k % 2;
            uint32_t s_dae = k / 2;

            // 1. DAE Scatter Stage: Scatter Tile (s_dae - 1) of Batch b_dae
            if (s_dae >= 1 && (s_dae - 1) < max_steps) {
                uint32_t s_scat = s_dae - 1;
                uint32_t t_scat = s_scat % num_tiles;
                const uint32_t* tile_hoff_w0 = p_hoff_base + t_scat * 96;
                const uint32_t* tile_hoff_w1 = tile_hoff_w0 + 48;

                if (dual_active_mask[b_dae] != 0) {
                    step_tile_scatter(scratch_out_dual[b_dae], vnodes[b_dae], tile_hoff_w0, tile_hoff_w1, dual_active_mask[b_dae], false /* direct store */, post_clip_max);
                }

                // Parity Convergence Evaluation & Telemetry when Tile 39 finishes scatter
                if (t_scat == num_tiles - 1) {
                    uint32_t iter = s_scat / num_tiles;
                    uint32_t info_errs[4] = {0, 0, 0, 0};
                    uint32_t total_neg[4] = {0, 0, 0, 0};
                    for (uint32_t v = 0; v < 2048; ++v) {
                        uint32_t off = get_tile_vnode_byte_offset(v);
                        const uint16_t* p = reinterpret_cast<const uint16_t*>(vnodes[b_dae] + off);
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
                            const uint16_t* p = reinterpret_cast<const uint16_t*>(vnodes[b_dae] + off);
                            for (int c = 0; c < 4; ++c) {
                                if (bf16_to_fp32(p[c]) < 0.0f) syn[c] ^= 1;
                            }
                        }
                        for (int c = 0; c < 4; ++c) {
                            if (syn[c] != 0) unsat_checks[c]++;
                        }
                    }

                    std::string b_name = (b_dae == 0) ? "Batch A" : "Batch B";
                    std::cout << "[" << b_name << " Iter " << std::setw(2) << (iter + 1) << "] "
                              << "Info BE: [" << info_errs[0] << ", " << info_errs[1] << ", " << info_errs[2] << ", " << info_errs[3] << "] | "
                              << "Tx Neg: [" << total_neg[0] << ", " << total_neg[1] << ", " << total_neg[2] << ", " << total_neg[3] << "] | "
                              << "Unsat Checks: [" << unsat_checks[0] << ", " << unsat_checks[1] << ", " << unsat_checks[2] << ", " << unsat_checks[3] << "]"
                              << std::endl;

                    for (int c = 0; c < 4; ++c) {
                        if (unsat_checks[c] == 0 && info_errs[c] == 0) {
                            dual_active_mask[b_dae] &= ~(1 << c);
                        }
                    }
                }
            }

            // 2. DAE Gather Stage: Gather Tile s_dae of Batch b_dae
            if (s_dae < max_steps) {
                if (dual_active_mask[b_dae] != 0) {
                    uint32_t t_gath = s_dae % num_tiles;
                    const uint32_t* tile_hoff_w0 = p_hoff_base + t_gath * 96;
                    const uint32_t* tile_hoff_w1 = tile_hoff_w0 + 48;
                    step_tile_gather(scratch_in_dual[b_dae], vnodes[b_dae], tile_hoff_w0, tile_hoff_w1);
                }
            }

            // 3. TRISC Compute Stage: Compute Tile s_comp of Batch b_comp
            if (k >= 1) {
                uint32_t b_comp = (k - 1) % 2;
                uint32_t s_comp = (k - 1) / 2;
                if (s_comp < max_steps && dual_active_mask[b_comp] != 0) {
                    uint32_t t_comp = s_comp % num_tiles;
                    uint32_t iter_comp = s_comp / num_tiles;
                    uint32_t deg = schedule[t_comp].check_degree;
                    step_tile_compute(scratch_in_dual[b_comp], scratch_out_dual[b_comp], (*r_fifos[b_comp])[t_comp], deg, iter_comp, post_clip_max, tensix_precision, false /* direct L_new */);
                }
            }

            if (dual_active_mask[0] == 0 && dual_active_mask[1] == 0) {
                std::cout << ">>> [CONVERGENCE] All 8 codewords across Batches A & B reached 0 bit errors and 0 syndrome failures!" << std::endl;
                if (standalone_mode) break;
            }
        }
    } else if (g_schedule_mode == ScheduleMode::PIPELINED && !passthrough_mode && !r_dataflow_mode) {
        uint32_t num_tiles = schedule.size();
        uint32_t max_steps = num_iterations * num_tiles;

        for (uint32_t k = 0; k < max_steps + 2; ++k) {
            uint32_t b_in   = k % 2;
            uint32_t b_comp = (k - 1) % 2;
            uint32_t b_scat = (k - 2) % 2;

            uint32_t t_gath = k % num_tiles;
            uint32_t t_comp = (k >= 1) ? ((k - 1) % num_tiles) : 0;
            uint32_t t_scat = (k >= 2) ? ((k - 2) % num_tiles) : 0;

            // 1. Scatter Stage: Scatter Tile t_scat (Enabled for k >= 2)
            if (k >= 2) {
                const uint32_t* tile_hoff_w0 = p_hoff_base + t_scat * 96;
                const uint32_t* tile_hoff_w1 = tile_hoff_w0 + 48;
                step_tile_scatter(scratch_out[b_scat], vnode_l1.data(), tile_hoff_w0, tile_hoff_w1, active_mask, true, post_clip_max);
            }

            // 2. Gather Stage: Gather Tile t_gath (Enabled for k < max_steps)
            if (k < max_steps) {
                const uint32_t* tile_hoff_w0 = p_hoff_base + t_gath * 96;
                const uint32_t* tile_hoff_w1 = tile_hoff_w0 + 48;
                step_tile_gather(scratch_in[b_in], vnode_l1.data(), tile_hoff_w0, tile_hoff_w1);

                if (k == 0) {
                    fs::create_directories("dump");
                    std::ofstream f_gold_scat_in("dump/golden_scratch_in.bin", std::ios::binary);
                    f_gold_scat_in.write(reinterpret_cast<const char*>(scratch_in[b_in].data()), 2048);
                    f_gold_scat_in.close();
                }
            }

            // 3. Compute Stage: Compute Tile t_comp (Enabled for 1 <= k <= max_steps)
            if (k >= 1 && k <= max_steps) {
                uint32_t deg = schedule[t_comp].check_degree;
                uint32_t iter = (k - 1) / num_tiles;
                step_tile_compute(scratch_in[b_comp], scratch_out[b_comp], r_msg_fifo[t_comp], deg, iter, post_clip_max, tensix_precision, true);

                if (k == 1) {
                    std::ofstream f_gold_scat_out("dump/golden_scratch_out.bin", std::ios::binary);
                    f_gold_scat_out.write(reinterpret_cast<const char*>(scratch_out[b_comp].data()), 2048);
                    f_gold_scat_out.close();
                }
            }

            // 4. Parity Convergence Evaluation & Telemetry (Triggered when Tile 39 finishes scatter)
            if (k >= 2 && t_scat == num_tiles - 1) {
                uint32_t iter = (k - 2) / num_tiles;
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

                for (int c = 0; c < 4; ++c) {
                    if (unsat_checks[c] == 0 && info_errs[c] == 0) {
                        active_mask &= ~(1 << c);
                    }
                }

                if (active_mask == 0) {
                    std::cout << ">>> [CONVERGENCE] All 4 codewords reached 0 bit errors and 0 syndrome failures at iteration " << (iter + 1) << "!" << std::endl;
                    if (standalone_mode) break;
                }
            }
        }
    } else {
        // Synchronous stop-and-wait loop (Diagnostic Reference & Single-Tile Tests)
        for (uint32_t iter = 0; iter < num_iterations; ++iter) {
            for (size_t t = 0; t < num_tiles_to_run; ++t) {
                uint32_t deg = schedule[t].check_degree;
                const uint32_t* tile_hoff_w0 = p_hoff_base + t * 96;
                const uint32_t* tile_hoff_w1 = tile_hoff_w0 + 48;

                // 1. Gather Stage
                step_tile_gather(scratch_in[0], vnode_l1.data(), tile_hoff_w0, tile_hoff_w1);

                if (iter == 0 && t == 0) {
                    fs::create_directories("dump");
                    std::ofstream f_gold_scat_in("dump/golden_scratch_in.bin", std::ios::binary);
                    f_gold_scat_in.write(reinterpret_cast<const char*>(scratch_in[0].data()), 2048);
                    f_gold_scat_in.close();
                }

                // 2. Compute Stage
                if (passthrough_mode) {
                    scratch_out[0] = scratch_in[0];
                } else if (r_dataflow_mode) {
                    const int face_bases[4] = {0, 8, 16, 24};
                    const float deg6_r0[3] = {1.0f, 3.0f, 5.0f};
                    const float deg6_r1[3] = {2.0f, 4.0f, 6.0f};
                    const float deg3_r0[3] = {1.0f, 3.0f, 2.0f};
                    const float deg3_r1[3] = {2.0f, 1.0f, 3.0f};
                    const float* k_r0 = (deg == 6) ? deg6_r0 : deg3_r0;
                    const float* k_r1 = (deg == 6) ? deg6_r1 : deg3_r1;

                    for (int f = 0; f < 4; ++f) {
                        int base = face_bases[f];
                        for (int k = 0; k < 3; ++k) {
                            float r0 = k_r0[k];
                            float r1 = k_r1[k];
                            for (int sub = 0; sub < 2; ++sub) {
                                uint32_t reg = base + 2 * k + sub;
                                for (uint32_t lane = 0; lane < 32; ++lane) {
                                    float r_val = (lane < 16) ? r0 : r1;
                                    uint32_t offset = get_scratch_byte_offset(reg, lane);
                                    float l_val = bf16_to_fp32(*reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(scratch_in[0].data()) + offset));
                                    float r_old = (iter == 0) ? 0.0f : bf16_to_fp32(*reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(r_msg_fifo[t].data()) + offset));
                                    float q = to_bf16(l_val - r_old);
                                    float r_new = (iter == 0) ? to_bf16(r_val) : r_old;
                                    float l_new = to_bf16(q + r_new);

                                    *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(r_msg_fifo[t].data()) + offset) = fp32_to_bf16(r_new);
                                    *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(scratch_out[0].data()) + offset)   = fp32_to_bf16(l_new);
                                }
                            }
                        }
                        for (int sub = 0; sub < 2; ++sub) {
                            uint32_t reg = base + 6 + sub;
                            for (uint32_t lane = 0; lane < 32; ++lane) {
                                uint32_t offset = get_scratch_byte_offset(reg, lane);
                                *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(r_msg_fifo[t].data()) + offset) = 0;
                                *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(scratch_out[0].data()) + offset)   = 0;
                            }
                        }
                    }
                } else {
                    step_tile_compute(scratch_in[0], scratch_out[0], r_msg_fifo[t], deg, iter, post_clip_max, tensix_precision);
                }

                if (iter == 0 && t == 0) {
                    std::ofstream f_gold_scat_out("dump/golden_scratch_out.bin", std::ios::binary);
                    f_gold_scat_out.write(reinterpret_cast<const char*>(scratch_out[0].data()), 2048);
                    f_gold_scat_out.close();
                }

                // 3. Scatter Stage
                step_tile_scatter(scratch_out[0], vnode_l1.data(), tile_hoff_w0, tile_hoff_w1, active_mask);
            }

        if (passthrough_mode) {
            uint32_t mismatches = 0;
            for (uint32_t v = 0; v < 2560; ++v) {
                uint32_t off = get_tile_vnode_byte_offset(v);
                const uint16_t* p = reinterpret_cast<const uint16_t*>(vnode_l1.data() + off);
                for (int c = 0; c < 4; ++c) {
                    if (p[c] != static_cast<uint16_t>(1024 + v)) {
                        if (mismatches < 25) {
                            std::cout << "  [PASSTHROUGH MISMATCH] Iter " << (iter + 1)
                                      << " | Vnode " << std::setw(4) << v << " CW" << c
                                      << " (byte offset 0x" << std::hex << off << std::dec << "): "
                                      << "Expected " << (1024 + v) << ", got " << p[c] << std::endl;
                        }
                        mismatches++;
                    }
                }
            }
            if (mismatches == 0) {
                std::cout << "[Iter " << std::setw(2) << (iter + 1) << "] "
                          << "PASSTHROUGH CHECK: 100% BIT-EXACT MATCH! (0 / 10,240 errors across all 2,560 Vnodes)"
                          << std::endl;
            } else {
                std::cout << "[Iter " << std::setw(2) << (iter + 1) << "] "
                          << "PASSTHROUGH CHECK: FAILED with " << mismatches << " / 10,240 corrupted elements!"
                          << std::endl;
            }
        } else if (r_dataflow_mode) {
            if (iter == 0) {
                // Snapshot iteration 1 state as baseline
                iter1_vnodes = vnode_l1;
                fs::create_directories("dump");
                std::ofstream f_base("dump/golden_r_dataflow_iter1_vnodes.bin", std::ios::binary);
                f_base.write(reinterpret_cast<const char*>(iter1_vnodes.data()), 20480);
                f_base.close();
                std::cout << "[Iter " << std::setw(2) << (iter + 1) << "] "
                          << "R-DATAFLOW SNAPSHOT: Post-iteration 1 state stored as baseline (dump/golden_r_dataflow_iter1_vnodes.bin)."
                          << std::endl;
            } else {
                uint32_t mismatches = 0;
                for (uint32_t v = 0; v < 2560; ++v) {
                    uint32_t off = get_tile_vnode_byte_offset(v);
                    const uint16_t* p = reinterpret_cast<const uint16_t*>(vnode_l1.data() + off);
                    const uint16_t* p1 = reinterpret_cast<const uint16_t*>(iter1_vnodes.data() + off);
                    for (int c = 0; c < 4; ++c) {
                        if (p[c] != p1[c]) {
                            if (mismatches < 25) {
                                std::cout << "  [R-DATAFLOW MISMATCH] Iter " << (iter + 1)
                                          << " | Vnode " << std::setw(4) << v << " CW" << c
                                          << " (byte offset 0x" << std::hex << off << std::dec << "): "
                                          << "Expected 0x" << std::hex << p1[c] << " (" << bf16_to_fp32(p1[c]) << "), "
                                          << "got 0x" << p[c] << " (" << bf16_to_fp32(p[c]) << ")" << std::dec << std::endl;
                            }
                            mismatches++;
                        }
                    }
                }
                if (mismatches == 0) {
                    std::cout << "[Iter " << std::setw(2) << (iter + 1) << "] "
                              << "R-DATAFLOW CHECK: 100% BIT-EXACT MATCH! (0 / 10,240 errors vs Iteration 1)"
                              << std::endl;
                } else {
                    std::cout << "[Iter " << std::setw(2) << (iter + 1) << "] "
                              << "R-DATAFLOW CHECK: FAILED with " << mismatches << " / 10,240 corrupted elements vs Iteration 1!"
                              << std::endl;
                }
            }
        } else {
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

            for (int c = 0; c < 4; ++c) {
                if (unsat_checks[c] == 0 && info_errs[c] == 0) {
                    active_mask &= ~(1 << c);
                }
            }

            if (active_mask == 0) {
                std::cout << ">>> [CONVERGENCE] All 4 codewords reached 0 bit errors and 0 syndrome failures at iteration " << (iter + 1) << "!" << std::endl;
                if (standalone_mode) break;
            }
        }
    }
}

    std::string golden_out_path = "dump/golden_lockstep_final_vnodes.bin";
    fs::create_directories("dump");
    std::ofstream f_out(golden_out_path, std::ios::binary);
    f_out.write(reinterpret_cast<const char*>(vnode_l1.data()), 20480);
    f_out.close();
    std::ofstream f_out_short("dump/golden_final_vnodes.bin", std::ios::binary);
    f_out_short.write(reinterpret_cast<const char*>(vnode_l1.data()), 20480);
    f_out_short.close();
    std::cout << "\n[SUCCESS] Wrote 40-Tile Face-Aligned Lockstep Golden Reference: " << golden_out_path << std::endl;

    if (!standalone_mode && !passthrough_mode && !tensix_noise_mode && fs::exists(tensix_final_path) && fs::exists("tools/diff_tensix_vs_emulator.py")) {
        std::cout << "\n=================================================================" << std::endl;
        std::cout << " RUNNING LOCKSTEP DIFFERENTIAL COMPARISON: CPU vs TENSIX SILICON" << std::endl;
        std::cout << "=================================================================" << std::endl;

        std::string cmd = "/home/ttuser/.tenstorrent-venv/bin/python tools/diff_tensix_vs_emulator.py " + golden_out_path + " " + tensix_final_path;
        int ret = std::system(cmd.c_str());
        (void)ret;
    }

    return 0;
}
