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
#include <mutex>
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

// Tile Matrix Core & SFPI Vector Emulation
#include "../engines/tile_matrix_4cw/ldpc_tile_matrix_core.h"
#include "../engines/tile_matrix_4cw/sfpi_host_emu.h"

// Section 8 Integration: Scalar 5W Golden Model Reference (Untouched in namespace)
namespace Scalar5w {
#include "../engines/scalar_5w/ldpc_decoder_core.h"
}

namespace fs = std::filesystem;

// ==============================================================================
// Tenstorrent QuietBox 2: 40-Tile Face-Aligned Stepper (5CW Performance Enhanced)
// Incorporates Section 8 Engineering Roadmap Recommendations:
//   1. Intrinsic Check Belief Damping (beta = 0.9375 by default)
//   2. Accumulator-Aware Contiguous Tile Scheduling (32 Deg-6 then 8 Deg-3)
//   3. Selective FP32 Intermediate Accumulation (Eliminates multi-stage bf16 rounding)
//   4. Active-Codeword Mask Scatter Freezing (Locks converged frames in L1 SRAM)
//
// Features Direct Head-to-Head Simulation vs. Untouched scalar_5w Model on 16 Cores.
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

// Section 8 Roadmap Enhancement 2: Configurable Accumulator-Aware Contiguous Schedule
std::vector<DynamicTileDescriptor> build_enhanced_tile_schedule(
    ParityCheckMatrix& pcm,
    bool contiguous_schedule = true
) {
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
                    c0_even = 0 + kh;
                    c0_odd  = 4 + kh;
                    c1_even = 8 + kh;
                    c1_odd  = 12 + kh;
                } else {
                    d = (p - 24) / 4;
                    kh = (p - 24) % 4;
                    c0_even = 16 + kh;
                    c0_odd  = 20 + kh;
                    c1_even = 24 + kh;
                    c1_odd  = 28 + kh;
                }
            } else {
                if (p < 24) {
                    uint32_t blk = p / 12;
                    d  = (p % 12) / 4;
                    kh = (p % 12) % 4;
                    c0_even = 0  + blk * 8 + kh;
                    c0_odd  = 0  + blk * 8 + 4 + kh;
                    c1_even = 16 + blk * 8 + kh;
                    c1_odd  = 16 + blk * 8 + 4 + kh;
                } else {
                    uint32_t blk = (p - 24) / 12;
                    d  = ((p - 24) % 12) / 4;
                    kh = ((p - 24) % 12) % 4;
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

        for (uint32_t p = 0; p < 48; ++p) pcm.h_offsets_pairs.push_back(w0_pairs[p]);
        for (uint32_t p = 0; p < 48; ++p) pcm.h_offsets_pairs.push_back(w1_pairs[p]);

        current_h_offset += 96 * sizeof(uint32_t);
        current_r_page   += 1;
        schedule.push_back(desc);
    };

    if (contiguous_schedule) {
        // Section 8 Roadmap 2: Contiguous schedule sweeps all 32 deg-6 tiles first,
        // then contiguously sweeps all 8 deg-3 accumulator tiles.
        for (size_t t = 0; t < 32; ++t) add_tile_to_schedule(deg6_tiles[t], 6);
        for (size_t t = 0; t < 8; ++t)  add_tile_to_schedule(deg3_tiles[t], 3);
    } else {
        // Legacy 4:1 interleaved schedule
        for (size_t i = 0; i < 8; ++i) {
            for (size_t j = 0; j < 4; ++j) add_tile_to_schedule(deg6_tiles[i * 4 + j], 6);
            add_tile_to_schedule(deg3_tiles[i], 3);
        }
    }

    return schedule;
}

// ------------------------------------------------------------------------------
// Bit-Exact SFPI Vector TRISC Check Node Compute (Enhanced with Section 8 Roadmap)
// ------------------------------------------------------------------------------
inline void compute_tile_deg6_sfpi_enhanced(
    uint32_t iter,
    float damping_factor = 0.9375f,
    bool fp32_interm = true
) {
    const int face_bases[4] = {0, 8, 16, 24};
    for (int f = 0; f < 4; ++f) {
        int base_r = face_bases[f];
        for (int sub = 0; sub < 2; ++sub) {
            int p0 = base_r + 0 + sub;
            int p1 = base_r + 2 + sub;
            int p2 = base_r + 4 + sub;

            int sc0 = base_r + 6;
            int sc1 = base_r + 7;
            int sc2 = 32 + base_r + 6;
            int sc3 = 32 + base_r + 7;

            // Stage 0: Q = L - R_old
            if (iter != 0) {
                sfpi::dst_reg[p0] = sfpi::round_bf16(sfpi::vFloat(sfpi::dst_reg[p0]) - sfpi::vFloat(sfpi::dst_reg[32 + p0]));
                sfpi::dst_reg[p1] = sfpi::round_bf16(sfpi::vFloat(sfpi::dst_reg[p1]) - sfpi::vFloat(sfpi::dst_reg[32 + p1]));
                sfpi::dst_reg[p2] = sfpi::round_bf16(sfpi::vFloat(sfpi::dst_reg[p2]) - sfpi::vFloat(sfpi::dst_reg[32 + p2]));
            }

            // Stage 1: Transpose raw Q vectors
            sfpi::dst_reg[32 + p0] = sfpi::swap_subvec16(sfpi::vFloat(sfpi::dst_reg[p0]));
            sfpi::dst_reg[32 + p1] = sfpi::swap_subvec16(sfpi::vFloat(sfpi::dst_reg[p1]));
            sfpi::dst_reg[32 + p2] = sfpi::swap_subvec16(sfpi::vFloat(sfpi::dst_reg[p2]));

            // Stage 2: 1-Cycle Native Integer XOR sign parity
            {
                sfpi::vInt s0 = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[p0])) ^ sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[32 + p0]));
                sfpi::vInt s1 = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[p1])) ^ sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[32 + p1]));
                sfpi::vInt s2 = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[p2])) ^ sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[32 + p2]));
                sfpi::dst_reg[sc0] = sfpi::as<sfpi::vFloat>((s0 ^ s1) ^ s2);
            }

            // Stage 3: Pairwise AMin* within edge pairs
            if (fp32_interm) {
                // Section 8 Roadmap 3: Preserve FP32 precision during intermediate stages
                sfpi::vFloat m0    = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[p0]));
                sfpi::vFloat m0_sw = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[32 + p0]));
                sfpi::dst_reg[sc1] = sfpi::sfpi_amin_star_branchless(m0, m0_sw);

                sfpi::vFloat m1    = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[p1]));
                sfpi::vFloat m1_sw = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[32 + p1]));
                sfpi::dst_reg[sc2] = sfpi::sfpi_amin_star_branchless(m1, m1_sw);

                sfpi::vFloat m2    = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[p2]));
                sfpi::vFloat m2_sw = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[32 + p2]));
                sfpi::dst_reg[sc3] = sfpi::sfpi_amin_star_branchless(m2, m2_sw);
            } else {
                sfpi::vFloat m0    = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[p0]));
                sfpi::vFloat m0_sw = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[32 + p0]));
                sfpi::dst_reg[sc1] = sfpi::round_bf16(sfpi::sfpi_amin_star_branchless(m0, m0_sw));

                sfpi::vFloat m1    = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[p1]));
                sfpi::vFloat m1_sw = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[32 + p1]));
                sfpi::dst_reg[sc2] = sfpi::round_bf16(sfpi::sfpi_amin_star_branchless(m1, m1_sw));

                sfpi::vFloat m2    = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[p2]));
                sfpi::vFloat m2_sw = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[32 + p2]));
                sfpi::dst_reg[sc3] = sfpi::round_bf16(sfpi::sfpi_amin_star_branchless(m2, m2_sw));
            }

            // Stage 4: Outgoing check messages with Intrinsic Damping (Section 8 Roadmap 1)
            // Edge pair 0
            {
                sfpi::vFloat m_oth0 = sfpi::sfpi_amin_star_branchless(sfpi::vFloat(sfpi::dst_reg[sc2]), sfpi::vFloat(sfpi::dst_reg[sc3]));
                sfpi::vFloat m0_sw  = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[32 + p0]));
                sfpi::vFloat r_mag0 = sfpi::sfpi_amin_star_branchless(m_oth0, m0_sw);
                if (damping_factor < 1.0f) r_mag0 = r_mag0 * damping_factor;

                sfpi::vInt s_edge0  = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc0])) ^ sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[p0]));
                sfpi::vFloat r_out0 = sfpi::round_bf16(sfpi::copysgn(r_mag0, s_edge0));
                sfpi::dst_reg[32 + p0] = sfpi::round_bf16(sfpi::vFloat(sfpi::dst_reg[p0]) + r_out0);
                sfpi::dst_reg[p0]      = r_out0;
            }

            // Edge pair 1
            {
                sfpi::vFloat m_oth1 = sfpi::sfpi_amin_star_branchless(sfpi::vFloat(sfpi::dst_reg[sc1]), sfpi::vFloat(sfpi::dst_reg[sc3]));
                sfpi::vFloat m1_sw  = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[32 + p1]));
                sfpi::vFloat r_mag1 = sfpi::sfpi_amin_star_branchless(m_oth1, m1_sw);
                if (damping_factor < 1.0f) r_mag1 = r_mag1 * damping_factor;

                sfpi::vInt s_edge1  = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc0])) ^ sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[p1]));
                sfpi::vFloat r_out1 = sfpi::round_bf16(sfpi::copysgn(r_mag1, s_edge1));
                sfpi::dst_reg[32 + p1] = sfpi::round_bf16(sfpi::vFloat(sfpi::dst_reg[p1]) + r_out1);
                sfpi::dst_reg[p1]      = r_out1;
            }

            // Edge pair 2
            {
                sfpi::vFloat m_oth2 = sfpi::sfpi_amin_star_branchless(sfpi::vFloat(sfpi::dst_reg[sc1]), sfpi::vFloat(sfpi::dst_reg[sc2]));
                sfpi::vFloat m2_sw  = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[32 + p2]));
                sfpi::vFloat r_mag2 = sfpi::sfpi_amin_star_branchless(m_oth2, m2_sw);
                if (damping_factor < 1.0f) r_mag2 = r_mag2 * damping_factor;

                sfpi::vInt s_edge2  = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc0])) ^ sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[p2]));
                sfpi::vFloat r_out2 = sfpi::round_bf16(sfpi::copysgn(r_mag2, s_edge2));
                sfpi::dst_reg[32 + p2] = sfpi::round_bf16(sfpi::vFloat(sfpi::dst_reg[p2]) + r_out2);
                sfpi::dst_reg[p2]      = r_out2;
            }
        }
        sfpi::dst_reg[base_r + 6] = 0.0f;
        sfpi::dst_reg[base_r + 7] = 0.0f;
        sfpi::dst_reg[32 + base_r + 6] = 0.0f;
        sfpi::dst_reg[32 + base_r + 7] = 0.0f;
    }
}

inline void compute_tile_deg3_sfpi_enhanced(
    uint32_t iter,
    float damping_factor = 0.9375f,
    bool fp32_interm = true
) {
    (void)fp32_interm;
    const int face_bases[4] = {0, 8, 16, 24};
    for (int f = 0; f < 4; ++f) {
        int base_r = face_bases[f];
        for (int sub = 0; sub < 2; ++sub) {
            int p0 = base_r + 0 + sub;
            int p1 = base_r + 2 + sub;
            int p2 = base_r + 4 + sub;

            int sc0 = base_r + 6;
            int sc1 = base_r + 7;
            int sc2 = 32 + base_r + 6;
            int sc3 = 32 + base_r + 7;

            if (iter != 0) {
                sfpi::dst_reg[p0] = sfpi::round_bf16(sfpi::vFloat(sfpi::dst_reg[p0]) - sfpi::vFloat(sfpi::dst_reg[32 + p0]));
                sfpi::dst_reg[p1] = sfpi::round_bf16(sfpi::vFloat(sfpi::dst_reg[p1]) - sfpi::vFloat(sfpi::dst_reg[32 + p1]));
                sfpi::dst_reg[p2] = sfpi::round_bf16(sfpi::vFloat(sfpi::dst_reg[p2]) - sfpi::vFloat(sfpi::dst_reg[32 + p2]));
            }

            sfpi::dst_reg[sc0] = sfpi::swap_subvec16(sfpi::vFloat(sfpi::dst_reg[p0]));
            sfpi::dst_reg[sc1] = sfpi::swap_subvec16(sfpi::vFloat(sfpi::dst_reg[p1]));
            sfpi::dst_reg[sc2] = sfpi::swap_subvec16(sfpi::vFloat(sfpi::dst_reg[p2]));

            {
                sfpi::vFloat q1    = sfpi::dst_reg[p1];
                sfpi::vFloat q1_sw = sfpi::dst_reg[sc1];

                sfpi::vFloat q_e2 = q1;
                v_if (sfpi::as<sfpi::vUInt>(sfpi::vConstTileId) >= 32) {
                    q_e2 = q1_sw;
                } v_endif;
                sfpi::dst_reg[sc1] = q_e2;

                sfpi::vFloat q_e0 = q1;
                v_if (sfpi::as<sfpi::vUInt>(sfpi::vConstTileId) < 32) {
                    q_e0 = q1_sw;
                } v_endif;
                sfpi::dst_reg[sc3] = q_e0;
            }

            // Block 0
            {
                sfpi::vFloat q0    = sfpi::dst_reg[p0];
                sfpi::vFloat q0_sw = sfpi::dst_reg[sc0];
                sfpi::vFloat q_e2  = sfpi::dst_reg[sc1];

                sfpi::vInt s0    = sfpi::as<sfpi::vInt>(q0);
                sfpi::vInt s0_sw = sfpi::as<sfpi::vInt>(q0_sw);
                sfpi::vInt s_e2  = sfpi::as<sfpi::vInt>(q_e2);
                sfpi::vInt s_tot = (s0 ^ s0_sw) ^ s_e2;

                sfpi::vFloat m0    = sfpi::abs(q0);
                sfpi::vFloat m0_sw = sfpi::abs(q0_sw);
                sfpi::vFloat m_e2  = sfpi::abs(q_e2);

                sfpi::vFloat r_mag0 = sfpi::sfpi_amin_star_branchless(m0_sw, m_e2);
                if (damping_factor < 1.0f) r_mag0 = r_mag0 * damping_factor;
                sfpi::vFloat r_out0 = sfpi::round_bf16(sfpi::copysgn(r_mag0, s_tot ^ s0));

                sfpi::vFloat r_mag_e2 = sfpi::sfpi_amin_star_branchless(m0, m0_sw);
                if (damping_factor < 1.0f) r_mag_e2 = r_mag_e2 * damping_factor;
                sfpi::vFloat r_out_e2 = sfpi::round_bf16(sfpi::copysgn(r_mag_e2, s_tot ^ s_e2));

                sfpi::dst_reg[32 + p0] = sfpi::round_bf16(sfpi::vFloat(sfpi::dst_reg[p0]) + r_out0);
                sfpi::dst_reg[p0]      = r_out0;
                sfpi::dst_reg[sc1]     = r_out_e2;
            }

            // Block 1
            {
                sfpi::vFloat q2    = sfpi::dst_reg[p2];
                sfpi::vFloat q2_sw = sfpi::dst_reg[sc2];
                sfpi::vFloat q_e0  = sfpi::dst_reg[sc3];

                sfpi::vInt s2    = sfpi::as<sfpi::vInt>(q2);
                sfpi::vInt s2_sw = sfpi::as<sfpi::vInt>(q2_sw);
                sfpi::vInt s_e0  = sfpi::as<sfpi::vInt>(q_e0);
                sfpi::vInt s_tot = (s2 ^ s2_sw) ^ s_e0;

                sfpi::vFloat m2    = sfpi::abs(q2);
                sfpi::vFloat m2_sw = sfpi::abs(q2_sw);
                sfpi::vFloat m_e0  = sfpi::abs(q_e0);

                sfpi::vFloat r_mag2 = sfpi::sfpi_amin_star_branchless(m2_sw, m_e0);
                if (damping_factor < 1.0f) r_mag2 = r_mag2 * damping_factor;
                sfpi::vFloat r_out2 = sfpi::round_bf16(sfpi::copysgn(r_mag2, s_tot ^ s2));

                sfpi::vFloat r_mag_e0 = sfpi::sfpi_amin_star_branchless(m2, m2_sw);
                if (damping_factor < 1.0f) r_mag_e0 = r_mag_e0 * damping_factor;
                sfpi::vFloat r_out_e0 = sfpi::round_bf16(sfpi::copysgn(r_mag_e0, s_tot ^ s_e0));

                sfpi::dst_reg[32 + p2] = sfpi::round_bf16(sfpi::vFloat(sfpi::dst_reg[p2]) + r_out2);
                sfpi::dst_reg[p2]      = r_out2;
                sfpi::dst_reg[sc3]     = r_out_e0;
            }

            // Merge back shared edge p1
            {
                sfpi::vFloat r_e2 = sfpi::dst_reg[sc1];
                sfpi::vFloat r_e0 = sfpi::dst_reg[sc3];
                sfpi::vFloat r1 = r_e0;
                v_if (sfpi::as<sfpi::vUInt>(sfpi::vConstTileId) < 32) {
                    r1 = r_e2;
                } v_endif;

                sfpi::dst_reg[32 + p1] = sfpi::round_bf16(sfpi::vFloat(sfpi::dst_reg[p1]) + r1);
                sfpi::dst_reg[p1]      = r1;
            }
        }
        sfpi::dst_reg[base_r + 6] = 0.0f;
        sfpi::dst_reg[base_r + 7] = 0.0f;
        sfpi::dst_reg[32 + base_r + 6] = 0.0f;
        sfpi::dst_reg[32 + base_r + 7] = 0.0f;
    }
}

inline void step_trisc_sfpi_enhanced(
    uint32_t t, uint32_t deg, uint32_t iter,
    float post_clip_max,
    float damping_factor,
    bool fp32_interm,
    const std::vector<uint16_t>& scratch_in,
    std::vector<std::vector<uint16_t>>& r_msg_fifo,
    std::vector<uint16_t>& scratch_out
) {
    for (int r = 0; r < 32; ++r) {
        for (uint32_t lane = 0; lane < 32; ++lane) {
            uint32_t off = get_scratch_byte_offset(r, lane);
            sfpi::dst_reg[r][lane] = bf16_to_fp32(*reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(scratch_in.data()) + off));
            if (iter == 0) {
                sfpi::dst_reg[32 + r][lane] = 0.0f;
            } else {
                sfpi::dst_reg[32 + r][lane] = bf16_to_fp32(*reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(r_msg_fifo[t].data()) + off));
            }
        }
    }

    if (deg == 6) {
        compute_tile_deg6_sfpi_enhanced(iter, damping_factor, fp32_interm);
    } else {
        compute_tile_deg3_sfpi_enhanced(iter, damping_factor, fp32_interm);
    }

    for (int r = 0; r < 32; ++r) {
        for (uint32_t lane = 0; lane < 32; ++lane) {
            uint32_t off = get_scratch_byte_offset(r, lane);
            float r_new = sfpi::dst_reg[r][lane];
            float l_new = sfpi::dst_reg[32 + r][lane];
            if (post_clip_max > 0.0f) {
                if (l_new > post_clip_max) l_new = post_clip_max;
                if (l_new < -post_clip_max) l_new = -post_clip_max;
            }
            *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(r_msg_fifo[t].data()) + off) = fp32_to_bf16(r_new);
            *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(scratch_out.data()) + off)   = fp32_to_bf16(l_new);
        }
    }
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

            // 50/50 Intra-Codeword Interleaving: Guarantees every codeword receives an identical
            // 50% Z0 and 50% Z1 mix across variable nodes, perfectly matching scalar_5w and compute_trisc.cpp hardware.
            if ((v & 1) != 0) {
                p[0] = fp32_to_bf16(l1);
                p[1] = fp32_to_bf16(l0);
                p[2] = fp32_to_bf16(l3);
                p[3] = fp32_to_bf16(l2);
            } else {
                p[0] = fp32_to_bf16(l0);
                p[1] = fp32_to_bf16(l1);
                p[2] = fp32_to_bf16(l2);
                p[3] = fp32_to_bf16(l3);
            }
        }
    }

    // 2. Punctured variable nodes v in [2048..2559] (Tiles 8 & 9: 4,096 bytes = 512 variables x 8 bytes)
    std::memset(vnode_l1_ptr + 8 * 2048, 0, 4096);
}

// ------------------------------------------------------------------------------
// Option B: 8-Codeword Unified Initialization (VNode128, 40 KB Unified Array)
// ------------------------------------------------------------------------------
inline void init_vnodes_8cw(
    uint8_t* vnode_l1_ptr,
    float mu_llr,
    float sigma_llr,
    Xoshiro128Plus& rng
) {
    bool bench = (sigma_llr <= 0.0f);
    uint16_t mu_bf = fp32_to_bf16(mu_llr);

    std::memset(vnode_l1_ptr, 0, 40960);

    // 1. Transmitted variable nodes v in [0..2047] (Tiles 0..7)
    for (uint32_t v = 0; v < 2048; ++v) {
        uint32_t off = get_tile_vnode_byte_offset(v) << 1;
        uint16_t* p = reinterpret_cast<uint16_t*>(vnode_l1_ptr + off);
        if (bench) {
            for (int c = 0; c < 8; ++c) p[c] = mu_bf;
        } else {
            float l0, l1, l2, l3, l4, l5, l6, l7;
            generate_awgn_pair(rng, mu_llr, sigma_llr, l0, l1);
            generate_awgn_pair(rng, mu_llr, sigma_llr, l2, l3);
            generate_awgn_pair(rng, mu_llr, sigma_llr, l4, l5);
            generate_awgn_pair(rng, mu_llr, sigma_llr, l6, l7);

            // 50/50 Intra-Codeword Interleaving
            if ((v & 1) != 0) {
                p[0] = fp32_to_bf16(l1);
                p[1] = fp32_to_bf16(l0);
                p[2] = fp32_to_bf16(l3);
                p[3] = fp32_to_bf16(l2);
                p[4] = fp32_to_bf16(l5);
                p[5] = fp32_to_bf16(l4);
                p[6] = fp32_to_bf16(l7);
                p[7] = fp32_to_bf16(l6);
            } else {
                p[0] = fp32_to_bf16(l0);
                p[1] = fp32_to_bf16(l1);
                p[2] = fp32_to_bf16(l2);
                p[3] = fp32_to_bf16(l3);
                p[4] = fp32_to_bf16(l4);
                p[5] = fp32_to_bf16(l5);
                p[6] = fp32_to_bf16(l6);
                p[7] = fp32_to_bf16(l7);
            }
        }
    }
}

struct BatchStats8CW {
    uint32_t bit_errors[8] = {0};
    uint32_t iters_taken[8] = {0};
    bool converged[8] = {false};
};

// ------------------------------------------------------------------------------
// Fused 64-Bit / 128-Bit Disjoint Gather & Scatter for Inter-Batch Ping-Pong
// In the Inter-Batch Ping-Pong schedule, Batch A (CW 0..3) and Batch B (CW 4..7)
// alternate DAE and TRISC compute execution every step.
// Tile s gathers strictly AFTER Tile s-1 has scattered into L1 SRAM.
// Because Batch A and Batch B are 100% disjoint in variable node memory,
// direct store produces ZERO RAW/WAW hazards and exact Gauss-Seidel convergence.
// ------------------------------------------------------------------------------

inline void gather_face_batch_a(
    uintptr_t dst_scratch_addr_a,
    const uint32_t* p_hoff_24,
    const char* vnode_bytes_ptr,
    uint32_t offset_shift = 1
) {
    auto* dst_even128_a = reinterpret_cast<v2ui*>(dst_scratch_addr_a);
    const uint32_t* p_h = p_hoff_24;
    for (uint32_t d = 0; d < 6; ++d) {
        auto* dst_odd128_a = dst_even128_a + 2;
        uint32_t off0 = p_h[0], off1 = p_h[1];
        uint32_t off0_e = (off0 & 0xFFFF) << offset_shift;
        uint32_t off0_o = (off0 >> 16) << offset_shift;
        uint32_t off1_e = (off1 & 0xFFFF) << offset_shift;
        uint32_t off1_o = (off1 >> 16) << offset_shift;

        uint64_t w0_e = *reinterpret_cast<const uint64_t*>(vnode_bytes_ptr + off0_e);
        uint64_t w0_o = *reinterpret_cast<const uint64_t*>(vnode_bytes_ptr + off0_o);
        uint64_t w1_e = *reinterpret_cast<const uint64_t*>(vnode_bytes_ptr + off1_e);
        uint64_t w1_o = *reinterpret_cast<const uint64_t*>(vnode_bytes_ptr + off1_o);

        dst_even128_a[0] = (v2ui){ w0_e, w1_e };
        dst_odd128_a[0]  = (v2ui){ w0_o, w1_o };

        uint32_t off2 = p_h[2], off3 = p_h[3];
        uint32_t off2_e = (off2 & 0xFFFF) << offset_shift;
        uint32_t off2_o = (off2 >> 16) << offset_shift;
        uint32_t off3_e = (off3 & 0xFFFF) << offset_shift;
        uint32_t off3_o = (off3 >> 16) << offset_shift;

        uint64_t w2_e = *reinterpret_cast<const uint64_t*>(vnode_bytes_ptr + off2_e);
        uint64_t w2_o = *reinterpret_cast<const uint64_t*>(vnode_bytes_ptr + off2_o);
        uint64_t w3_e = *reinterpret_cast<const uint64_t*>(vnode_bytes_ptr + off3_e);
        uint64_t w3_o = *reinterpret_cast<const uint64_t*>(vnode_bytes_ptr + off3_o);

        dst_even128_a[1] = (v2ui){ w2_e, w3_e };
        dst_odd128_a[1]  = (v2ui){ w2_o, w3_o };

        dst_even128_a += 4;
        p_h += 4;
    }
    v2ui zero = (v2ui){0, 0};
    for (int i = 0; i < 8; ++i) dst_even128_a[i] = zero;
}

inline void gather_face_batch_b(
    uintptr_t dst_scratch_addr_b,
    const uint32_t* p_hoff_24,
    const char* vnode_bytes_ptr,
    uint32_t offset_shift = 1
) {
    auto* dst_even128_b = reinterpret_cast<v2ui*>(dst_scratch_addr_b);
    const uint32_t* p_h = p_hoff_24;
    for (uint32_t d = 0; d < 6; ++d) {
        auto* dst_odd128_b = dst_even128_b + 2;
        uint32_t off0 = p_h[0], off1 = p_h[1];
        uint32_t off0_e = (off0 & 0xFFFF) << offset_shift;
        uint32_t off0_o = (off0 >> 16) << offset_shift;
        uint32_t off1_e = (off1 & 0xFFFF) << offset_shift;
        uint32_t off1_o = (off1 >> 16) << offset_shift;

        uint64_t w0_e = *(reinterpret_cast<const uint64_t*>(vnode_bytes_ptr + off0_e) + 1);
        uint64_t w0_o = *(reinterpret_cast<const uint64_t*>(vnode_bytes_ptr + off0_o) + 1);
        uint64_t w1_e = *(reinterpret_cast<const uint64_t*>(vnode_bytes_ptr + off1_e) + 1);
        uint64_t w1_o = *(reinterpret_cast<const uint64_t*>(vnode_bytes_ptr + off1_o) + 1);

        dst_even128_b[0] = (v2ui){ w0_e, w1_e };
        dst_odd128_b[0]  = (v2ui){ w0_o, w1_o };

        uint32_t off2 = p_h[2], off3 = p_h[3];
        uint32_t off2_e = (off2 & 0xFFFF) << offset_shift;
        uint32_t off2_o = (off2 >> 16) << offset_shift;
        uint32_t off3_e = (off3 & 0xFFFF) << offset_shift;
        uint32_t off3_o = (off3 >> 16) << offset_shift;

        uint64_t w2_e = *(reinterpret_cast<const uint64_t*>(vnode_bytes_ptr + off2_e) + 1);
        uint64_t w2_o = *(reinterpret_cast<const uint64_t*>(vnode_bytes_ptr + off2_o) + 1);
        uint64_t w3_e = *(reinterpret_cast<const uint64_t*>(vnode_bytes_ptr + off3_e) + 1);
        uint64_t w3_o = *(reinterpret_cast<const uint64_t*>(vnode_bytes_ptr + off3_o) + 1);

        dst_even128_b[1] = (v2ui){ w2_e, w3_e };
        dst_odd128_b[1]  = (v2ui){ w2_o, w3_o };

        dst_even128_b += 4;
        p_h += 4;
    }
    v2ui zero = (v2ui){0, 0};
    for (int i = 0; i < 8; ++i) dst_even128_b[i] = zero;
}

inline void scatter_face_batch_a(
    uintptr_t src_face_addr_a,
    const uint32_t* p_hoff_24,
    char* vnode_out_ptr,
    uint32_t offset_shift = 1,
    uint8_t active_mask_a = 0x0F
) {
    const auto* src_even128_a = reinterpret_cast<const v2ui*>(src_face_addr_a);
    const uint32_t* p_h = p_hoff_24;
    bool needs_mask = ((active_mask_a & 0x0F) != 0x0F);
    uint64_t mask_a = 0xFFFFFFFFFFFFFFFFULL;
    if (needs_mask) {
        mask_a = ((active_mask_a & 1) ? 0x000000000000FFFFULL : 0ULL) |
                 ((active_mask_a & 2) ? 0x00000000FFFF0000ULL : 0ULL) |
                 ((active_mask_a & 4) ? 0x0000FFFF00000000ULL : 0ULL) |
                 ((active_mask_a & 8) ? 0xFFFF000000000000ULL : 0ULL);
    }
    uint64_t inv_mask_a = ~mask_a;

    auto write_vnode_a = [&](char* dst, uint64_t val) {
        auto* p = reinterpret_cast<uint64_t*>(dst);
        if (!needs_mask) {
            *p = val;
        } else {
            *p = (val & mask_a) | (*p & inv_mask_a);
        }
    };

    for (uint32_t d = 0; d < 6; ++d) {
        const auto* src_odd128_a = src_even128_a + 2;
        uint32_t off0 = p_h[0], off1 = p_h[1];
        uint32_t off0_e = (off0 & 0xFFFF) << offset_shift;
        uint32_t off0_o = (off0 >> 16) << offset_shift;
        uint32_t off1_e = (off1 & 0xFFFF) << offset_shift;
        uint32_t off1_o = (off1 >> 16) << offset_shift;

        v2ui v_even0_a = src_even128_a[0];
        v2ui v_odd0_a  = src_odd128_a[0];

        write_vnode_a(vnode_out_ptr + off0_e, v_even0_a[0]);
        write_vnode_a(vnode_out_ptr + off0_o, v_odd0_a[0]);
        write_vnode_a(vnode_out_ptr + off1_e, v_even0_a[1]);
        write_vnode_a(vnode_out_ptr + off1_o, v_odd0_a[1]);

        uint32_t off2 = p_h[2], off3 = p_h[3];
        uint32_t off2_e = (off2 & 0xFFFF) << offset_shift;
        uint32_t off2_o = (off2 >> 16) << offset_shift;
        uint32_t off3_e = (off3 & 0xFFFF) << offset_shift;
        uint32_t off3_o = (off3 >> 16) << offset_shift;

        v2ui v_even1_a = src_even128_a[1];
        v2ui v_odd1_a  = src_odd128_a[1];

        write_vnode_a(vnode_out_ptr + off2_e, v_even1_a[0]);
        write_vnode_a(vnode_out_ptr + off2_o, v_odd1_a[0]);
        write_vnode_a(vnode_out_ptr + off3_e, v_even1_a[1]);
        write_vnode_a(vnode_out_ptr + off3_o, v_odd1_a[1]);

        src_even128_a += 4;
        p_h += 4;
    }
}

inline void scatter_face_batch_b(
    uintptr_t src_face_addr_b,
    const uint32_t* p_hoff_24,
    char* vnode_out_ptr,
    uint32_t offset_shift = 1,
    uint8_t active_mask_b = 0x0F
) {
    const auto* src_even128_b = reinterpret_cast<const v2ui*>(src_face_addr_b);
    const uint32_t* p_h = p_hoff_24;
    bool needs_mask = ((active_mask_b & 0x0F) != 0x0F);
    uint64_t mask_b = 0xFFFFFFFFFFFFFFFFULL;
    if (needs_mask) {
        mask_b = ((active_mask_b & 1) ? 0x000000000000FFFFULL : 0ULL) |
                 ((active_mask_b & 2) ? 0x00000000FFFF0000ULL : 0ULL) |
                 ((active_mask_b & 4) ? 0x0000FFFF00000000ULL : 0ULL) |
                 ((active_mask_b & 8) ? 0xFFFF000000000000ULL : 0ULL);
    }
    uint64_t inv_mask_b = ~mask_b;

    auto write_vnode_b = [&](char* dst, uint64_t val) {
        auto* p = reinterpret_cast<uint64_t*>(dst) + 1;
        if (!needs_mask) {
            *p = val;
        } else {
            *p = (val & mask_b) | (*p & inv_mask_b);
        }
    };

    for (uint32_t d = 0; d < 6; ++d) {
        const auto* src_odd128_b = src_even128_b + 2;
        uint32_t off0 = p_h[0], off1 = p_h[1];
        uint32_t off0_e = (off0 & 0xFFFF) << offset_shift;
        uint32_t off0_o = (off0 >> 16) << offset_shift;
        uint32_t off1_e = (off1 & 0xFFFF) << offset_shift;
        uint32_t off1_o = (off1 >> 16) << offset_shift;

        v2ui v_even0_b = src_even128_b[0];
        v2ui v_odd0_b  = src_odd128_b[0];

        write_vnode_b(vnode_out_ptr + off0_e, v_even0_b[0]);
        write_vnode_b(vnode_out_ptr + off0_o, v_odd0_b[0]);
        write_vnode_b(vnode_out_ptr + off1_e, v_even0_b[1]);
        write_vnode_b(vnode_out_ptr + off1_o, v_odd0_b[1]);

        uint32_t off2 = p_h[2], off3 = p_h[3];
        uint32_t off2_e = (off2 & 0xFFFF) << offset_shift;
        uint32_t off2_o = (off2 >> 16) << offset_shift;
        uint32_t off3_e = (off3 & 0xFFFF) << offset_shift;
        uint32_t off3_o = (off3 >> 16) << offset_shift;

        v2ui v_even1_b = src_even128_b[1];
        v2ui v_odd1_b  = src_odd128_b[1];

        write_vnode_b(vnode_out_ptr + off2_e, v_even1_b[0]);
        write_vnode_b(vnode_out_ptr + off2_o, v_odd1_b[0]);
        write_vnode_b(vnode_out_ptr + off3_e, v_even1_b[1]);
        write_vnode_b(vnode_out_ptr + off3_o, v_odd1_b[1]);

        src_even128_b += 4;
        p_h += 4;
    }
}

inline void gather_tile_batch_a(uintptr_t dst_scratch_a, const uint32_t* tile_hoff, const char* vnode_bytes_ptr) {
    gather_face_batch_a(dst_scratch_a + 0,    tile_hoff + 0,  vnode_bytes_ptr, 1);
    gather_face_batch_a(dst_scratch_a + 512,  tile_hoff + 48, vnode_bytes_ptr, 1);
    gather_face_batch_a(dst_scratch_a + 1024, tile_hoff + 24, vnode_bytes_ptr, 1);
    gather_face_batch_a(dst_scratch_a + 1536, tile_hoff + 72, vnode_bytes_ptr, 1);
}

inline void gather_tile_batch_b(uintptr_t dst_scratch_b, const uint32_t* tile_hoff, const char* vnode_bytes_ptr) {
    gather_face_batch_b(dst_scratch_b + 0,    tile_hoff + 0,  vnode_bytes_ptr, 1);
    gather_face_batch_b(dst_scratch_b + 512,  tile_hoff + 48, vnode_bytes_ptr, 1);
    gather_face_batch_b(dst_scratch_b + 1024, tile_hoff + 24, vnode_bytes_ptr, 1);
    gather_face_batch_b(dst_scratch_b + 1536, tile_hoff + 72, vnode_bytes_ptr, 1);
}

inline void scatter_tile_batch_a(uintptr_t src_scratch_a, const uint32_t* tile_hoff, char* vnode_out_ptr, uint8_t active_mask_a) {
    scatter_face_batch_a(src_scratch_a + 0,    tile_hoff + 0,  vnode_out_ptr, 1, active_mask_a);
    scatter_face_batch_a(src_scratch_a + 512,  tile_hoff + 48, vnode_out_ptr, 1, active_mask_a);
    scatter_face_batch_a(src_scratch_a + 1024, tile_hoff + 24, vnode_out_ptr, 1, active_mask_a);
    scatter_face_batch_a(src_scratch_a + 1536, tile_hoff + 72, vnode_out_ptr, 1, active_mask_a);
}

inline void scatter_tile_batch_b(uintptr_t src_scratch_b, const uint32_t* tile_hoff, char* vnode_out_ptr, uint8_t active_mask_b) {
    scatter_face_batch_b(src_scratch_b + 0,    tile_hoff + 0,  vnode_out_ptr, 1, active_mask_b);
    scatter_face_batch_b(src_scratch_b + 512,  tile_hoff + 48, vnode_out_ptr, 1, active_mask_b);
    scatter_face_batch_b(src_scratch_b + 1024, tile_hoff + 24, vnode_out_ptr, 1, active_mask_b);
    scatter_face_batch_b(src_scratch_b + 1536, tile_hoff + 72, vnode_out_ptr, 1, active_mask_b);
}

// Hardware-Accurate 8-Codeword Batch Decoder (Decoupled Access-Execute)
inline BatchStats8CW decode_tile_matrix_8cw_pipelined(
    const ParityCheckMatrix& pcm,
    const std::vector<DynamicTileDescriptor>& schedule,
    uint8_t* vnode_l1_unified,
    std::vector<std::vector<uint16_t>>& r_msg_fifo_a,
    std::vector<std::vector<uint16_t>>& r_msg_fifo_b,
    std::vector<uint16_t> scratch_in_a[2],
    std::vector<uint16_t> scratch_in_b[2],
    std::vector<uint16_t> scratch_out_a[2],
    std::vector<uint16_t> scratch_out_b[2],
    uint32_t max_iter,
    float damping_factor = 0.9375f,
    bool fp32_interm = true,
    float post_clip_max = 0.0f,
    bool use_sign_scan = false,
    bool lockstep = false
) {
    BatchStats8CW stats;
    uint8_t active_mask_a = 0x0F;
    uint8_t active_mask_b = 0x0F;

    uint32_t num_tiles = static_cast<uint32_t>(schedule.size());
    const uint32_t* p_hoff_base = pcm.h_offsets_pairs.data();
    const char* vnode_bytes_ptr = reinterpret_cast<const char*>(vnode_l1_unified);
    char* vnode_out_ptr = reinterpret_cast<char*>(vnode_l1_unified);

    auto check_codeword_parity = [&](int c) -> bool {
        if (use_sign_scan) {
            for (uint32_t v = 0; v < 2048; ++v) {
                uint32_t off = get_tile_vnode_byte_offset(v) << 1;
                const uint16_t* p = reinterpret_cast<const uint16_t*>(vnode_bytes_ptr + off);
                if (bf16_to_fp32(p[c]) < 0.0f) return false;
            }
            return true;
        } else {
            for (uint32_t m = 0; m < pcm.M; ++m) {
                int syn = 0;
                for (uint32_t off : pcm.check_to_var_offsets[m]) {
                    const uint16_t* p = reinterpret_cast<const uint16_t*>(vnode_bytes_ptr + (off << 1));
                    if (bf16_to_fp32(p[c]) < 0.0f) syn ^= 1;
                }
                if (syn != 0) return false;
            }
            return true;
        }
    };

    if (lockstep) {
        // Synchronous Stop-and-Wait Diagnostic Baseline
        for (uint32_t iter = 0; iter < max_iter; ++iter) {
            for (uint32_t t = 0; t < num_tiles; ++t) {
                const uint32_t* tile_hoff = p_hoff_base + t * 96;
                uintptr_t dst_scratch_a = reinterpret_cast<uintptr_t>(scratch_in_a[0].data());
                uintptr_t dst_scratch_b = reinterpret_cast<uintptr_t>(scratch_in_b[0].data());
                gather_face_128bit(dst_scratch_a + 0,    dst_scratch_b + 0,    tile_hoff + 0,  vnode_bytes_ptr, 1);
                gather_face_128bit(dst_scratch_a + 512,  dst_scratch_b + 512,  tile_hoff + 48, vnode_bytes_ptr, 1);
                gather_face_128bit(dst_scratch_a + 1024, dst_scratch_b + 1024, tile_hoff + 24, vnode_bytes_ptr, 1);
                gather_face_128bit(dst_scratch_a + 1536, dst_scratch_b + 1536, tile_hoff + 72, vnode_bytes_ptr, 1);

                uint32_t deg = schedule[t].check_degree;
                step_trisc_sfpi_enhanced(t, deg, iter, post_clip_max, damping_factor, fp32_interm,
                                         scratch_in_a[0], r_msg_fifo_a, scratch_out_a[0]);
                step_trisc_sfpi_enhanced(t, deg, iter, post_clip_max, damping_factor, fp32_interm,
                                         scratch_in_b[0], r_msg_fifo_b, scratch_out_b[0]);

                uintptr_t src_scratch_a = reinterpret_cast<uintptr_t>(scratch_out_a[0].data());
                uintptr_t src_scratch_b = reinterpret_cast<uintptr_t>(scratch_out_b[0].data());
                scatter_face_128bit(src_scratch_a + 0,    src_scratch_b + 0,    tile_hoff + 0,  vnode_out_ptr, 1, active_mask_a, active_mask_b);
                scatter_face_128bit(src_scratch_a + 512,  src_scratch_b + 512,  tile_hoff + 48, vnode_out_ptr, 1, active_mask_a, active_mask_b);
                scatter_face_128bit(src_scratch_a + 1024, src_scratch_b + 1024, tile_hoff + 24, vnode_out_ptr, 1, active_mask_a, active_mask_b);
                scatter_face_128bit(src_scratch_a + 1536, src_scratch_b + 1536, tile_hoff + 72, vnode_out_ptr, 1, active_mask_a, active_mask_b);
            }

            for (int c = 0; c < 4; ++c) {
                if ((active_mask_a & (1 << c)) && check_codeword_parity(c)) {
                    stats.converged[c] = true;
                    stats.iters_taken[c] = iter + 1;
                    active_mask_a &= ~(1 << c);
                }
                if ((active_mask_b & (1 << c)) && check_codeword_parity(4 + c)) {
                    stats.converged[4 + c] = true;
                    stats.iters_taken[4 + c] = iter + 1;
                    active_mask_b &= ~(1 << c);
                }
            }

            if (active_mask_a == 0 && active_mask_b == 0) break;
        }
    } else {
        // High-Throughput Inter-Batch Ping-Pong Pipelined DAE Engine
        // Zero RAW/WAW Hazards: Alternates Batch A and Batch B every step.
        // Scatter tile (s-1) occurs strictly BEFORE gather tile (s) within each batch.
        // TRISC compute duty cycle = 100%, DAE data movement is 100% hidden.
        uint32_t max_steps = max_iter * num_tiles;
        uint32_t total_steps = 2 * max_steps + 2;

        for (uint32_t k = 0; k < total_steps; ++k) {
            uint32_t b_dae = k % 2;
            uint32_t s_dae = k / 2;

            if (b_dae == 0) {
                // Even Step: DAE on Batch A, TRISC on Batch B
                // 1. DAE Scatter Batch A (tile s_dae - 1)
                if (s_dae >= 1 && (s_dae - 1) < max_steps) {
                    uint32_t s_scat = s_dae - 1;
                    uint32_t t_scat = s_scat % num_tiles;
                    uintptr_t src_a = reinterpret_cast<uintptr_t>(scratch_out_a[0].data());
                    scatter_tile_batch_a(src_a, p_hoff_base + t_scat * 96, vnode_out_ptr, active_mask_a);

                    if (t_scat == num_tiles - 1) {
                        uint32_t iter_scat = s_scat / num_tiles;
                        for (int c = 0; c < 4; ++c) {
                            if ((active_mask_a & (1 << c)) && check_codeword_parity(c)) {
                                stats.converged[c] = true;
                                stats.iters_taken[c] = iter_scat + 1;
                                active_mask_a &= ~(1 << c);
                            }
                        }
                    }
                }

                // 2. DAE Gather Batch A (tile s_dae)
                // Guaranteed Zero Hazard: Gathers AFTER tile s_dae - 1 has scattered!
                if (s_dae < max_steps && active_mask_a != 0) {
                    uint32_t t_gath = s_dae % num_tiles;
                    uintptr_t dst_a = reinterpret_cast<uintptr_t>(scratch_in_a[0].data());
                    gather_tile_batch_a(dst_a, p_hoff_base + t_gath * 96, vnode_bytes_ptr);
                }

                // 3. TRISC Compute Batch B (tile s_dae - 1)
                if (s_dae >= 1 && (s_dae - 1) < max_steps && active_mask_b != 0) {
                    uint32_t s_comp = s_dae - 1;
                    uint32_t t_comp = s_comp % num_tiles;
                    uint32_t iter_comp = s_comp / num_tiles;
                    uint32_t deg = schedule[t_comp].check_degree;
                    step_trisc_sfpi_enhanced(t_comp, deg, iter_comp, post_clip_max, damping_factor, fp32_interm,
                                             scratch_in_b[0], r_msg_fifo_b, scratch_out_b[0]);
                }
            } else {
                // Odd Step: DAE on Batch B, TRISC on Batch A
                // 1. DAE Scatter Batch B (tile s_dae - 1)
                if (s_dae >= 1 && (s_dae - 1) < max_steps) {
                    uint32_t s_scat = s_dae - 1;
                    uint32_t t_scat = s_scat % num_tiles;
                    uintptr_t src_b = reinterpret_cast<uintptr_t>(scratch_out_b[0].data());
                    scatter_tile_batch_b(src_b, p_hoff_base + t_scat * 96, vnode_out_ptr, active_mask_b);

                    if (t_scat == num_tiles - 1) {
                        uint32_t iter_scat = s_scat / num_tiles;
                        for (int c = 0; c < 4; ++c) {
                            if ((active_mask_b & (1 << c)) && check_codeword_parity(4 + c)) {
                                stats.converged[4 + c] = true;
                                stats.iters_taken[4 + c] = iter_scat + 1;
                                active_mask_b &= ~(1 << c);
                            }
                        }
                    }
                }

                // 2. DAE Gather Batch B (tile s_dae)
                // Guaranteed Zero Hazard: Gathers AFTER tile s_dae - 1 has scattered!
                if (s_dae < max_steps && active_mask_b != 0) {
                    uint32_t t_gath = s_dae % num_tiles;
                    uintptr_t dst_b = reinterpret_cast<uintptr_t>(scratch_in_b[0].data());
                    gather_tile_batch_b(dst_b, p_hoff_base + t_gath * 96, vnode_bytes_ptr);
                }

                // 3. TRISC Compute Batch A (tile s_dae)
                if (s_dae < max_steps && active_mask_a != 0) {
                    uint32_t s_comp = s_dae;
                    uint32_t t_comp = s_comp % num_tiles;
                    uint32_t iter_comp = s_comp / num_tiles;
                    uint32_t deg = schedule[t_comp].check_degree;
                    step_trisc_sfpi_enhanced(t_comp, deg, iter_comp, post_clip_max, damping_factor, fp32_interm,
                                             scratch_in_a[0], r_msg_fifo_a, scratch_out_a[0]);
                }
            }

            if (active_mask_a == 0 && active_mask_b == 0) break;
        }
    }

    // Residual Bit Error Check on Information Bits (v < K) for Unconverged Codewords
    for (int c = 0; c < 8; ++c) {
        if (!stats.converged[c]) {
            stats.iters_taken[c] = max_iter;
            for (uint32_t off : pcm.info_var_offsets) {
                const uint16_t* p = reinterpret_cast<const uint16_t*>(vnode_bytes_ptr + (off << 1));
                if (bf16_to_fp32(p[c]) < 0.0f) stats.bit_errors[c]++;
            }
        }
    }

    return stats;
}

struct BatchStats {
    uint32_t bit_errors[4] = {0, 0, 0, 0};
    uint32_t iters_taken[4] = {0, 0, 0, 0};
    bool converged[4] = {false, false, false, false};
};

// Hardware-Accurate 4-Codeword 40-Tile Batch Decoder with Section 8 Active-Mask Scatter Freezing
inline BatchStats decode_tile_matrix_batch_enhanced(
    const ParityCheckMatrix& pcm,
    const std::vector<DynamicTileDescriptor>& schedule,
    uint8_t* vnode_l1,
    std::vector<std::vector<uint16_t>>& r_msg_fifo,
    std::vector<uint16_t>& scratch_in,
    std::vector<uint16_t>& scratch_out,
    uint32_t max_iter,
    float damping_factor = 0.9375f,
    bool fp32_interm = true,
    float post_clip_max = 0.0f
) {
    BatchStats stats;
    uint8_t active_mask = 0x0F;
    const uint32_t* p_hoff_base = pcm.h_offsets_pairs.data();

    for (uint32_t iter = 0; iter < max_iter; ++iter) {
        for (size_t t = 0; t < schedule.size(); ++t) {
            uint32_t deg = schedule[t].check_degree;
            const uint32_t* tile_hoff_w0 = p_hoff_base + t * 96;
            const uint32_t* tile_hoff_w1 = tile_hoff_w0 + 48;

            // 1. Dual Gather Stage
            for (int p = 0; p < 48; ++p) {
                uint32_t off0 = tile_hoff_w0[p];
                uint32_t off0_even = off0 & 0xFFFF;
                uint32_t off0_odd  = off0 >> 16;
                uint32_t s0_even = SCRATCH_FACE_LUT.w0_even[p];
                uint32_t s0_odd  = SCRATCH_FACE_LUT.w0_odd[p];

                *reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(scratch_in.data()) + s0_even) =
                    *reinterpret_cast<const uint64_t*>(vnode_l1 + off0_even);
                *reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(scratch_in.data()) + s0_odd) =
                    *reinterpret_cast<const uint64_t*>(vnode_l1 + off0_odd);

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

            // 2. Compute Stage with Damping and FP32 Intermediate Precision
            step_trisc_sfpi_enhanced(t, deg, iter, post_clip_max, damping_factor, fp32_interm, scratch_in, r_msg_fifo, scratch_out);

            // 3. Scatter Stage with Section 8 Active-Mask Codeword Freezing
            for (int p = 0; p < 48; ++p) {
                // Worker 0
                uint32_t off0 = tile_hoff_w0[p];
                uint32_t off0_even = off0 & 0xFFFF;
                uint32_t off0_odd  = off0 >> 16;
                uint32_t s0_even = SCRATCH_FACE_LUT.w0_even[p];
                uint32_t s0_odd  = SCRATCH_FACE_LUT.w0_odd[p];

                uint16_t* dst_e0 = reinterpret_cast<uint16_t*>(vnode_l1 + off0_even);
                const uint16_t* src_e0 = reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(scratch_out.data()) + s0_even);
                uint16_t* dst_o0 = reinterpret_cast<uint16_t*>(vnode_l1 + off0_odd);
                const uint16_t* src_o0 = reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(scratch_out.data()) + s0_odd);

                for (int c = 0; c < 4; ++c) {
                    if (active_mask & (1 << c)) {
                        dst_e0[c] = src_e0[c];
                        dst_o0[c] = src_o0[c];
                    }
                }

                // Worker 1
                uint32_t off1 = tile_hoff_w1[p];
                uint32_t off1_even = off1 & 0xFFFF;
                uint32_t off1_odd  = off1 >> 16;
                uint32_t s1_even = SCRATCH_FACE_LUT.w1_even[p];
                uint32_t s1_odd  = SCRATCH_FACE_LUT.w1_odd[p];

                uint16_t* dst_e1 = reinterpret_cast<uint16_t*>(vnode_l1 + off1_even);
                const uint16_t* src_e1 = reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(scratch_out.data()) + s1_even);
                uint16_t* dst_o1 = reinterpret_cast<uint16_t*>(vnode_l1 + off1_odd);
                const uint16_t* src_o1 = reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(scratch_out.data()) + s1_odd);

                for (int c = 0; c < 4; ++c) {
                    if (active_mask & (1 << c)) {
                        dst_e1[c] = src_e1[c];
                        dst_o1[c] = src_o1[c];
                    }
                }
            }
        }

        // Parity Check Evaluation for Active Codewords
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
                active_mask &= ~(1 << c); // Freeze this codeword!
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
// Simulation Result Data Structure
// ------------------------------------------------------------------------------
struct SimSummary {
    std::string name;
    double eb_n0_db;
    uint64_t total_blocks;
    uint64_t frame_errors;
    uint64_t bit_errors;
    double fer;
    double ber;
    double avg_iters;
    double msps;
    double elapsed_s;
};

// ------------------------------------------------------------------------------
// Runner 1: Untouched scalar_5w Golden Model Reference across 16 Cores
// ------------------------------------------------------------------------------
SimSummary run_scalar_5w_monte_carlo(
    const ParityCheckMatrix& pcm,
    double eb_n0_db,
    uint64_t max_blocks,
    uint64_t min_frame_errors,
    uint32_t max_iter,
    int num_threads
) {
    uint32_t max_check_deg = 6;
    uint32_t stride = max_check_deg + 1;
    std::vector<uint16_t> flattened_indices(pcm.M * stride, 0xFFFF);
    for (uint32_t m = 0; m < pcm.M; ++m) {
        uint32_t deg = static_cast<uint32_t>(pcm.check_to_var[m].size());
        flattened_indices[m * stride] = static_cast<uint16_t>(deg);
        for (size_t d = 0; d < deg && d < max_check_deg; ++d) {
            flattened_indices[m * stride + 1 + d] = static_cast<uint16_t>(pcm.check_to_var[m][d]);
        }
    }

    float code_rate = static_cast<float>(pcm.K) / (pcm.N - pcm.P);
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
                std::cout << "\r[scalar_5w | " << std::fixed << std::setprecision(2) << eb_n0_db << " dB] "
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
                    std::cout << "[scalar_5w | " << std::fixed << std::setprecision(2) << eb_n0_db << " dB] "
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

    std::atomic<uint32_t> global_chunk_counter{0};
    uint32_t chunk_codewords = 100; // 50 pairs per chunk

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        std::vector<uint32_t> channel_llrs_pair(pcm.N, 0);
        std::vector<uint32_t> r_msg_base(pcm.M * Scalar5w::MAX_DEG_STRIDE, 0);
        uint32_t stats[4] = {0};

        while (!stop_flag.load(std::memory_order_relaxed)) {
            uint32_t chunk_idx = global_chunk_counter.fetch_add(1, std::memory_order_relaxed);
            uint64_t current_total = static_cast<uint64_t>(chunk_idx) * chunk_codewords;
            if (current_total >= max_blocks) {
                stop_flag.store(true, std::memory_order_relaxed);
                break;
            }

            uint32_t chunk_be = 0;
            uint32_t chunk_fe = 0;
            uint64_t chunk_iters = 0;
            uint32_t pairs_in_chunk = chunk_codewords / 2;

            for (uint32_t p = 0; p < pairs_in_chunk; ++p) {
                if (stop_flag.load(std::memory_order_relaxed)) break;

                uint32_t seed_idx = chunk_idx * pairs_in_chunk + p;
                uint32_t s0 = 0x12345678U + seed_idx * 1013904223U + tid * 31337U;
                uint32_t s1 = 0x87654321U + seed_idx * 65537U + tid * 17U;
                uint32_t s2 = 0xabcdef01U + seed_idx * 3571U + tid * 1009U;
                uint32_t s3 = 0x10fedcbaU + seed_idx * 199U + tid * 79U;

                Scalar5w::run_ldpc_worker(
                    0, 2, pcm.N, pcm.M, max_check_deg, pcm.P, max_iter,
                    mu_llr, sigma_llr, 0.0f, 0.0f, 0.0f,
                    s0, s1, s2, s3,
                    flattened_indices.data(),
                    channel_llrs_pair.data(),
                    r_msg_base.data(),
                    stats
                );

                chunk_be += stats[0];
                chunk_fe += stats[1];
                chunk_iters += static_cast<uint64_t>(stats[2]) * 2;
            }

            uint64_t new_fe = total_frame_errors.fetch_add(chunk_fe, std::memory_order_relaxed) + chunk_fe;
            total_bit_errors.fetch_add(chunk_be, std::memory_order_relaxed);
            uint64_t new_cws = total_codewords.fetch_add(chunk_codewords, std::memory_order_relaxed) + chunk_codewords;
            total_cw_iterations.fetch_add(chunk_iters, std::memory_order_relaxed);

            if (new_fe >= min_frame_errors || new_cws >= max_blocks) {
                stop_flag.store(true, std::memory_order_relaxed);
                break;
            }
        }
    }

    stop_flag.store(true, std::memory_order_relaxed);
    if (monitor.joinable()) monitor.join();

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration<double>(t_end - t_start).count();

    SimSummary res;
    res.name = "scalar_5w (Golden Model)";
    res.eb_n0_db = eb_n0_db;
    res.total_blocks = total_codewords.load();
    res.frame_errors = total_frame_errors.load();
    res.bit_errors = total_bit_errors.load();
    res.elapsed_s = elapsed_s;
    res.fer = (res.total_blocks > 0) ? static_cast<double>(res.frame_errors) / res.total_blocks : 0.0;
    res.ber = (res.total_blocks > 0) ? static_cast<double>(res.bit_errors) / (res.total_blocks * pcm.K) : 0.0;
    res.avg_iters = (res.total_blocks > 0) ? static_cast<double>(total_cw_iterations.load()) / res.total_blocks : 0.0;
    res.msps = (res.total_blocks > 0 && elapsed_s > 0) ? (res.total_blocks * (pcm.N - pcm.P)) / (elapsed_s * 1e6) : 0.0;
    return res;
}

// ------------------------------------------------------------------------------
// Runner 2: Enhanced tile_matrix_4cw Model across 16 Cores
// ------------------------------------------------------------------------------
SimSummary run_tile_matrix_monte_carlo(
    const ParityCheckMatrix& pcm,
    const std::vector<DynamicTileDescriptor>& schedule,
    double eb_n0_db,
    uint64_t max_blocks,
    uint64_t min_frame_errors,
    uint32_t max_iter,
    int num_threads,
    float damping_factor,
    bool fp32_interm,
    float post_clip,
    const std::string& model_name = "tile_matrix_4cw (Enhanced 5CW)"
) {
    float code_rate = static_cast<float>(pcm.K) / (pcm.N - pcm.P);
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
                std::cout << "\r[" << model_name << " | " << std::fixed << std::setprecision(2) << eb_n0_db << " dB] "
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
                    std::cout << "[" << model_name << " | " << std::fixed << std::setprecision(2) << eb_n0_db << " dB] "
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

    #pragma omp parallel num_threads(num_threads)
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

        uint32_t chunk_batches = 25; // 25 batches * 4 = 100 codewords per chunk

        while (!stop_flag.load(std::memory_order_relaxed)) {
            uint32_t local_fe = 0;
            uint32_t local_be = 0;
            uint64_t local_cw_iters = 0;

            uint32_t actual_batches = 0;
            for (uint32_t b = 0; b < chunk_batches; ++b) {
                if (stop_flag.load(std::memory_order_relaxed)) break;
                for (auto& fifo : r_msg_fifo) {
                    std::fill(fifo.begin(), fifo.end(), 0);
                }
                global_batch_counter.fetch_add(1, std::memory_order_relaxed);
                init_vnodes_host(vnode_l1.data(), mu_llr, sigma_llr, rng);
                actual_batches++;

                BatchStats stats = decode_tile_matrix_batch_enhanced(
                    pcm, schedule, vnode_l1.data(), r_msg_fifo, scratch_in, scratch_out,
                    max_iter, damping_factor, fp32_interm, post_clip
                );

                for (int c = 0; c < 4; ++c) {
                    local_cw_iters += stats.iters_taken[c];
                    if (stats.bit_errors[c] > 0 || !stats.converged[c]) {
                        local_fe++;
                        local_be += stats.bit_errors[c];
                    }
                }
            }

            uint64_t current_cws = total_codewords.fetch_add(actual_batches * 4, std::memory_order_relaxed) + actual_batches * 4;
            uint64_t new_fe = total_frame_errors.fetch_add(local_fe, std::memory_order_relaxed) + local_fe;
            total_bit_errors.fetch_add(local_be, std::memory_order_relaxed);
            total_cw_iterations.fetch_add(local_cw_iters, std::memory_order_relaxed);

            if (new_fe >= min_frame_errors || current_cws >= max_blocks) {
                stop_flag.store(true, std::memory_order_relaxed);
                break;
            }
        }
    }

    stop_flag.store(true, std::memory_order_relaxed);
    if (monitor.joinable()) monitor.join();

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration<double>(t_end - t_start).count();

    SimSummary res;
    res.name = model_name;
    res.eb_n0_db = eb_n0_db;
    res.total_blocks = total_codewords.load();
    res.frame_errors = total_frame_errors.load();
    res.bit_errors = total_bit_errors.load();
    res.elapsed_s = elapsed_s;
    res.fer = (res.total_blocks > 0) ? static_cast<double>(res.frame_errors) / res.total_blocks : 0.0;
    res.ber = (res.total_blocks > 0) ? static_cast<double>(res.bit_errors) / (res.total_blocks * pcm.K) : 0.0;
    res.avg_iters = (res.total_blocks > 0) ? static_cast<double>(total_cw_iterations.load()) / res.total_blocks : 0.0;
    res.msps = (res.total_blocks > 0 && elapsed_s > 0) ? (res.total_blocks * (pcm.N - pcm.P)) / (elapsed_s * 1e6) : 0.0;
    return res;
}

// ------------------------------------------------------------------------------
// Runner 3: Pipelined 8-Codeword (3-Stage Macro Ring) across Host Cores
// ------------------------------------------------------------------------------
SimSummary run_tile_matrix_8cw_pipelined_monte_carlo(
    const ParityCheckMatrix& pcm,
    const std::vector<DynamicTileDescriptor>& schedule,
    double eb_n0_db,
    uint64_t max_blocks,
    uint64_t min_frame_errors,
    uint32_t max_iter,
    int num_threads,
    float damping_factor,
    bool fp32_interm,
    float post_clip,
    bool use_sign_scan = false,
    bool lockstep = false,
    const std::string& model_name = "tile_matrix_8cw (Pipelined 3-Stage)"
) {
    float code_rate = static_cast<float>(pcm.K) / (pcm.N - pcm.P);
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
                std::cout << "\r[" << model_name << " | " << std::fixed << std::setprecision(2) << eb_n0_db << " dB] "
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
                    std::cout << "[" << model_name << " | " << std::fixed << std::setprecision(2) << eb_n0_db << " dB] "
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

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        Xoshiro128Plus rng;
        uint32_t seed = 1234567U + static_cast<uint32_t>(eb_n0_db * 100.0) * 10007U;
        rng.init(seed);
        for (int i = 0; i < tid; ++i) { rng.jump(); rng.jump(); }

        std::vector<uint8_t> vnode_l1(40960, 0);
        std::vector<std::vector<uint16_t>> r_msg_fifo_a(schedule.size(), std::vector<uint16_t>(1024, 0));
        std::vector<std::vector<uint16_t>> r_msg_fifo_b(schedule.size(), std::vector<uint16_t>(1024, 0));
        std::vector<uint16_t> scratch_in_a[2]  = { std::vector<uint16_t>(1024, 0), std::vector<uint16_t>(1024, 0) };
        std::vector<uint16_t> scratch_in_b[2]  = { std::vector<uint16_t>(1024, 0), std::vector<uint16_t>(1024, 0) };
        std::vector<uint16_t> scratch_out_a[2] = { std::vector<uint16_t>(1024, 0), std::vector<uint16_t>(1024, 0) };
        std::vector<uint16_t> scratch_out_b[2] = { std::vector<uint16_t>(1024, 0), std::vector<uint16_t>(1024, 0) };

        uint32_t chunk_batches = 12; // 12 batches * 8 = 96 codewords per chunk

        while (!stop_flag.load(std::memory_order_relaxed)) {
            uint32_t local_fe = 0;
            uint32_t local_be = 0;
            uint64_t local_cw_iters = 0;

            uint32_t actual_batches = 0;
            for (uint32_t b = 0; b < chunk_batches; ++b) {
                if (stop_flag.load(std::memory_order_relaxed)) break;
                for (auto& fifo : r_msg_fifo_a) std::fill(fifo.begin(), fifo.end(), 0);
                for (auto& fifo : r_msg_fifo_b) std::fill(fifo.begin(), fifo.end(), 0);

                global_batch_counter.fetch_add(1, std::memory_order_relaxed);
                init_vnodes_8cw(vnode_l1.data(), mu_llr, sigma_llr, rng);
                actual_batches++;

                BatchStats8CW stats = decode_tile_matrix_8cw_pipelined(
                    pcm, schedule, vnode_l1.data(),
                    r_msg_fifo_a, r_msg_fifo_b,
                    scratch_in_a, scratch_in_b,
                    scratch_out_a, scratch_out_b,
                    max_iter, damping_factor, fp32_interm, post_clip, use_sign_scan, lockstep
                );

                for (int c = 0; c < 8; ++c) {
                    local_cw_iters += stats.iters_taken[c];
                    if (stats.bit_errors[c] > 0 || !stats.converged[c]) {
                        local_fe++;
                        local_be += stats.bit_errors[c];
                    }
                }
            }

            uint64_t current_cws = total_codewords.fetch_add(actual_batches * 8, std::memory_order_relaxed) + actual_batches * 8;
            uint64_t new_fe = total_frame_errors.fetch_add(local_fe, std::memory_order_relaxed) + local_fe;
            total_bit_errors.fetch_add(local_be, std::memory_order_relaxed);
            total_cw_iterations.fetch_add(local_cw_iters, std::memory_order_relaxed);

            if (new_fe >= min_frame_errors || current_cws >= max_blocks) {
                stop_flag.store(true, std::memory_order_relaxed);
                break;
            }
        }
    }

    stop_flag.store(true, std::memory_order_relaxed);
    if (monitor.joinable()) monitor.join();

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration<double>(t_end - t_start).count();

    SimSummary res;
    res.name = lockstep ? "tile_matrix_8cw (Lockstep Stop&Wait)" : model_name;
    res.eb_n0_db = eb_n0_db;
    res.total_blocks = total_codewords.load();
    res.frame_errors = total_frame_errors.load();
    res.bit_errors = total_bit_errors.load();
    res.elapsed_s = elapsed_s;
    res.fer = (res.total_blocks > 0) ? static_cast<double>(res.frame_errors) / res.total_blocks : 0.0;
    res.ber = (res.total_blocks > 0) ? static_cast<double>(res.bit_errors) / (res.total_blocks * pcm.K) : 0.0;
    res.avg_iters = (res.total_blocks > 0) ? static_cast<double>(total_cw_iterations.load()) / res.total_blocks : 0.0;
    res.msps = (res.total_blocks > 0 && elapsed_s > 0) ? (res.total_blocks * (pcm.N - pcm.P)) / (elapsed_s * 1e6) : 0.0;
    return res;
}

// ------------------------------------------------------------------------------
// Main Simulation Entrypoint
// ------------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string matrix_path = "matrices/AR4JA_r45_4c_128c_r12.chinn.out";
    if (!fs::exists(matrix_path) && fs::exists("../matrices/AR4JA_r45_4c_128c_r12.chinn.out")) {
        matrix_path = "../matrices/AR4JA_r45_4c_128c_r12.chinn.out";
    }

    ParityCheckMatrix pcm;
    if (!parse_chinn_out(matrix_path, pcm)) return 1;

    // Simulation Defaults
    double eb_n0_db = 2.0;
    uint64_t max_blocks = 100000ULL;
    uint64_t min_frame_errors = 25ULL;
    uint32_t max_iter = 200U;
    int num_threads = 16;
    float damping_factor = 0.9375f; // Section 8 Roadmap 1: Damping factor 15/16
    bool contiguous_schedule = true; // Section 8 Roadmap 2: Contiguous 32 Deg6 then 8 Deg3
    bool fp32_interm = true;         // Section 8 Roadmap 3: FP32 intermediate accumulation
    float post_clip = 0.0f;
    bool pipelined_8cw = false;
    bool use_sign_scan = false;
    bool lockstep = false;
    std::string mode = "head-to-head"; // "head-to-head", "scalar-only", "tile-only", "pipelined-8cw", "compare-all"
    std::string csv_path = "Results/results_h2h_scalar_vs_tile_enhanced.csv";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--head-to-head" || arg == "--h2h" || arg == "--compare") {
            mode = "head-to-head";
        } else if (arg == "--scalar-only" || arg == "--scalar") {
            mode = "scalar-only";
        } else if (arg == "--tile-only" || arg == "--tile") {
            mode = "tile-only";
        } else if (arg == "--pipelined-8cw" || arg == "--8cw" || arg == "--pipelined") {
            pipelined_8cw = true;
            if (mode == "head-to-head") mode = "compare-all";
            else mode = "pipelined-8cw";
        } else if (arg == "--lockstep" || arg == "--sync") {
            lockstep = true;
        } else if (arg == "--compare-all" || arg == "--all") {
            mode = "compare-all";
            pipelined_8cw = true;
        } else if (arg == "--sign-scan") {
            use_sign_scan = true;
        } else if (arg == "--ebn0" && i + 1 < argc) {
            eb_n0_db = std::stod(argv[++i]);
        } else if (arg == "--blocks" && i + 1 < argc) {
            max_blocks = std::stoull(argv[++i]);
        } else if (arg == "--min-fe" && i + 1 < argc) {
            min_frame_errors = std::stoull(argv[++i]);
        } else if (arg == "--max-iter" && i + 1 < argc) {
            max_iter = std::stoul(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            num_threads = std::stoi(argv[++i]);
        } else if (arg == "--damping" && i + 1 < argc) {
            damping_factor = std::stof(argv[++i]);
        } else if (arg == "--schedule" && i + 1 < argc) {
            std::string sched_arg = argv[++i];
            contiguous_schedule = (sched_arg == "contiguous" || sched_arg == "contig");
        } else if (arg == "--fp32-interm") {
            fp32_interm = true;
        } else if (arg == "--bf16-interm") {
            fp32_interm = false;
        } else if (arg == "--csv" && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (arg == "--monte-carlo" && i + 1 < argc) {
            // Backwards compatibility with test_tile_matrix_40tile_lockstep_stepper_sfpi CLI
            eb_n0_db = std::stod(argv[++i]);
            if (i + 1 < argc && argv[i + 1][0] != '-') max_blocks = std::stoull(argv[++i]);
            if (i + 1 < argc && argv[i + 1][0] != '-') min_frame_errors = std::stoull(argv[++i]);
            if (i + 1 < argc && argv[i + 1][0] != '-') max_iter = std::stoul(argv[++i]);
            if (i + 1 < argc && argv[i + 1][0] != '-') num_threads = std::stoi(argv[++i]);
            if (i + 1 < argc && argv[i + 1][0] != '-') post_clip = std::stof(argv[++i]);
            if (i + 1 < argc && argv[i + 1][0] != '-') csv_path = argv[++i];
        }
    }

    if (num_threads <= 0) num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);

    auto schedule = build_enhanced_tile_schedule(pcm, contiguous_schedule);

    std::cout << "\n=================================================================" << std::endl;
    std::cout << " QUIETBOX 2: HEAD-TO-HEAD DECODER PERFORMANCE BENCHMARK" << std::endl;
    std::cout << " Matrix:            CCSDS AR4JA Rate-1/2 (N=2560, K=1024)" << std::endl;
    std::cout << " Mode:              " << mode << (pipelined_8cw ? " (8-Codeword Pipelined Ring Enabled)" : "") << std::endl;
    std::cout << " Eb/N0:             " << std::fixed << std::setprecision(2) << eb_n0_db << " dB" << std::endl;
    std::cout << " Max Blocks:        " << max_blocks << std::endl;
    std::cout << " Min Frame Errors:  " << min_frame_errors << std::endl;
    std::cout << " Max Iterations:    " << max_iter << std::endl;
    std::cout << " Host Threads:      " << num_threads << " (AMD Ryzen 7 9700X OpenMP)" << std::endl;
    std::cout << " Section 8 Enhancements (Tile Matrix):" << std::endl;
    std::cout << "   - Damping Factor: " << damping_factor << " (Intrinsic Belief Damping)" << std::endl;
    std::cout << "   - Tile Schedule:  " << (contiguous_schedule ? "Contiguous (32 Deg-6 then 8 Deg-3)" : "Interleaved (4:1 Deg-6/Deg-3)") << std::endl;
    std::cout << "   - Intermediate:   " << (fp32_interm ? "FP32 (Zero intermediate bf16 truncation)" : "bfloat16 RNE (Standard)") << std::endl;
    std::cout << "   - Active Mask:    Enabled (In-Place L1 SRAM Freezing on Convergence)" << std::endl;
    std::cout << "   - Early Parity:   " << (use_sign_scan ? "Hardware 2048-symbol sign scan" : "Full syndrome check (H*c == 0)") << std::endl;
    std::cout << "=================================================================\n" << std::endl;

    std::vector<SimSummary> results;

    if (mode == "compare-all" || mode == "head-to-head" || mode == "scalar-only") {
        std::cout << ">>> Running Model 1: scalar_5w Golden Model Reference (Untouched C++ Engine) <<<" << std::endl;
        SimSummary s_res = run_scalar_5w_monte_carlo(pcm, eb_n0_db, max_blocks, min_frame_errors, max_iter, num_threads);
        results.push_back(s_res);
        std::cout << "\n[DONE] scalar_5w Completed: Blocks=" << s_res.total_blocks << ", FE=" << s_res.frame_errors
                  << ", FER=" << std::scientific << std::setprecision(3) << s_res.fer
                  << ", BER=" << std::scientific << std::setprecision(3) << s_res.ber
                  << ", AvgIter=" << std::fixed << std::setprecision(2) << s_res.avg_iters
                  << ", Speed=" << std::fixed << std::setprecision(2) << s_res.msps << " Msps ("
                  << std::fixed << std::setprecision(1) << s_res.elapsed_s << "s)\n" << std::endl;
    }

    if (mode == "compare-all" || mode == "head-to-head" || mode == "tile-only") {
        std::cout << ">>> Running Model 2: tile_matrix_4cw Enhanced (5CW Performance Architecture) <<<" << std::endl;
        SimSummary t_res = run_tile_matrix_monte_carlo(
            pcm, schedule, eb_n0_db, max_blocks, min_frame_errors, max_iter,
            num_threads, damping_factor, fp32_interm, post_clip, "tile_matrix_4cw (Enhanced 5CW)"
        );
        results.push_back(t_res);
        std::cout << "\n[DONE] tile_matrix_4cw Completed: Blocks=" << t_res.total_blocks << ", FE=" << t_res.frame_errors
                  << ", FER=" << std::scientific << std::setprecision(3) << t_res.fer
                  << ", BER=" << std::scientific << std::setprecision(3) << t_res.ber
                  << ", AvgIter=" << std::fixed << std::setprecision(2) << t_res.avg_iters
                  << ", Speed=" << std::fixed << std::setprecision(2) << t_res.msps << " Msps ("
                  << std::fixed << std::setprecision(1) << t_res.elapsed_s << "s)\n" << std::endl;
    }

    if (mode == "compare-all" || mode == "pipelined-8cw") {
        std::string m3_name = lockstep ? "tile_matrix_8cw (Lockstep Stop&Wait)" : "tile_matrix_8cw (Pipelined 3-Stage)";
        std::cout << ">>> Running Model 3: " << m3_name << " <<<" << std::endl;
        SimSummary p_res = run_tile_matrix_8cw_pipelined_monte_carlo(
            pcm, schedule, eb_n0_db, max_blocks, min_frame_errors, max_iter,
            num_threads, damping_factor, fp32_interm, post_clip, use_sign_scan, lockstep, m3_name
        );
        results.push_back(p_res);
        std::cout << "\n[DONE] tile_matrix_8cw Completed: Blocks=" << p_res.total_blocks << ", FE=" << p_res.frame_errors
                  << ", FER=" << std::scientific << std::setprecision(3) << p_res.fer
                  << ", BER=" << std::scientific << std::setprecision(3) << p_res.ber
                  << ", AvgIter=" << std::fixed << std::setprecision(2) << p_res.avg_iters
                  << ", Speed=" << std::fixed << std::setprecision(2) << p_res.msps << " Msps ("
                  << std::fixed << std::setprecision(1) << p_res.elapsed_s << "s)\n" << std::endl;
    }

    // Print Comparative Summary Table
    std::cout << "\n==============================================================================================================" << std::endl;
    std::cout << "                                     HEAD-TO-HEAD COMPARISON RESULTS TABLE                                    " << std::endl;
    std::cout << "==============================================================================================================" << std::endl;
    std::cout << std::left << std::setw(32) << "Decoder Model"
              << std::setw(12) << "Eb/N0 (dB)"
              << std::setw(16) << "Total Blocks"
              << std::setw(10) << "FE"
              << std::setw(14) << "FER"
              << std::setw(14) << "BER"
              << std::setw(12) << "Avg Iter"
              << std::setw(14) << "Speed (Msps)"
              << std::endl;
    std::cout << "--------------------------------------------------------------------------------------------------------------" << std::endl;

    for (const auto& r : results) {
        std::cout << std::left << std::setw(32) << r.name
                  << std::fixed << std::setprecision(2) << std::setw(12) << r.eb_n0_db
                  << std::setw(16) << r.total_blocks
                  << std::setw(10) << r.frame_errors
                  << std::scientific << std::setprecision(3) << std::setw(14) << r.fer
                  << std::scientific << std::setprecision(3) << std::setw(14) << r.ber
                  << std::fixed << std::setprecision(2) << std::setw(12) << r.avg_iters
                  << std::fixed << std::setprecision(2) << std::setw(14) << r.msps
                  << std::endl;
    }
    std::cout << "==============================================================================================================\n" << std::endl;

    // Export CSV (Append mode to accumulate sweep points)
    fs::create_directories(fs::path(csv_path).parent_path());
    bool file_exists = fs::exists(csv_path) && fs::file_size(csv_path) > 0;
    std::ofstream f_csv(csv_path, std::ios::app);
    if (f_csv.is_open()) {
        if (!file_exists) {
            f_csv << "model,ebn0_db,total_blocks,frame_errors,bit_errors,fer,ber,avg_iters,speed_msps,elapsed_seconds\n";
        }
        for (const auto& r : results) {
            f_csv << r.name << "," << r.eb_n0_db << "," << r.total_blocks << "," << r.frame_errors << ","
                  << r.bit_errors << "," << r.fer << "," << r.ber << "," << r.avg_iters << ","
                  << r.msps << "," << r.elapsed_s << "\n";
        }
        f_csv.close();
        std::cout << "[SUCCESS] Wrote comparative results CSV to: " << csv_path << std::endl;
    }

    return 0;
}
