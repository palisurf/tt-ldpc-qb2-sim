// ==============================================================================
// tests/test_circulant_dma_stepper.cpp
//
// Tenstorrent QuietBox 2: 4-Codeword Circulant DMA Golden Stepper
// Simulates the line-rate 512-bit DMA burst gather/scatter on Blackhole Tensix:
//   - 4 Codewords Interleaved (1 variable node = 8 Bytes = 64 bits)
//   - Zero indirect pointer tables (driven by ar4ja_circulant_descriptors_4cw.h)
//   - 128-Byte Contiguous Slice DMA Bursts (2 clock cycles @ 64 B/cycle)
//   - 40-Tile Face-Aligned Schedule (32 Deg-6 Tiles + 8 Deg-3 Tiles)
//   - Branchless Approximate-Min* Check Node Kernel on DEST registers
//   - Simulation-First Gatekeeper for hardware bit-exactness
// ==============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cassert>
#include <random>
#include <chrono>
#include <atomic>
#include <omp.h>

#include "../engines/circulant_dma_4cw/ar4ja_circulant_descriptors_4cw.h"
#include "../engines/tile_matrix_4cw/ldpc_tile_matrix_core.h"

using namespace QC_LDPC_4CW;

// ------------------------------------------------------------------------------
// Dr. Jones' Branchless Approximate-Min* Evaluation (True Recursive Formulation)
// ------------------------------------------------------------------------------
inline float approx_min_star(float a, float b) {
    float min_v = std::min(a, b);
    float diff  = std::abs(a - b);

    // 2-region piecewise linear correction from MILCOM 2003
    static const float c_0_69315   = 0.69315f;
    static const float c_neg_0_4   = -0.40f;
    static const float c_0_65715   = 0.65715f;
    static const float c_neg_0_328 = -0.328f;

    float line1 = c_0_69315 + c_neg_0_4 * diff;
    float line2 = c_0_65715 + c_neg_0_328 * diff;
    float corr  = std::max(std::max(line1, line2), 0.0f);

    return std::max(min_v - corr, 0.0f);
}

// ------------------------------------------------------------------------------
// Single-Tile Circulant DMA Stepper Pass
// ------------------------------------------------------------------------------
struct TileScratchpad {
    alignas(64) uint16_t face0[256]; // 16x16 bfloat16 words = 512 Bytes
    alignas(64) uint16_t face1[256];
    alignas(64) uint16_t face2[256];
    alignas(64) uint16_t face3[256];
};

inline void gather_slice(uint16_t* dest, const CirculantSliceDesc& desc, const char* vnode_base) {
    std::memcpy(dest, vnode_base + desc.offset_a, desc.len_a);
    if (desc.len_b > 0) {
        std::memcpy(reinterpret_cast<char*>(dest) + desc.len_a, vnode_base + desc.offset_b, desc.len_b);
    }
}

inline void scatter_slice(const CirculantSliceDesc& desc, const uint16_t* src, char* vnode_base) {
    std::memcpy(vnode_base + desc.offset_a, src, desc.len_a);
    if (desc.len_b > 0) {
        std::memcpy(vnode_base + desc.offset_b, reinterpret_cast<const char*>(src) + desc.len_a, desc.len_b);
    }
}

inline void step_tile_circulant_dma(
    uint32_t tile_idx,
    Packed4CW_BF16* vnodes,      // CB_VNODE (2560 nodes x 8 Bytes)
    Packed4CW_BF16* r_msg_tile,  // CB_R Check Message FIFO page (192 edges x 8 Bytes)
    bool is_iter0,
    TileScratchpad& scratch,
    uint32_t active_mask = 0xF,
    bool passthrough_mode = false
) {
    bool is_deg6 = (tile_idx < DEG6_TILES);
    uint32_t num_checks = is_deg6 ? 32 : 64;
    uint32_t deg        = is_deg6 ? 6 : 3;

    char* vnode_base = reinterpret_cast<char*>(vnodes);

    // =========================================================================
    // STAGE 1: CONTIGUOUS 128-BYTE DMA GATHER BURSTS (DUAL WORKERS)
    // =========================================================================
    // Worker 0: Gathers Face 0 and Face 2
    #pragma GCC unroll 3
    for (uint32_t d = 0; d < 3; ++d) {
        gather_slice(scratch.face0 + d * 64, TILE_SLICES_W0_F0[tile_idx][d], vnode_base);
        gather_slice(scratch.face2 + d * 64, TILE_SLICES_W0_F2[tile_idx][d], vnode_base);
    }
    // Worker 1: Gathers Face 1 and Face 3
    #pragma GCC unroll 3
    for (uint32_t d = 0; d < 3; ++d) {
        gather_slice(scratch.face1 + d * 64, TILE_SLICES_W1_F1[tile_idx][d], vnode_base);
        gather_slice(scratch.face3 + d * 64, TILE_SLICES_W1_F3[tile_idx][d], vnode_base);
    }

    // =========================================================================
    // STAGE 2: BRANCHLESS APPROXIMATE-MIN* CHECK NODE MATH
    // =========================================================================
    if (!passthrough_mode) {
        // Process across all 4 independent codewords in parallel
        for (uint32_t cw = 0; cw < 4; ++cw) {
            if (!(active_mask & (1 << cw))) continue; // Codeword frozen upon parity convergence
            for (uint32_t c = 0; c < num_checks; ++c) {
                float q[6];
                float r_in[6];
                uint32_t edge_offset_base = c * deg;

                // Extract incoming LLRs from face registers
                if (is_deg6) {
                    bool is_w1 = (c >= 16);
                    uint32_t c_loc = c % 16;
                    uint16_t* f_low  = is_w1 ? scratch.face1 : scratch.face0;
                    uint16_t* f_high = is_w1 ? scratch.face3 : scratch.face2;

                    for (uint32_t d = 0; d < 3; ++d) {
                        uint32_t idx = (d * 4 + (c_loc / 4)) * 16 + ((c_loc % 4) * 4 + cw);
                        float l_in = bf16_to_fp32(f_low[idx]);
                        r_in[d] = is_iter0 ? 0.0f : bf16_to_fp32(r_msg_tile[edge_offset_base + d].cw[cw]);
                        q[d] = l_in - r_in[d];
                    }
                    for (uint32_t d = 3; d < 6; ++d) {
                        uint32_t idx = ((d - 3) * 4 + (c_loc / 4)) * 16 + ((c_loc % 4) * 4 + cw);
                        float l_in = bf16_to_fp32(f_high[idx]);
                        r_in[d] = is_iter0 ? 0.0f : bf16_to_fp32(r_msg_tile[edge_offset_base + d].cw[cw]);
                        q[d] = l_in - r_in[d];
                    }
                } else { // Degree 3
                    uint32_t face_idx = c / 16;
                    uint32_t c_loc    = c % 16;
                    uint16_t* f = (face_idx == 0) ? scratch.face0 :
                                  (face_idx == 1) ? scratch.face1 :
                                  (face_idx == 2) ? scratch.face2 : scratch.face3;

                    for (uint32_t d = 0; d < 3; ++d) {
                        uint32_t idx = (d * 4 + (c_loc / 4)) * 16 + ((c_loc % 4) * 4 + cw);
                        float l_in = bf16_to_fp32(f[idx]);
                        r_in[d] = is_iter0 ? 0.0f : bf16_to_fp32(r_msg_tile[edge_offset_base + d].cw[cw]);
                        q[d] = l_in - r_in[d];
                    }
                }

                // Exact Pairwise Tree Approximate-Min* Check Node Kernel (Matching TRISC compute_trisc.cpp)
                float r_out[6];
                if (deg == 6) {
                    float m0 = std::abs(q[0]), m3 = std::abs(q[3]);
                    float m1 = std::abs(q[1]), m4 = std::abs(q[4]);
                    float m2 = std::abs(q[2]), m5 = std::abs(q[5]);

                    float s0 = (q[0] < 0.0f) ? -1.0f : 1.0f;
                    float s1 = (q[1] < 0.0f) ? -1.0f : 1.0f;
                    float s2 = (q[2] < 0.0f) ? -1.0f : 1.0f;
                    float s3 = (q[3] < 0.0f) ? -1.0f : 1.0f;
                    float s4 = (q[4] < 0.0f) ? -1.0f : 1.0f;
                    float s5 = (q[5] < 0.0f) ? -1.0f : 1.0f;

                    float s_tot = (s0 * s3) * (s1 * s4) * (s2 * s5);

                    float m03 = approx_min_star(m0, m3);
                    float m14 = approx_min_star(m1, m4);
                    float m25 = approx_min_star(m2, m5);

                    float r_mag_0 = approx_min_star(approx_min_star(m14, m25), m3);
                    float r_mag_3 = approx_min_star(approx_min_star(m14, m25), m0);
                    float r_mag_1 = approx_min_star(approx_min_star(m03, m25), m4);
                    float r_mag_4 = approx_min_star(approx_min_star(m03, m25), m1);
                    float r_mag_2 = approx_min_star(approx_min_star(m03, m14), m5);
                    float r_mag_5 = approx_min_star(approx_min_star(m03, m14), m2);

                    r_out[0] = s_tot * s0 * r_mag_0;
                    r_out[1] = s_tot * s1 * r_mag_1;
                    r_out[2] = s_tot * s2 * r_mag_2;
                    r_out[3] = s_tot * s3 * r_mag_3;
                    r_out[4] = s_tot * s4 * r_mag_4;
                    r_out[5] = s_tot * s5 * r_mag_5;
                } else { // deg == 3
                    float m0 = std::abs(q[0]);
                    float m1 = std::abs(q[1]);
                    float m2 = std::abs(q[2]);

                    float s0 = (q[0] < 0.0f) ? -1.0f : 1.0f;
                    float s1 = (q[1] < 0.0f) ? -1.0f : 1.0f;
                    float s2 = (q[2] < 0.0f) ? -1.0f : 1.0f;

                    float s_tot = (s0 * s1) * s2;

                    float r_mag_0 = approx_min_star(m1, m2);
                    float r_mag_1 = approx_min_star(m0, m2);
                    float r_mag_2 = approx_min_star(m0, m1);

                    r_out[0] = s_tot * s0 * r_mag_0;
                    r_out[1] = s_tot * s1 * r_mag_1;
                    r_out[2] = s_tot * s2 * r_mag_2;
                }

                for (uint32_t d = 0; d < deg; ++d) {
                    // Write R_new to FIFO
                    r_msg_tile[edge_offset_base + d].cw[cw] = fp32_to_bf16(r_out[d]);

                    // Posterior update: L_new = Q + R_new
                    float l_new = q[d] + r_out[d];

                    // Writeback into face registers
                    uint16_t bf_val = fp32_to_bf16(l_new);
                    if (is_deg6) {
                        bool is_w1 = (c >= 16);
                        uint32_t c_loc = c % 16;
                        uint16_t* f = (d < 3) ? (is_w1 ? scratch.face1 : scratch.face0)
                                              : (is_w1 ? scratch.face3 : scratch.face2);
                        uint32_t d_eff = (d < 3) ? d : (d - 3);
                        uint32_t idx = (d_eff * 4 + (c_loc / 4)) * 16 + ((c_loc % 4) * 4 + cw);
                        f[idx] = bf_val;
                    } else {
                        uint32_t face_idx = c / 16;
                        uint32_t c_loc    = c % 16;
                        uint16_t* f = (face_idx == 0) ? scratch.face0 :
                                      (face_idx == 1) ? scratch.face1 :
                                      (face_idx == 2) ? scratch.face2 : scratch.face3;
                        uint32_t idx = (d * 4 + (c_loc / 4)) * 16 + ((c_loc % 4) * 4 + cw);
                        f[idx] = bf_val;
                    }
                }
            }
        }
    }

    // =========================================================================
    // STAGE 3: CONTIGUOUS 128-BYTE DMA SCATTER BURSTS (DIRECT WRITEBACK)
    // =========================================================================
    // Worker 0: Scatters Face 0 and Face 2
    #pragma GCC unroll 3
    for (uint32_t d = 0; d < 3; ++d) {
        scatter_slice(TILE_SLICES_W0_F0[tile_idx][d], scratch.face0 + d * 64, vnode_base);
        scatter_slice(TILE_SLICES_W0_F2[tile_idx][d], scratch.face2 + d * 64, vnode_base);
    }
    // Worker 1: Scatters Face 1 and Face 3
    #pragma GCC unroll 3
    for (uint32_t d = 0; d < 3; ++d) {
        scatter_slice(TILE_SLICES_W1_F1[tile_idx][d], scratch.face1 + d * 64, vnode_base);
        scatter_slice(TILE_SLICES_W1_F3[tile_idx][d], scratch.face3 + d * 64, vnode_base);
    }
}

// ------------------------------------------------------------------------------
// PassThrough Verification Mode (Test Gather/Scatter Symmetry with Zero Clashing)
// ------------------------------------------------------------------------------
bool run_passthrough_test() {
    std::cout << "================================================================================" << std::endl;
    std::cout << "RUNNING CIRCULANT DMA PASSTHROUGH VERIFICATION (4 CODEWORDS)" << std::endl;
    std::cout << "================================================================================" << std::endl;

    alignas(64) std::vector<Packed4CW_BF16> vnodes(2560);
    alignas(64) std::vector<Packed4CW_BF16> vnodes_orig(2560);
    alignas(64) std::vector<Packed4CW_BF16> r_msg(TOTAL_TILES * 192);

    for (uint32_t v = 0; v < 2560; ++v) {
        for (uint32_t cw = 0; cw < 4; ++cw) {
            float val = 1024.0f + static_cast<float>(v) + 0.1f * cw;
            vnodes[v].cw[cw] = fp32_to_bf16(val);
            vnodes_orig[v].cw[cw] = vnodes[v].cw[cw];
        }
    }

    TileScratchpad scratch;

    // Run 5 full circular schedule passes with identity pass-through
    for (uint32_t iter = 0; iter < 5; ++iter) {
        for (uint32_t t = 0; t < TOTAL_TILES; ++t) {
            step_tile_circulant_dma(t, vnodes.data(), &r_msg[t * 192], (iter == 0), scratch, 0xF, true);
        }
    }

    uint32_t mismatches = 0;
    for (uint32_t v = 0; v < 2560; ++v) {
        for (uint32_t cw = 0; cw < 4; ++cw) {
            if (vnodes[v].cw[cw] != vnodes_orig[v].cw[cw]) {
                if (mismatches < 5) {
                    std::cout << "[MISMATCH] v = " << v << ", cw = " << cw
                              << " Orig: " << vnodes_orig[v].cw[cw]
                              << " Got: " << vnodes[v].cw[cw] << std::endl;
                }
                mismatches++;
            }
        }
    }

    if (mismatches == 0) {
        std::cout << "[SUCCESS] 100% BIT-EXACT GATHER/SCATTER SYMMETRY! 0 MISMATCHES ACROSS 10,240 WORDS!" << std::endl;
        return true;
    } else {
        std::cout << "[FAILURE] " << mismatches << " MISMATCHES DETECTED!" << std::endl;
        return false;
    }
}

// ------------------------------------------------------------------------------
// Monte Carlo Simulation Engine
// ------------------------------------------------------------------------------
void run_monte_carlo(double ebn0_db, uint64_t max_blocks, uint64_t target_fe, uint32_t max_iter, int num_threads) {
    std::cout << "================================================================================" << std::endl;
    std::cout << "STARTING CIRCULANT 512-BIT DMA MONTE CARLO SIMULATION (4 CODEWORDS)" << std::endl;
    std::cout << "  Eb/N0:          " << std::fixed << std::setprecision(2) << ebn0_db << " dB" << std::endl;
    std::cout << "  Max Blocks:     " << max_blocks << std::endl;
    std::cout << "  Target Errors:  " << target_fe << std::endl;
    std::cout << "  Max Iterations: " << max_iter << std::endl;
    std::cout << "  Host Threads:   " << num_threads << std::endl;
    std::cout << "================================================================================" << std::endl;

    double rate = 0.5; // CCSDS AR4JA Rate-1/2
    double ebn0_lin = std::pow(10.0, ebn0_db / 10.0);
    double sigma = 1.0 / std::sqrt(2.0 * rate * ebn0_lin);
    double llr_scale = 2.0 / (sigma * sigma);

    std::atomic<uint64_t> total_sim_blocks(0);
    std::atomic<uint64_t> total_fe(0);
    std::atomic<uint64_t> total_be(0);
    std::atomic<uint64_t> total_iterations(0);

    auto start_time = std::chrono::steady_clock::now();

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        std::mt19937_64 rng(1337 + tid * 10007);
        std::normal_distribution<float> dist(0.0f, static_cast<float>(sigma));

        alignas(64) std::vector<Packed4CW_BF16> vnodes(2560);
        alignas(64) std::vector<Packed4CW_BF16> r_msg(TOTAL_TILES * 192);
        TileScratchpad scratch;

        while (total_fe.load(std::memory_order_relaxed) < target_fe &&
               total_sim_blocks.load(std::memory_order_relaxed) < max_blocks) {

            // Channel Initialization: All-zero codeword (BPSK: +1.0) + AWGN noise
            for (uint32_t v = 0; v < 2560; ++v) {
                if (v >= 2048) {
                    // Punctured nodes (2048..2559) initialized to zero LLR
                    vnodes[v].set_all(0);
                } else {
                    for (uint32_t cw = 0; cw < 4; ++cw) {
                        float y = 1.0f + dist(rng);
                        float llr = static_cast<float>(llr_scale) * y;
                        vnodes[v].cw[cw] = fp32_to_bf16(llr);
                    }
                }
            }

            uint32_t active_mask = 0xF; // 4 active codewords
            uint32_t cw_iters[4] = {max_iter, max_iter, max_iter, max_iter};

            for (uint32_t iter = 0; iter < max_iter; ++iter) {
                bool is_iter0 = (iter == 0);

                for (uint32_t t = 0; t < TOTAL_TILES; ++t) {
                    step_tile_circulant_dma(t, vnodes.data(), &r_msg[t * 192], is_iter0, scratch, active_mask);
                }

                // Full Parity Syndrome Verification (H * c^T == 0) across all 1536 checks
                for (uint32_t cw = 0; cw < 4; ++cw) {
                    if (!(active_mask & (1 << cw))) continue;

                    bool satisfied = true;
                    for (uint32_t m = 0; m < 1536; ++m) {
                        int syn = 0;
                        uint32_t deg = (m < 512) ? 3 : 6;
                        for (uint32_t d = 0; d < deg; ++d) {
                            uint32_t v = CHECK_TO_VAR[m][d];
                            if (bf16_to_fp32(vnodes[v].cw[cw]) < 0.0f) syn ^= 1;
                        }
                        if (syn != 0) {
                            satisfied = false;
                            break;
                        }
                    }
                    if (satisfied) {
                        cw_iters[cw] = iter + 1;
                        active_mask &= ~(1 << cw); // Freeze converged codeword
                    }
                }

                if (active_mask == 0) break;
            }

            // Accumulate statistics
            uint32_t batch_fe = 0;
            uint32_t batch_be = 0;
            for (uint32_t cw = 0; cw < 4; ++cw) {
                if (active_mask & (1 << cw)) {
                    // Did not converge
                    batch_fe++;
                    for (uint32_t k = 0; k < 1024; ++k) {
                        float val = bf16_to_fp32(vnodes[k].cw[cw]);
                        if (val <= 0.0f) batch_be++;
                    }
                }
            }

            total_sim_blocks.fetch_add(4, std::memory_order_relaxed);
            total_fe.fetch_add(batch_fe, std::memory_order_relaxed);
            total_be.fetch_add(batch_be, std::memory_order_relaxed);
            total_iterations.fetch_add(cw_iters[0] + cw_iters[1] + cw_iters[2] + cw_iters[3], std::memory_order_relaxed);

            uint64_t cur_blocks = total_sim_blocks.load(std::memory_order_relaxed);
            if (cur_blocks % 40000 == 0 && tid == 0) {
                auto now = std::chrono::steady_clock::now();
                double el = std::chrono::duration<double>(now - start_time).count();
                double fer = static_cast<double>(total_fe.load()) / cur_blocks;
                double mean_iter = static_cast<double>(total_iterations.load()) / cur_blocks;
                double msps = (cur_blocks * 2048.0) / (el * 1e6);
                std::cout << "\r[SIM] Blocks: " << cur_blocks
                          << " | FE: " << total_fe.load()
                          << " | FER: " << std::scientific << std::setprecision(3) << fer
                          << " | MeanIter: " << std::fixed << std::setprecision(2) << mean_iter
                          << " | Speed: " << std::setprecision(2) << msps << " Msps" << std::flush;
            }
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    double total_sec = std::chrono::duration<double>(end_time - start_time).count();
    uint64_t fin_blocks = total_sim_blocks.load();
    uint64_t fin_fe = total_fe.load();
    uint64_t fin_be = total_be.load();
    double fin_fer = static_cast<double>(fin_fe) / fin_blocks;
    double fin_ber = static_cast<double>(fin_be) / (fin_blocks * 1024.0);
    double fin_iter = static_cast<double>(total_iterations.load()) / fin_blocks;
    double fin_msps = (fin_blocks * 2048.0) / (total_sec * 1e6);

    std::cout << "\n================================================================================" << std::endl;
    std::cout << "MONTE CARLO SIMULATION RESULTS (CIRCULANT 512-BIT DMA 4-CW)" << std::endl;
    std::cout << "  Total Blocks:     " << fin_blocks << std::endl;
    std::cout << "  Frame Errors:     " << fin_fe << std::endl;
    std::cout << "  Bit Errors:       " << fin_be << std::endl;
    std::cout << "  FER:              " << std::scientific << std::setprecision(4) << fin_fer << std::endl;
    std::cout << "  BER:              " << std::scientific << std::setprecision(4) << fin_ber << std::endl;
    std::cout << "  Mean Iterations:  " << std::fixed << std::setprecision(3) << fin_iter << std::endl;
    std::cout << "  Elapsed Time:     " << std::setprecision(2) << total_sec << " s" << std::endl;
    std::cout << "  Host Throughput:  " << std::setprecision(2) << fin_msps << " Msps ("
              << (fin_blocks / total_sec) << " blocks/sec)" << std::endl;
    std::cout << "================================================================================" << std::endl;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--passthrough") {
        bool pass = run_passthrough_test();
        return pass ? 0 : 1;
    }

    if (argc > 1 && std::string(argv[1]) == "--monte-carlo") {
        double ebn0 = (argc > 2) ? std::stod(argv[2]) : 2.0;
        uint64_t blocks = (argc > 3) ? std::stoull(argv[3]) : 100000;
        uint64_t target_fe = (argc > 4) ? std::stoull(argv[4]) : 25;
        uint32_t max_iter = (argc > 5) ? std::stoul(argv[5]) : 200;
        int threads = (argc > 6) ? std::stoi(argv[6]) : omp_get_max_threads();
        run_monte_carlo(ebn0, blocks, target_fe, max_iter, threads);
        return 0;
    }

    std::cout << "Usage:" << std::endl;
    std::cout << "  " << argv[0] << " --passthrough" << std::endl;
    std::cout << "  " << argv[0] << " --monte-carlo [ebn0] [max_blocks] [target_fe] [max_iter] [threads]" << std::endl;
    return 0;
}
