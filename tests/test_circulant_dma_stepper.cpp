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
//   - Zero-Transpose Vertical SIMD Check Node SFPU Math
//   - Native Tenstorrent SFPU Vector Emulation (sfpi_host_emu.h)
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
#include "../engines/tile_matrix_4cw/sfpi_host_emu.h"

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
// Single-Tile Circulant DMA Stepper Pass Data Structures
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

inline void scatter_slice_masked(const CirculantSliceDesc& desc, const uint16_t* src, char* vnode_base, uint32_t active_mask) {
    if (active_mask == 0xF) {
        scatter_slice(desc, src, vnode_base);
        return;
    }
    auto scatter_bytes = [&](uint32_t offset, const uint16_t* s, uint32_t len) {
        Packed4CW_BF16* dest = reinterpret_cast<Packed4CW_BF16*>(vnode_base + offset);
        const Packed4CW_BF16* source = reinterpret_cast<const Packed4CW_BF16*>(s);
        uint32_t num_words = len / sizeof(Packed4CW_BF16);
        for (uint32_t i = 0; i < num_words; ++i) {
            for (uint32_t cw = 0; cw < 4; ++cw) {
                if (active_mask & (1 << cw)) {
                    dest[i].cw[cw] = source[i].cw[cw];
                }
            }
        }
    };
    scatter_bytes(desc.offset_a, src, desc.len_a);
    if (desc.len_b > 0) {
        scatter_bytes(desc.offset_b, reinterpret_cast<const uint16_t*>(reinterpret_cast<const char*>(src) + desc.len_a), desc.len_b);
    }
}

// ------------------------------------------------------------------------------
// Bit-Exact Zero-Transpose SFPI Vector TRISC Check Node Compute
// Directly pairs Face 0 with Face 2 (Worker 0) and Face 1 with Face 3 (Worker 1)
// ------------------------------------------------------------------------------
inline void compute_tile_deg6_sfpi(
    uint32_t iter,
    float damping_factor = 1.0f
) {
    // 2 Workers: Worker 0 (Checks 0..15: Face 0 & 2), Worker 1 (Checks 16..31: Face 1 & 3)
    const int w_low_bases[2]  = {0, 8};   // Face 0, Face 1
    const int w_high_bases[2] = {16, 24}; // Face 2, Face 3

    if (iter != 0) {
        for (int r = 0; r < 32; ++r) {
            sfpi::dst_reg[r] = sfpi::round_bf16(sfpi::vFloat(sfpi::dst_reg[r]) - sfpi::vFloat(sfpi::dst_reg[32 + r]));
        }
    }

    for (int w = 0; w < 2; ++w) {
        int f_low  = w_low_bases[w];
        int f_high = w_high_bases[w];

        for (int sub = 0; sub < 2; ++sub) {
            int r0 = f_low  + 0 + sub; // Edge 0
            int r3 = f_high + 0 + sub; // Edge 3
            int r1 = f_low  + 2 + sub; // Edge 1
            int r4 = f_high + 2 + sub; // Edge 4
            int r2 = f_low  + 4 + sub; // Edge 2
            int r5 = f_high + 4 + sub; // Edge 5

            sfpi::vFloat q0 = sfpi::dst_reg[r0];
            sfpi::vFloat q3 = sfpi::dst_reg[r3];
            sfpi::vFloat q1 = sfpi::dst_reg[r1];
            sfpi::vFloat q4 = sfpi::dst_reg[r4];
            sfpi::vFloat q2 = sfpi::dst_reg[r2];
            sfpi::vFloat q5 = sfpi::dst_reg[r5];

            // Magnitudes and Signs
            sfpi::vFloat m0 = sfpi::abs(q0);
            sfpi::vFloat m3 = sfpi::abs(q3);
            sfpi::vFloat m1 = sfpi::abs(q1);
            sfpi::vFloat m4 = sfpi::abs(q4);
            sfpi::vFloat m2 = sfpi::abs(q2);
            sfpi::vFloat m5 = sfpi::abs(q5);

            sfpi::vInt s0 = sfpi::as<sfpi::vInt>(q0);
            sfpi::vInt s3 = sfpi::as<sfpi::vInt>(q3);
            sfpi::vInt s1 = sfpi::as<sfpi::vInt>(q1);
            sfpi::vInt s4 = sfpi::as<sfpi::vInt>(q4);
            sfpi::vInt s2 = sfpi::as<sfpi::vInt>(q2);
            sfpi::vInt s5 = sfpi::as<sfpi::vInt>(q5);

            int sc0 = f_low + 6;
            // Native 1-Cycle Integer XOR Sign Parity -> stored in dst_reg[sc0]
            {
                sfpi::vInt s0 = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[r0]));
                sfpi::vInt s3 = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[r3]));
                sfpi::vInt s1 = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[r1]));
                sfpi::vInt s4 = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[r4]));
                sfpi::vInt s2 = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[r2]));
                sfpi::vInt s5 = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[r5]));
                sfpi::dst_reg[sc0] = sfpi::as<sfpi::vFloat>(((s0 ^ s3) ^ (s1 ^ s4)) ^ (s2 ^ s5));
            }

            // Step 1: Compute all 3 pairwise checks into transient FP32 LRegs
            sfpi::vFloat m03 = sfpi::sfpi_amin_star_branchless(sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r0])),
                                                               sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r3])));
            sfpi::vFloat m14 = sfpi::sfpi_amin_star_branchless(sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r1])),
                                                               sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r4])));
            sfpi::vFloat m25 = sfpi::sfpi_amin_star_branchless(sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r2])),
                                                               sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r5])));

            // Step 2: Edge pair (0, 3) using m_oth0 = min*(m14, m25)
            {
                sfpi::vFloat m_oth0 = sfpi::sfpi_amin_star_branchless(m14, m25);
                sfpi::vFloat r_mag0 = sfpi::sfpi_amin_star_branchless(m_oth0, sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r3]))) * damping_factor;
                sfpi::vFloat r_mag3 = sfpi::sfpi_amin_star_branchless(m_oth0, sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r0]))) * damping_factor;

                // Finish Edge 0:
                {
                    sfpi::vFloat q0 = sfpi::dst_reg[r0];
                    sfpi::vInt s_tot = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc0]));
                    sfpi::vFloat r_out0 = sfpi::round_bf16(sfpi::copysgn(r_mag0, s_tot ^ sfpi::as<sfpi::vInt>(q0)));
                    sfpi::dst_reg[32 + r0] = sfpi::round_bf16(q0 + r_out0);
                    sfpi::dst_reg[r0]      = r_out0;
                }

                // Finish Edge 3:
                {
                    sfpi::vFloat q3 = sfpi::dst_reg[r3];
                    sfpi::vInt s_tot = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc0]));
                    sfpi::vFloat r_out3 = sfpi::round_bf16(sfpi::copysgn(r_mag3, s_tot ^ sfpi::as<sfpi::vInt>(q3)));
                    sfpi::dst_reg[32 + r3] = sfpi::round_bf16(q3 + r_out3);
                    sfpi::dst_reg[r3]      = r_out3;
                }
            }

            // Step 3: Edge pair (1, 4) using m_oth1 = min*(m03, m25). m25 dies!
            {
                sfpi::vFloat m_oth1 = sfpi::sfpi_amin_star_branchless(m03, m25);
                sfpi::vFloat r_mag1 = sfpi::sfpi_amin_star_branchless(m_oth1, sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r4]))) * damping_factor;
                sfpi::vFloat r_mag4 = sfpi::sfpi_amin_star_branchless(m_oth1, sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r1]))) * damping_factor;

                // Finish Edge 1:
                {
                    sfpi::vFloat q1 = sfpi::dst_reg[r1];
                    sfpi::vInt s_tot = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc0]));
                    sfpi::vFloat r_out1 = sfpi::round_bf16(sfpi::copysgn(r_mag1, s_tot ^ sfpi::as<sfpi::vInt>(q1)));
                    sfpi::dst_reg[32 + r1] = sfpi::round_bf16(q1 + r_out1);
                    sfpi::dst_reg[r1]      = r_out1;
                }

                // Finish Edge 4:
                {
                    sfpi::vFloat q4 = sfpi::dst_reg[r4];
                    sfpi::vInt s_tot = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc0]));
                    sfpi::vFloat r_out4 = sfpi::round_bf16(sfpi::copysgn(r_mag4, s_tot ^ sfpi::as<sfpi::vInt>(q4)));
                    sfpi::dst_reg[32 + r4] = sfpi::round_bf16(q4 + r_out4);
                    sfpi::dst_reg[r4]      = r_out4;
                }
            }

            // Step 4: Edge pair (2, 5) using m_oth2 = min*(m03, m14). Both m03 and m14 die!
            {
                sfpi::vFloat m_oth2 = sfpi::sfpi_amin_star_branchless(m03, m14);
                sfpi::vFloat r_mag2 = sfpi::sfpi_amin_star_branchless(m_oth2, sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r5]))) * damping_factor;
                sfpi::vFloat r_mag5 = sfpi::sfpi_amin_star_branchless(m_oth2, sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r2]))) * damping_factor;

                // Finish Edge 2:
                {
                    sfpi::vFloat q2 = sfpi::dst_reg[r2];
                    sfpi::vInt s_tot = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc0]));
                    sfpi::vFloat r_out2 = sfpi::round_bf16(sfpi::copysgn(r_mag2, s_tot ^ sfpi::as<sfpi::vInt>(q2)));
                    sfpi::dst_reg[32 + r2] = sfpi::round_bf16(q2 + r_out2);
                    sfpi::dst_reg[r2]      = r_out2;
                }

                // Finish Edge 5:
                {
                    sfpi::vFloat q5 = sfpi::dst_reg[r5];
                    sfpi::vInt s_tot = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc0]));
                    sfpi::vFloat r_out5 = sfpi::round_bf16(sfpi::copysgn(r_mag5, s_tot ^ sfpi::as<sfpi::vInt>(q5)));
                    sfpi::dst_reg[32 + r5] = sfpi::round_bf16(q5 + r_out5);
                    sfpi::dst_reg[r5]      = r_out5;
                }
            }
        }

        // Zero out unused scratch/padding registers (regs 6, 7 of each face)
        sfpi::dst_reg[f_low  + 6] = 0.0f;
        sfpi::dst_reg[f_low  + 7] = 0.0f;
        sfpi::dst_reg[f_high + 6] = 0.0f;
        sfpi::dst_reg[f_high + 7] = 0.0f;
        sfpi::dst_reg[32 + f_low  + 6] = 0.0f;
        sfpi::dst_reg[32 + f_low  + 7] = 0.0f;
        sfpi::dst_reg[32 + f_high + 6] = 0.0f;
        sfpi::dst_reg[32 + f_high + 7] = 0.0f;
    }
}

// ------------------------------------------------------------------------------
// Bit-Exact Zero-Transpose SFPI Vector TRISC Check Node Compute for Degree-3
// ------------------------------------------------------------------------------
inline void compute_tile_deg3_sfpi(
    uint32_t iter,
    float damping_factor = 1.0f
) {
    if (iter != 0) {
        for (int r = 0; r < 32; ++r) {
            sfpi::dst_reg[r] = sfpi::round_bf16(sfpi::vFloat(sfpi::dst_reg[r]) - sfpi::vFloat(sfpi::dst_reg[32 + r]));
        }
    }

    const int face_bases[4] = {0, 8, 16, 24};
    for (int f = 0; f < 4; ++f) {
        int base_r = face_bases[f];
        for (int sub = 0; sub < 2; ++sub) {
            int r0 = base_r + 0 + sub;
            int r1 = base_r + 2 + sub;
            int r2 = base_r + 4 + sub;

            sfpi::vFloat q0 = sfpi::dst_reg[r0];
            sfpi::vFloat q1 = sfpi::dst_reg[r1];
            sfpi::vFloat q2 = sfpi::dst_reg[r2];

            sfpi::vFloat m0 = sfpi::abs(q0);
            sfpi::vFloat m1 = sfpi::abs(q1);
            sfpi::vFloat m2 = sfpi::abs(q2);

            sfpi::vInt s0 = sfpi::as<sfpi::vInt>(q0);
            sfpi::vInt s1 = sfpi::as<sfpi::vInt>(q1);
            sfpi::vInt s2 = sfpi::as<sfpi::vInt>(q2);

            sfpi::vInt s_tot = (s0 ^ s1) ^ s2;

            sfpi::vFloat r_mag0 = sfpi::sfpi_amin_star_branchless(m1, m2);
            sfpi::vFloat r_mag1 = sfpi::sfpi_amin_star_branchless(m0, m2);
            sfpi::vFloat r_mag2 = sfpi::sfpi_amin_star_branchless(m0, m1);

            if (damping_factor < 1.0f) {
                r_mag0 = r_mag0 * damping_factor;
                r_mag1 = r_mag1 * damping_factor;
                r_mag2 = r_mag2 * damping_factor;
            }

            sfpi::vFloat r_out0 = sfpi::round_bf16(sfpi::copysgn(r_mag0, s_tot ^ s0));
            sfpi::vFloat r_out1 = sfpi::round_bf16(sfpi::copysgn(r_mag1, s_tot ^ s1));
            sfpi::vFloat r_out2 = sfpi::round_bf16(sfpi::copysgn(r_mag2, s_tot ^ s2));

            sfpi::dst_reg[32 + r0] = sfpi::round_bf16(q0 + r_out0);
            sfpi::dst_reg[r0]      = r_out0;

            sfpi::dst_reg[32 + r1] = sfpi::round_bf16(q1 + r_out1);
            sfpi::dst_reg[r1]      = r_out1;

            sfpi::dst_reg[32 + r2] = sfpi::round_bf16(q2 + r_out2);
            sfpi::dst_reg[r2]      = r_out2;
        }

        sfpi::dst_reg[base_r + 6] = 0.0f;
        sfpi::dst_reg[base_r + 7] = 0.0f;
        sfpi::dst_reg[32 + base_r + 6] = 0.0f;
        sfpi::dst_reg[32 + base_r + 7] = 0.0f;
    }
}

// ------------------------------------------------------------------------------
// Single-Tile Circulant DMA Stepper Pass: Scalar Implementation
// ------------------------------------------------------------------------------
inline void step_tile_circulant_dma_scalar(
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

    // Stage 1: Gather
    #pragma GCC unroll 3
    for (uint32_t d = 0; d < 3; ++d) {
        gather_slice(scratch.face0 + d * 64, TILE_SLICES_W0_F0[tile_idx][d], vnode_base);
        gather_slice(scratch.face2 + d * 64, TILE_SLICES_W0_F2[tile_idx][d], vnode_base);
    }
    #pragma GCC unroll 3
    for (uint32_t d = 0; d < 3; ++d) {
        gather_slice(scratch.face1 + d * 64, TILE_SLICES_W1_F1[tile_idx][d], vnode_base);
        gather_slice(scratch.face3 + d * 64, TILE_SLICES_W1_F3[tile_idx][d], vnode_base);
    }

    // Stage 2: Check Node Math
    if (!passthrough_mode) {
        for (uint32_t cw = 0; cw < 4; ++cw) {
            if (!(active_mask & (1 << cw))) continue;
            for (uint32_t c = 0; c < num_checks; ++c) {
                float q[6];
                float r_in[6];
                uint32_t edge_offset_base = c * deg;

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
                    r_msg_tile[edge_offset_base + d].cw[cw] = fp32_to_bf16(r_out[d]);
                    float l_new = q[d] + r_out[d];
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

    // Stage 3: Scatter
    #pragma GCC unroll 3
    for (uint32_t d = 0; d < 3; ++d) {
        scatter_slice_masked(TILE_SLICES_W0_F0[tile_idx][d], scratch.face0 + d * 64, vnode_base, active_mask);
        scatter_slice_masked(TILE_SLICES_W0_F2[tile_idx][d], scratch.face2 + d * 64, vnode_base, active_mask);
    }
    #pragma GCC unroll 3
    for (uint32_t d = 0; d < 3; ++d) {
        scatter_slice_masked(TILE_SLICES_W1_F1[tile_idx][d], scratch.face1 + d * 64, vnode_base, active_mask);
        scatter_slice_masked(TILE_SLICES_W1_F3[tile_idx][d], scratch.face3 + d * 64, vnode_base, active_mask);
    }
}

// ------------------------------------------------------------------------------
// Single-Tile Circulant DMA Stepper Pass: SFPU Vector Implementation
// ------------------------------------------------------------------------------
inline void step_tile_circulant_dma_sfpu(
    uint32_t tile_idx,
    Packed4CW_BF16* vnodes,      // CB_VNODE (2560 nodes x 8 Bytes)
    uint16_t* r_msg_tile,        // CB_R Check Message FIFO page (1024 words = 2048 Bytes)
    bool is_iter0,
    TileScratchpad& scratch,
    uint32_t active_mask = 0xF,
    bool passthrough_mode = false,
    float damping_factor = 1.0f,
    uint32_t max_steps = 0
) {
    bool is_deg6 = (tile_idx < DEG6_TILES);
    char* vnode_base = reinterpret_cast<char*>(vnodes);

    // Stage 1: Circulant DMA Gather (Worker 0: F0, F2; Worker 1: F1, F3)
    #pragma GCC unroll 3
    for (uint32_t d = 0; d < 3; ++d) {
        gather_slice(scratch.face0 + d * 64, TILE_SLICES_W0_F0[tile_idx][d], vnode_base);
        gather_slice(scratch.face2 + d * 64, TILE_SLICES_W0_F2[tile_idx][d], vnode_base);
    }
    #pragma GCC unroll 3
    for (uint32_t d = 0; d < 3; ++d) {
        gather_slice(scratch.face1 + d * 64, TILE_SLICES_W1_F1[tile_idx][d], vnode_base);
        gather_slice(scratch.face3 + d * 64, TILE_SLICES_W1_F3[tile_idx][d], vnode_base);
    }

    if (!passthrough_mode) {
        // Stage 2: Unpack from Scratchpad & CB_R_MSG into sfpi::dst_reg[0..63]
        const uint16_t* scratch_in = reinterpret_cast<const uint16_t*>(&scratch);
        for (int r = 0; r < 32; ++r) {
            for (uint32_t lane = 0; lane < 32; ++lane) {
                uint32_t word_off = get_scratch_byte_offset(r, lane) / 2;
                sfpi::dst_reg[r][lane] = bf16_to_fp32(scratch_in[word_off]);
                if (is_iter0) {
                    sfpi::dst_reg[32 + r][lane] = 0.0f;
                } else {
                    sfpi::dst_reg[32 + r][lane] = bf16_to_fp32(r_msg_tile[word_off]);
                }
            }
        }

        // Stage 3: Pure SFPU Vector Check Compute on dst_reg (Zero Transposition!)
        if (is_deg6) {
            compute_tile_deg6_sfpi(is_iter0 ? 0 : 1, damping_factor);
        } else {
            compute_tile_deg3_sfpi(is_iter0 ? 0 : 1, damping_factor);
        }

        // Stage 4: Pack from sfpi::dst_reg back to Scratchpad & CB_R_MSG
        uint16_t* scratch_out = reinterpret_cast<uint16_t*>(&scratch);
        for (int r = 0; r < 32; ++r) {
            for (uint32_t lane = 0; lane < 32; ++lane) {
                uint32_t word_off = get_scratch_byte_offset(r, lane) / 2;
                r_msg_tile[word_off]  = fp32_to_bf16(sfpi::dst_reg[r][lane]);       // R_new (Tile 0)
                scratch_out[word_off] = fp32_to_bf16(sfpi::dst_reg[32 + r][lane]); // L_new (Tile 1)
            }
        }
    }

    // Stage 5: Circulant DMA Scatter with Active-Mask Freezing
    #pragma GCC unroll 3
    for (uint32_t d = 0; d < 3; ++d) {
        scatter_slice_masked(TILE_SLICES_W0_F0[tile_idx][d], scratch.face0 + d * 64, vnode_base, active_mask);
        scatter_slice_masked(TILE_SLICES_W0_F2[tile_idx][d], scratch.face2 + d * 64, vnode_base, active_mask);
    }
    #pragma GCC unroll 3
    for (uint32_t d = 0; d < 3; ++d) {
        scatter_slice_masked(TILE_SLICES_W1_F1[tile_idx][d], scratch.face1 + d * 64, vnode_base, active_mask);
        scatter_slice_masked(TILE_SLICES_W1_F3[tile_idx][d], scratch.face3 + d * 64, vnode_base, active_mask);
    }
}

// ------------------------------------------------------------------------------
// PassThrough Verification Mode (Test Gather/Scatter Symmetry with Zero Clashing)
// ------------------------------------------------------------------------------
bool run_passthrough_test(bool use_sfpu = false) {
    std::cout << "================================================================================" << std::endl;
    std::cout << "RUNNING CIRCULANT DMA PASSTHROUGH VERIFICATION (4 CODEWORDS) ["
              << (use_sfpu ? "SFPU VECTOR" : "SCALAR") << "]" << std::endl;
    std::cout << "================================================================================" << std::endl;

    alignas(64) std::vector<Packed4CW_BF16> vnodes(2560);
    alignas(64) std::vector<Packed4CW_BF16> vnodes_orig(2560);
    alignas(64) std::vector<Packed4CW_BF16> r_msg_scalar(TOTAL_TILES * 192);
    alignas(64) std::vector<uint16_t> r_msg_sfpu(TOTAL_TILES * 1024, 0);

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
            if (use_sfpu) {
                step_tile_circulant_dma_sfpu(t, vnodes.data(), &r_msg_sfpu[t * 1024], (iter == 0), scratch, 0xF, true);
            } else {
                step_tile_circulant_dma_scalar(t, vnodes.data(), &r_msg_scalar[t * 192], (iter == 0), scratch, 0xF, true);
            }
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
// Differential Verification: Scalar Stepper vs. SFPU Vector Stepper
// ------------------------------------------------------------------------------
bool run_compare_test() {
    std::cout << "================================================================================" << std::endl;
    std::cout << "DIFFERENTIAL VERIFICATION: SCALAR STEPPER VS. SFPU VECTOR STEPPER" << std::endl;
    std::cout << "================================================================================" << std::endl;

    double rate = 0.5;
    double ebn0_db = 2.0;
    double ebn0_lin = std::pow(10.0, ebn0_db / 10.0);
    double sigma = 1.0 / std::sqrt(2.0 * rate * ebn0_lin);
    double llr_scale = 2.0 / (sigma * sigma);

    std::mt19937_64 rng(42);
    std::normal_distribution<float> dist(0.0f, static_cast<float>(sigma));

    alignas(64) std::vector<Packed4CW_BF16> vnodes_scalar(2560);
    alignas(64) std::vector<Packed4CW_BF16> vnodes_sfpu(2560);
    alignas(64) std::vector<Packed4CW_BF16> r_msg_scalar(TOTAL_TILES * 192);
    alignas(64) std::vector<uint16_t> r_msg_sfpu(TOTAL_TILES * 1024, 0);

    for (uint32_t v = 0; v < 2560; ++v) {
        if (v >= 2048) {
            vnodes_scalar[v].set_all(0);
            vnodes_sfpu[v].set_all(0);
        } else {
            for (uint32_t cw = 0; cw < 4; ++cw) {
                float y = 1.0f + dist(rng);
                float llr = static_cast<float>(llr_scale) * y;
                uint16_t bf = fp32_to_bf16(llr);
                vnodes_scalar[v].cw[cw] = bf;
                vnodes_sfpu[v].cw[cw] = bf;
            }
        }
    }

    TileScratchpad scratch_scalar;
    TileScratchpad scratch_sfpu;

    // Run 1 full circular schedule pass on both
    for (uint32_t t = 0; t < TOTAL_TILES; ++t) {
        step_tile_circulant_dma_scalar(t, vnodes_scalar.data(), &r_msg_scalar[t * 192], true, scratch_scalar, 0xF, false);
        step_tile_circulant_dma_sfpu(t, vnodes_sfpu.data(), &r_msg_sfpu[t * 1024], true, scratch_sfpu, 0xF, false, 1.0f);
    }

    double max_diff = 0.0;
    double sum_diff = 0.0;
    uint32_t exact_matches = 0;
    uint32_t total_elements = 2560 * 4;

    for (uint32_t v = 0; v < 2560; ++v) {
        for (uint32_t cw = 0; cw < 4; ++cw) {
            float val_scal = bf16_to_fp32(vnodes_scalar[v].cw[cw]);
            float val_sfpu = bf16_to_fp32(vnodes_sfpu[v].cw[cw]);
            double d = std::abs(val_scal - val_sfpu);
            if (vnodes_scalar[v].cw[cw] == vnodes_sfpu[v].cw[cw]) {
                exact_matches++;
            }
            if (d > max_diff) max_diff = d;
            sum_diff += d;
        }
    }

    double mean_diff = sum_diff / total_elements;
    std::cout << "  Total Elements Compared: " << total_elements << std::endl;
    std::cout << "  Bit-Exact Matches:       " << exact_matches << " / " << total_elements
              << " (" << std::fixed << std::setprecision(2) << (100.0 * exact_matches / total_elements) << "%)" << std::endl;
    std::cout << "  Max Absolute Difference: " << std::scientific << std::setprecision(4) << max_diff << std::endl;
    std::cout << "  Mean Absolute Difference:" << std::scientific << std::setprecision(4) << mean_diff << std::endl;

    if (max_diff < 0.1) {
        std::cout << "[SUCCESS] SCALAR AND SFPU VECTOR STEPPERS AGREE TO HIGH FIDELITY (bfloat16 precision)!" << std::endl;
        return true;
    } else {
        std::cout << "[WARNING] DIVERGENCE DETECTED (max_diff=" << max_diff << ")!" << std::endl;
        return false;
    }
}

// ------------------------------------------------------------------------------
// Monte Carlo Simulation Engine
// ------------------------------------------------------------------------------
void run_monte_carlo(double ebn0_db, uint64_t max_blocks, uint64_t target_fe, uint32_t max_iter, int num_threads, bool use_sfpu = false, float damping_factor = 1.0f) {
    std::cout << "================================================================================" << std::endl;
    std::cout << "STARTING CIRCULANT 512-BIT DMA MONTE CARLO SIMULATION (4 CODEWORDS)" << std::endl;
    std::cout << "  Engine Mode:    " << (use_sfpu ? "SFPU VECTOR (Zero-Transpose)" : "SCALAR") << std::endl;
    std::cout << "  Eb/N0:          " << std::fixed << std::setprecision(2) << ebn0_db << " dB" << std::endl;
    std::cout << "  Max Blocks:     " << max_blocks << std::endl;
    std::cout << "  Target Errors:  " << target_fe << std::endl;
    std::cout << "  Max Iterations: " << max_iter << std::endl;
    std::cout << "  Damping Factor: " << damping_factor << std::endl;
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
        alignas(64) std::vector<Packed4CW_BF16> r_msg_scalar(TOTAL_TILES * 192);
        alignas(64) std::vector<uint16_t> r_msg_sfpu(TOTAL_TILES * 1024, 0);
        TileScratchpad scratch;

        while (total_fe.load(std::memory_order_relaxed) < target_fe &&
               total_sim_blocks.load(std::memory_order_relaxed) < max_blocks) {

            // Channel Initialization: All-zero codeword (BPSK: +1.0) + AWGN noise
            for (uint32_t v = 0; v < 2560; ++v) {
                if (v >= 2048) {
                    vnodes[v].set_all(0);
                } else {
                    for (uint32_t cw = 0; cw < 4; ++cw) {
                        float y = 1.0f + dist(rng);
                        float llr = static_cast<float>(llr_scale) * y;
                        vnodes[v].cw[cw] = fp32_to_bf16(llr);
                    }
                }
            }

            uint32_t active_mask = 0xF;
            uint32_t cw_iters[4] = {max_iter, max_iter, max_iter, max_iter};

            for (uint32_t iter = 0; iter < max_iter; ++iter) {
                bool is_iter0 = (iter == 0);

                for (uint32_t t = 0; t < TOTAL_TILES; ++t) {
                    if (use_sfpu) {
                        step_tile_circulant_dma_sfpu(t, vnodes.data(), &r_msg_sfpu[t * 1024], is_iter0, scratch, active_mask, false, damping_factor);
                    } else {
                        step_tile_circulant_dma_scalar(t, vnodes.data(), &r_msg_scalar[t * 192], is_iter0, scratch, active_mask, false);
                    }
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
                        active_mask &= ~(1 << cw);
                    }
                }

                if (active_mask == 0) break;
            }

            // Accumulate statistics
            uint32_t batch_fe = 0;
            uint32_t batch_be = 0;
            for (uint32_t cw = 0; cw < 4; ++cw) {
                if (active_mask & (1 << cw)) {
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
    std::cout << "MONTE CARLO SIMULATION RESULTS (CIRCULANT 512-BIT DMA 4-CW) ["
              << (use_sfpu ? "SFPU VECTOR" : "SCALAR") << "]" << std::endl;
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

// ------------------------------------------------------------------------------
// Step Loaded Channel Noise Dump from Core (0,0) Hardware Execution
// ------------------------------------------------------------------------------
bool run_stepper_on_dump(const std::string& dump_path, uint32_t max_iter, bool use_sfpu, float damping_factor = 0.9375f, uint32_t max_steps = 0) {
    std::cout << "================================================================================" << std::endl;
    std::cout << "STEPPING 4-CODEWORD CIRCULANT DMA ENGINE ON LOADED NOISE DUMP" << std::endl;
    std::cout << "  Input File:     " << dump_path << std::endl;
    std::cout << "  Engine Mode:    " << (use_sfpu ? "SFPU VECTOR (Zero-Transpose)" : "SCALAR") << std::endl;
    std::cout << "  Max Iterations: " << max_iter << std::endl;
    std::cout << "  Damping Factor: " << damping_factor << std::endl;
    if (max_steps > 0) {
        std::cout << "  Max Steps:      " << max_steps << " (Tile " << (max_steps % TOTAL_TILES) << ", Iter " << (max_steps / TOTAL_TILES) << ")" << std::endl;
    }
    std::cout << "================================================================================" << std::endl;

    std::ifstream fin(dump_path, std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "[ERROR] Cannot open noise dump file: " << dump_path << std::endl;
        return false;
    }

    alignas(64) std::vector<Packed4CW_BF16> vnodes(2560);
    fin.read(reinterpret_cast<char*>(vnodes.data()), 2560 * sizeof(Packed4CW_BF16));
    fin.close();

    // Check Initial State
    uint32_t init_info_neg[4] = {0, 0, 0, 0};
    uint32_t init_tx_neg[4]   = {0, 0, 0, 0};
    for (uint32_t v = 0; v < 2048; ++v) {
        for (uint32_t cw = 0; cw < 4; ++cw) {
            float val = bf16_to_fp32(vnodes[v].cw[cw]);
            if (val < 0.0f) {
                init_tx_neg[cw]++;
                if (v < 1024) init_info_neg[cw]++;
            }
        }
    }
    std::cout << "[INITIAL CHANNEL STATE]" << std::endl;
    for (uint32_t cw = 0; cw < 4; ++cw) {
        std::cout << "  Codeword " << cw << ": " << init_info_neg[cw] << " info bit errors, "
                  << init_tx_neg[cw] << " total negative symbols (of 2048)" << std::endl;
    }

    alignas(64) std::vector<Packed4CW_BF16> r_msg_scalar(TOTAL_TILES * 192);
    alignas(64) std::vector<uint16_t> r_msg_sfpu(TOTAL_TILES * 1024, 0);
    TileScratchpad scratch;

    uint32_t active_mask = 0xF;
    uint32_t cw_conv_iter[4] = {0, 0, 0, 0};
    uint32_t current_step = 0;

    for (uint32_t iter = 0; iter < max_iter; ++iter) {
        bool is_iter0 = (iter == 0);

        for (uint32_t t = 0; t < TOTAL_TILES; ++t) {
            if (max_steps > 0 && current_step >= max_steps) {
                break;
            }
            if (use_sfpu) {
                step_tile_circulant_dma_sfpu(t, vnodes.data(), &r_msg_sfpu[t * 1024], is_iter0, scratch, active_mask, false, damping_factor, max_steps);
            } else {
                step_tile_circulant_dma_scalar(t, vnodes.data(), &r_msg_scalar[t * 192], is_iter0, scratch, active_mask, false);
            }
            current_step++;
        }
        if (max_steps > 0 && current_step >= max_steps) {
            std::cout << "[MAX STEPS REACHED] Stopped at Step " << current_step << " (Tile " << (current_step % TOTAL_TILES) << ", Iter " << (current_step / TOTAL_TILES) << ")" << std::endl;
            break;
        }

        std::cout << "[Iter " << std::setw(2) << iter + 1 << "] Active Mask: 0x" << std::hex << active_mask << std::dec << " | ";
        for (uint32_t cw = 0; cw < 4; ++cw) {
            if (!(active_mask & (1 << cw))) {
                std::cout << "CW" << cw << ": CONV(" << cw_conv_iter[cw] << ") ";
                continue;
            }

            uint32_t unsat_checks = 0;
            for (uint32_t m = 0; m < 1536; ++m) {
                int syn = 0;
                uint32_t deg = (m < 512) ? 3 : 6;
                for (uint32_t d = 0; d < deg; ++d) {
                    uint32_t v = CHECK_TO_VAR[m][d];
                    if (bf16_to_fp32(vnodes[v].cw[cw]) < 0.0f) syn ^= 1;
                }
                if (syn != 0) unsat_checks++;
            }

            uint32_t info_be = 0;
            for (uint32_t k = 0; k < 1024; ++k) {
                if (bf16_to_fp32(vnodes[k].cw[cw]) < 0.0f) info_be++;
            }

            std::cout << "CW" << cw << ": Unsat=" << unsat_checks << ", BE=" << info_be << " ";

            if (unsat_checks == 0) {
                cw_conv_iter[cw] = iter + 1;
                active_mask &= ~(1 << cw);
            }
        }
        std::cout << std::endl;

        if (active_mask == 0) {
            std::cout << "\n>>> [SUCCESS] All 4 codewords converged cleanly by iteration " << iter + 1 << "! <<<" << std::endl;
            std::ofstream fout("stepper_final_vnodes.bin", std::ios::binary);
            if (fout.is_open()) {
                fout.write(reinterpret_cast<const char*>(vnodes.data()), 2560 * sizeof(Packed4CW_BF16));
                fout.close();
                std::cout << "  [SUCCESS] Saved Stepper Final Vnodes Dump: stepper_final_vnodes.bin (20,480 Bytes)" << std::endl;
            }
            return true;
        }
    }

    std::ofstream fout("stepper_final_vnodes.bin", std::ios::binary);
    if (fout.is_open()) {
        fout.write(reinterpret_cast<const char*>(vnodes.data()), 2560 * sizeof(Packed4CW_BF16));
        fout.close();
        std::cout << "  [SUCCESS] Saved Stepper Final Vnodes Dump: stepper_final_vnodes.bin (20,480 Bytes)" << std::endl;
    }

    std::cout << "\n>>> [FAILURE] Residual Frame Errors Detected (Active Mask: 0x" << std::hex << active_mask << std::dec << ")! <<<" << std::endl;
    for (uint32_t cw = 0; cw < 4; ++cw) {
        if (active_mask & (1 << cw)) {
            uint32_t be = 0;
            for (uint32_t k = 0; k < 1024; ++k) {
                if (bf16_to_fp32(vnodes[k].cw[cw]) < 0.0f) be++;
            }
            std::cout << "  Codeword " << cw << " failed with " << be << " bit errors." << std::endl;
        }
    }
    return false;
}

int main(int argc, char** argv) {
    bool use_sfpu = false;
    uint32_t max_steps = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--sfpi") {
            use_sfpu = true;
        } else if (std::string(argv[i]) == "--max-steps" && i + 1 < argc) {
            max_steps = std::stoul(argv[i + 1]);
        }
    }

    if (argc > 1 && std::string(argv[1]) == "--compare") {
        bool pass = run_compare_test();
        return pass ? 0 : 1;
    }

    if (argc > 1 && (std::string(argv[1]) == "--passthrough" || (argc > 2 && std::string(argv[2]) == "--passthrough"))) {
        bool pass = run_passthrough_test(use_sfpu);
        return pass ? 0 : 1;
    }

    if (argc > 1 && (std::string(argv[1]) == "--load-init" || (argc > 2 && std::string(argv[2]) == "--load-init"))) {
        int base_idx = (std::string(argv[1]) == "--load-init") ? 1 : 2;
        std::string dump_path = (argc > base_idx + 1 && argv[base_idx + 1][0] != '-') ? argv[base_idx + 1] : "tensix_initial_vnodes.bin";
        uint32_t max_iter = (argc > base_idx + 2 && argv[base_idx + 2][0] != '-') ? std::stoul(argv[base_idx + 2]) : 16;
        float damping = (argc > base_idx + 3 && argv[base_idx + 3][0] != '-') ? std::stof(argv[base_idx + 3]) : 0.9375f;
        bool pass = run_stepper_on_dump(dump_path, max_iter, use_sfpu, damping, max_steps);
        return pass ? 0 : 1;
    }

    if (argc > 1 && (std::string(argv[1]) == "--monte-carlo" || (argc > 2 && std::string(argv[2]) == "--monte-carlo"))) {
        int base_idx = (std::string(argv[1]) == "--monte-carlo") ? 1 : 2;
        double ebn0 = (argc > base_idx + 1) ? std::stod(argv[base_idx + 1]) : 2.0;
        uint64_t blocks = (argc > base_idx + 2) ? std::stoull(argv[base_idx + 2]) : 100000;
        uint64_t target_fe = (argc > base_idx + 3) ? std::stoull(argv[base_idx + 3]) : 25;
        uint32_t max_iter = (argc > base_idx + 4) ? std::stoul(argv[base_idx + 4]) : 200;
        int threads = (argc > base_idx + 5) ? std::stoi(argv[base_idx + 5]) : omp_get_max_threads();
        run_monte_carlo(ebn0, blocks, target_fe, max_iter, threads, use_sfpu);
        return 0;
    }

    std::cout << "Usage:" << std::endl;
    std::cout << "  " << argv[0] << " [--sfpi] --passthrough" << std::endl;
    std::cout << "  " << argv[0] << " --compare" << std::endl;
    std::cout << "  " << argv[0] << " [--sfpi] --load-init [tensix_initial_vnodes.bin] [max_iter] [damping] [--max-steps N]" << std::endl;
    std::cout << "  " << argv[0] << " [--sfpi] --monte-carlo [ebn0] [max_blocks] [target_fe] [max_iter] [threads]" << std::endl;
    return 0;
}
