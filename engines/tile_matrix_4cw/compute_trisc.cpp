#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"
#include "sfpi.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "ldpc_tile_matrix_core.h"

// ==============================================================================
// Tenstorrent QuietBox 2: Tile Matrix 4-Codeword LDPC Decoder Engine
// Component: Tri-TRISC Compute & Access Coordination Kernel
// Architecture: Decoupled Access-Execute (DAE) with Native 32x32 Tensix Tiles
// Target: Tenstorrent Blackhole (440 Tensix Cores, 2,200 RISC-V Processors)
// ==============================================================================

// ------------------------------------------------------------------------------
// SFPU Vector Fast Natural Logarithm
// ------------------------------------------------------------------------------
sfpi_inline sfpi::vFloat sfpu_fast_ln(sfpi::vFloat a) {
    a = sfpi::max(a, 1e-7f);
    sfpi::vUInt u = sfpi::as<sfpi::vUInt>(a);
    sfpi::vInt exp_val = sfpi::as<sfpi::vInt>((u >> 23) & 0xFF) - 127;
    sfpi::vUInt m_bits = (u & 0x007FFFFF) | 0x3F800000;
    sfpi::vFloat m = sfpi::as<sfpi::vFloat>(m_bits);
    sfpi::vFloat z = m - 1.0f;

    sfpi::vFloat p = sfpi::vFloat(sfpi::sFloat16b(-0.0528f));
    p = p * z + sfpi::vFloat(sfpi::sFloat16b(0.1264f));
    p = p * z - sfpi::vFloat(sfpi::sFloat16b(0.1678f));
    p = p * z + sfpi::vFloat(sfpi::sFloat16b(0.2057f));
    p = p * z - sfpi::vFloat(sfpi::sFloat16b(0.2490f));
    p = p * z + sfpi::vFloat(sfpi::sFloat16b(0.3328f));
    p = p * z - 0.5f;
    p = p * z + 1.0f;
    sfpi::vFloat ln_m = p * z;

    sfpi::vFloat e_flt = sfpi::convert<sfpi::vFloat>(exp_val, sfpi::RoundMode::Nearest);
    return ln_m + e_flt * 0.69314718f;
}

// ------------------------------------------------------------------------------
// SFPU Vector Fast Square Root
// ------------------------------------------------------------------------------
sfpi_inline sfpi::vFloat sfpu_fast_sqrt(sfpi::vFloat val) {
    val = sfpi::max(val, 1e-7f);
    sfpi::vUInt magic = 0x5F375A86U;
    sfpi::vFloat approx = sfpi::as<sfpi::vFloat>(magic - (sfpi::as<sfpi::vUInt>(val) >> 1));
    sfpi::vFloat neg_half_val = val * -0.5f;
    approx = ((approx * approx) * neg_half_val + 1.5f) * approx;
    approx = ((approx * approx) * neg_half_val + 1.5f) * approx;
    return approx * val;
}

// ------------------------------------------------------------------------------
// SFPU Vector Branchless Approximate-Min*
// ------------------------------------------------------------------------------
sfpi_inline sfpi::vFloat sfpi_amin_star_branchless(sfpi::vFloat a, sfpi::vFloat b) {
    auto [min_v, max_v] = sfpi::min_max(a, b);
    sfpi::vFloat diff   = max_v - min_v;
    sfpi::vFloat line1  = sfpi::vFloat(sfpi::sFloat16b(0.69315f)) + sfpi::vFloat(sfpi::sFloat16b(-0.4000f)) * diff;
    sfpi::vFloat line2  = sfpi::vFloat(sfpi::sFloat16b(0.65715f)) + sfpi::vFloat(sfpi::sFloat16b(-0.3280f)) * diff;
    sfpi::vFloat corr   = sfpi::max(sfpi::max(line1, line2), 0.0f);
    return sfpi::max(min_v - corr, 0.0f);
}

// ------------------------------------------------------------------------------
// Hardware Round-to-Nearest-Even/Nearest (1-cycle sfpstochrnd)
// Eliminates bitfield truncation attenuation when committing to 16-bit DEST.
// ------------------------------------------------------------------------------
sfpi_inline sfpi::vFloat16b round_bf16(sfpi::vFloat val) {
    return sfpi::convert<sfpi::vFloat16b>(val, sfpi::RoundMode::Nearest);
}


// ------------------------------------------------------------------------------
// CRITICAL HARDWARE INVARIANT: SFPU Subvector Transpose Register Clobber Warning
// ------------------------------------------------------------------------------
// SFPTRANSP (subvec_transp) implicitly hard-wires L0..L3 and clobbers L4..L7.
// The GCC compiler (riscv-tt-elf-gcc) DOES NOT model this clobber. Therefore:
// 1. NEVER pass an existing live local variable into swap_subvec16 if its value is
//    needed downstream (it will be silently corrupted in-place).
// 2. ALWAYS invoke swap_subvec16 as the first line of a block directly on dst_reg.
// 3. ALWAYS freshly reload unswapped operands from dst_reg AFTER swap_subvec16.
// 4. Staging across multiple transpositions MUST use DEST registers (dst_reg[sc0]).
// See docs/MD_files/SFPU_LREG_HARDWARE_CLOBBER_AND_COMPILER_DIRECTIVE_SOLUTIONS.md
// ------------------------------------------------------------------------------
sfpi_inline sfpi::vInt swap_subvec16(sfpi::vInt vi) {
    sfpi::vInt a = vi, b = vi, c = vi, d = vi;
    sfpi::subvec_transp(a, b, c, d);
    sfpi::subvec_transp(c, d, a, b);
    return c;
}

sfpi_inline sfpi::vFloat swap_subvec16(sfpi::vFloat vf) {
    sfpi::vFloat a = vf, b = vf, c = vf, d = vf;
    sfpi::subvec_transp(a, b, c, d);
    sfpi::subvec_transp(c, d, a, b);
    return c;
}

// ------------------------------------------------------------------------------
// Isolated Scope: 32-Lane Vector Approximate-Min* for Degree-6 Tile (5 Blocks)
// Directly outputs r_out to eliminate subtraction cancellation!
// ------------------------------------------------------------------------------
void __attribute__((noinline)) compute_tile_deg6(uint32_t cb_in, uint32_t cb_out, uint32_t cb_r_msg, uint32_t iter) {
    if (iter == 0) {
        cb_wait_front(cb_in, 1);
        tile_regs_acquire();
        copy_tile(cb_in, 0, 0); // Tile L into dst_reg[0..31]
        cb_pop_front(cb_in, 1);
    } else {
        cb_wait_front(cb_in, 1);
        cb_wait_front(cb_r_msg, 1);
        tile_regs_acquire();
        copy_tile(cb_in, 0, 0);     // Tile L into dst_reg[0..31]
        copy_tile(cb_r_msg, 0, 1);  // Tile R_old from cb_r_msg into dst_reg[32..63]
        cb_pop_front(cb_in, 1);
        cb_pop_front(cb_r_msg, 1);
    }

#ifdef TRISC_MATH
    _llk_math_eltwise_sfpu_start_(0);

#if LDPC_TEST_PASSTHROUGH_MODE
    // In pass-through mode, Tile 0 is already in dst_reg[0..31] from copy_tile(cb_in, 0, 0).
#elif LDPC_TEST_R_DATAFLOW_MODE
    for (int r = 0; r < 32; ++r) {
        sfpi::vFloat tmp = sfpi::dst_reg[r];
        sfpi::dst_reg[32 + r] = tmp;
        sfpi::dst_reg[r]      = 0.0f;
    }
#else
    const int face_bases[4] = {0, 8, 16, 24};

    #pragma GCC unroll 4
    for (int f = 0; f < 4; ++f) {
        int base_r = face_bases[f];

        #pragma GCC unroll 1
        for (int sub = 0; sub < 2; ++sub) {
            int p0 = base_r + 0 + sub;
            int p1 = base_r + 2 + sub;
            int p2 = base_r + 4 + sub;

            int sc0 = base_r + 6;      // Tile 0 scratch 0: holds s_tot
            int sc1 = base_r + 7;      // Tile 0 scratch 1: holds M01
            int sc2 = 32 + base_r + 6; // Tile 1 scratch 0: holds M23
            int sc3 = 32 + base_r + 7; // Tile 1 scratch 1: holds M45

            // Stage 0: Q = L - R_old in-place in Tile 0 (iter > 0)
            if (iter != 0) {
                sfpi::dst_reg[p0] = round_bf16(sfpi::vFloat(sfpi::dst_reg[p0]) - sfpi::vFloat(sfpi::dst_reg[32 + p0]));
                sfpi::dst_reg[p1] = round_bf16(sfpi::vFloat(sfpi::dst_reg[p1]) - sfpi::vFloat(sfpi::dst_reg[32 + p1]));
                sfpi::dst_reg[p2] = round_bf16(sfpi::vFloat(sfpi::dst_reg[p2]) - sfpi::vFloat(sfpi::dst_reg[32 + p2]));
            }

            // Stage 1: Transpose raw Q vectors directly into dst_reg[32 + p*] (reusing Tile 1 as staging).
            // Exactly 3 swaps per sub-quartet (down from 9)!
            // SFPTRANSP operates on L0..L3 and clobbers L4..L7, but ZERO C++ variables are live in LRegs here.
            sfpi::dst_reg[32 + p0] = swap_subvec16(sfpi::vFloat(sfpi::dst_reg[p0]));
            sfpi::dst_reg[32 + p1] = swap_subvec16(sfpi::vFloat(sfpi::dst_reg[p1]));
            sfpi::dst_reg[32 + p2] = swap_subvec16(sfpi::vFloat(sfpi::dst_reg[p2]));

            // Stage 2: 1-Cycle Native Integer XOR sign parity -> stored in dst_reg[sc0]
            {
                sfpi::vInt s0 = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[p0])) ^ sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[32 + p0]));
                sfpi::vInt s1 = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[p1])) ^ sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[32 + p1]));
                sfpi::vInt s2 = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[p2])) ^ sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[32 + p2]));
                sfpi::dst_reg[sc0] = sfpi::as<sfpi::vFloat>((s0 ^ s1) ^ s2); // s_tot in sc0
            }

            // Stage 3: Pairwise AMin* within edge pairs -> stored in sc1 (M01), sc2 (M23), sc3 (M45)
            {
                sfpi::vFloat m0    = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[p0]));
                sfpi::vFloat m0_sw = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[32 + p0]));
                sfpi::dst_reg[sc1] = round_bf16(sfpi_amin_star_branchless(m0, m0_sw)); // M01
            }
            {
                sfpi::vFloat m1    = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[p1]));
                sfpi::vFloat m1_sw = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[32 + p1]));
                sfpi::dst_reg[sc2] = round_bf16(sfpi_amin_star_branchless(m1, m1_sw)); // M23
            }
            {
                sfpi::vFloat m2    = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[p2]));
                sfpi::vFloat m2_sw = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[32 + p2]));
                sfpi::dst_reg[sc3] = round_bf16(sfpi_amin_star_branchless(m2, m2_sw)); // M45
            }

            // Stage 4: Outgoing check messages with signs applied via SFPSETSGN (1 cycle)
            // Edge pair 0 (p0)
            {
                sfpi::vFloat m_oth0 = sfpi_amin_star_branchless(sfpi::vFloat(sfpi::dst_reg[sc2]), sfpi::vFloat(sfpi::dst_reg[sc3]));
                sfpi::vFloat m0_sw  = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[32 + p0]));
                sfpi::vFloat r_mag0 = sfpi_amin_star_branchless(m_oth0, m0_sw);
                sfpi::vInt s_edge0  = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc0])) ^ sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[p0]));
                sfpi::vFloat r_out0 = round_bf16(sfpi::copysgn(r_mag0, s_edge0));
                sfpi::dst_reg[32 + p0] = round_bf16(sfpi::vFloat(sfpi::dst_reg[p0]) + r_out0); // L_new in Tile 1
                sfpi::dst_reg[p0]      = r_out0;                                               // R_new in Tile 0
            }

            // Edge pair 1 (p1)
            {
                sfpi::vFloat m_oth1 = sfpi_amin_star_branchless(sfpi::vFloat(sfpi::dst_reg[sc1]), sfpi::vFloat(sfpi::dst_reg[sc3]));
                sfpi::vFloat m1_sw  = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[32 + p1]));
                sfpi::vFloat r_mag1 = sfpi_amin_star_branchless(m_oth1, m1_sw);
                sfpi::vInt s_edge1  = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc0])) ^ sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[p1]));
                sfpi::vFloat r_out1 = round_bf16(sfpi::copysgn(r_mag1, s_edge1));
                sfpi::dst_reg[32 + p1] = round_bf16(sfpi::vFloat(sfpi::dst_reg[p1]) + r_out1); // L_new in Tile 1
                sfpi::dst_reg[p1]      = r_out1;                                               // R_new in Tile 0
            }

            // Edge pair 2 (p2)
            {
                sfpi::vFloat m_oth2 = sfpi_amin_star_branchless(sfpi::vFloat(sfpi::dst_reg[sc1]), sfpi::vFloat(sfpi::dst_reg[sc2]));
                sfpi::vFloat m2_sw  = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[32 + p2]));
                sfpi::vFloat r_mag2 = sfpi_amin_star_branchless(m_oth2, m2_sw);
                sfpi::vInt s_edge2  = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc0])) ^ sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[p2]));
                sfpi::vFloat r_out2 = round_bf16(sfpi::copysgn(r_mag2, s_edge2));
                sfpi::dst_reg[32 + p2] = round_bf16(sfpi::vFloat(sfpi::dst_reg[p2]) + r_out2); // L_new in Tile 1
                sfpi::dst_reg[p2]      = r_out2;                                               // R_new in Tile 0
            }
        }

        // Zero out unused padding / scratch registers
        sfpi::dst_reg[base_r + 6] = 0.0f;
        sfpi::dst_reg[base_r + 7] = 0.0f;
        sfpi::dst_reg[32 + base_r + 6] = 0.0f;
        sfpi::dst_reg[32 + base_r + 7] = 0.0f;
    }
#endif


    _llk_math_eltwise_sfpu_done_();
#endif







    tile_regs_commit();
    tile_regs_wait();

    // Pack R_new (Dst reg 0) to cb_r_msg
    cb_reserve_back(cb_r_msg, 1);
    pack_init(cb_r_msg);
    pack_tile(0, cb_r_msg, 0);
    cb_push_back(cb_r_msg, 1);

    // Pack L_new to cb_out
    cb_reserve_back(cb_out, 1);
    pack_init(cb_out);
#if LDPC_TEST_PASSTHROUGH_MODE
    pack_tile(0, cb_out, 0);
#else
    pack_tile(1, cb_out, 0);
#endif
    cb_push_back(cb_out, 1);

    tile_regs_release();
}

sfpi_inline sfpi::vInt mux16_int(sfpi::vInt lo, sfpi::vInt hi, sfpi::vInt mask) {
    sfpi::vInt diff = lo ^ hi;
    diff &= mask;
    return lo ^ diff;
}

sfpi_inline sfpi::vFloat mux16_flt(sfpi::vFloat lo, sfpi::vFloat hi, sfpi::vInt mask) {
    sfpi::vInt ilo = sfpi::as<sfpi::vInt>(lo);
    sfpi::vInt ihi = sfpi::as<sfpi::vInt>(hi);
    sfpi::vInt diff = ilo ^ ihi;
    diff &= mask;
    return sfpi::as<sfpi::vFloat>(ilo ^ diff);
}

// ------------------------------------------------------------------------------
// Isolated Scope: 32-Lane Vector Approximate-Min* for Degree-3 Tile (8 Passes)
// Directly outputs r_out to eliminate subtraction cancellation!
// ------------------------------------------------------------------------------
void __attribute__((noinline)) compute_tile_deg3(uint32_t cb_in, uint32_t cb_out, uint32_t cb_r_msg, uint32_t iter) {
    if (iter == 0) {
        cb_wait_front(cb_in, 1);
        tile_regs_acquire();
        copy_tile(cb_in, 0, 0); // Tile L into dst_reg[0..31]
        cb_pop_front(cb_in, 1);
    } else {
        cb_wait_front(cb_in, 1);
        cb_wait_front(cb_r_msg, 1);
        tile_regs_acquire();
        copy_tile(cb_in, 0, 0);     // Tile L into dst_reg[0..31]
        copy_tile(cb_r_msg, 0, 1);  // Tile R_old from cb_r_msg into dst_reg[32..63]
        cb_pop_front(cb_in, 1);
        cb_pop_front(cb_r_msg, 1);
    }

#ifdef TRISC_MATH
    _llk_math_eltwise_sfpu_start_(0);

#if LDPC_TEST_PASSTHROUGH_MODE
    // In pass-through mode, Tile 0 is already in dst_reg[0..31] from copy_tile(cb_in, 0, 0).
#elif LDPC_TEST_R_DATAFLOW_MODE
    const int face_bases[4] = {0, 8, 16, 24};
    #pragma GCC unroll 4
    for (int f = 0; f < 4; ++f) {
        int base = face_bases[f];
        #pragma GCC unroll 3
        for (int k = 0; k < 3; ++k) {
            #pragma GCC unroll 2
            for (int sub = 0; sub < 2; ++sub) {
                int reg = base + 2 * k + sub;
                sfpi::vFloat l_val = sfpi::dst_reg[reg];
                sfpi::vFloat r_old = 0.0f;
                if (iter != 0) {
                    r_old = sfpi::dst_reg[32 + reg];
                }
                sfpi::vFloat q     = l_val - r_old;
                sfpi::vFloat r_new = swap_subvec16(q);
                sfpi::vFloat l_new = q + r_new;
                sfpi::dst_reg[32 + reg] = l_new; // L_new in Tile 1
                sfpi::dst_reg[reg]      = r_new; // R_new in Tile 0
            }
        }
        // Zero out padding registers: regs 6, 7 of this face
        sfpi::dst_reg[base + 6] = 0.0f;
        sfpi::dst_reg[base + 7] = 0.0f;
        sfpi::dst_reg[32 + base + 6] = 0.0f;
        sfpi::dst_reg[32 + base + 7] = 0.0f;
    }
#else
    const int face_bases[4] = {0, 8, 16, 24};

    #pragma GCC unroll 4
    for (int f = 0; f < 4; ++f) {
        int base_r = face_bases[f];

        #pragma GCC unroll 1
        for (int sub = 0; sub < 2; ++sub) {
            int p0 = base_r + 0 + sub;
            int p1 = base_r + 2 + sub;
            int p2 = base_r + 4 + sub;

            int sc0 = base_r + 6;      // Tile 0 scratch 0: holds q0_sw
            int sc1 = base_r + 7;      // Tile 0 scratch 1: holds q1_sw -> q_e2_blk0 -> r_out_e2
            int sc2 = 32 + base_r + 6; // Tile 1 scratch 0: holds q2_sw
            int sc3 = 32 + base_r + 7; // Tile 1 scratch 1: holds q_e0_blk1 -> r_out_e0

            // Stage 0: Q = L - R_old in-place in Tile 0 (iter > 0)
            if (iter != 0) {
                sfpi::dst_reg[p0] = round_bf16(sfpi::vFloat(sfpi::dst_reg[p0]) - sfpi::vFloat(sfpi::dst_reg[32 + p0]));
                sfpi::dst_reg[p1] = round_bf16(sfpi::vFloat(sfpi::dst_reg[p1]) - sfpi::vFloat(sfpi::dst_reg[32 + p1]));
                sfpi::dst_reg[p2] = round_bf16(sfpi::vFloat(sfpi::dst_reg[p2]) - sfpi::vFloat(sfpi::dst_reg[32 + p2]));
            }

            // Stage 1: Transpose raw Q vectors directly into DEST scratch registers.
            // SFPTRANSP implicitly operates on L0..L3 and clobbers L4..L7.
            // Calling directly from dst_reg ensures ZERO C++ variables are live in LRegs!
            sfpi::dst_reg[sc0] = swap_subvec16(sfpi::vFloat(sfpi::dst_reg[p0]));
            sfpi::dst_reg[sc1] = swap_subvec16(sfpi::vFloat(sfpi::dst_reg[p1]));
            sfpi::dst_reg[sc2] = swap_subvec16(sfpi::vFloat(sfpi::dst_reg[p2]));

            // Broadcast E2 of Block 0 into sc1, and E0 of Block 1 into sc3
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

            // Stage 2: Block 0 check node compute (edges E0: p0 lo, E1: p0 hi, E2: sc1)
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

                sfpi::vFloat r_mag0 = sfpi_amin_star_branchless(m0_sw, m_e2);
                sfpi::vFloat r_out0 = round_bf16(sfpi::copysgn(r_mag0, s_tot ^ s0));
                sfpi::dst_reg[32 + p0] = round_bf16(q0 + r_out0); // L_new in Tile 1
                sfpi::dst_reg[p0]      = r_out0;                  // R_new in Tile 0

                sfpi::vFloat r_mag_e2 = sfpi_amin_star_branchless(m0, m0_sw);
                sfpi::vFloat r_out_e2 = round_bf16(sfpi::copysgn(r_mag_e2, s_tot ^ s_e2));
                sfpi::dst_reg[sc1] = r_out_e2;        // Stage r_out_e2 in sc1
            }

            // Stage 3: Block 1 check node compute (edges E0: sc3, E1: p2 lo, E2: p2 hi)
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

                sfpi::vFloat r_mag2 = sfpi_amin_star_branchless(m2_sw, m_e0);
                sfpi::vFloat r_out2 = round_bf16(sfpi::copysgn(r_mag2, s_tot ^ s2));
                sfpi::dst_reg[32 + p2] = round_bf16(q2 + r_out2); // L_new in Tile 1
                sfpi::dst_reg[p2]      = r_out2;                  // R_new in Tile 0

                sfpi::vFloat r_mag_e0 = sfpi_amin_star_branchless(m2, m2_sw);
                sfpi::vFloat r_out_e0 = round_bf16(sfpi::copysgn(r_mag_e0, s_tot ^ s_e0));
                sfpi::dst_reg[sc3] = r_out_e0;        // Stage r_out_e0 in sc3
            }

            // Stage 4: Assemble p1 (lower 16 lanes is r_out_e2, upper 16 lanes is r_out_e0)
            {
                sfpi::vFloat r_out1 = sfpi::dst_reg[sc1];
                v_if (sfpi::as<sfpi::vUInt>(sfpi::vConstTileId) >= 32) {
                    r_out1 = sfpi::vFloat(sfpi::dst_reg[sc3]);
                } v_endif;

                sfpi::dst_reg[32 + p1] = round_bf16(sfpi::vFloat(sfpi::dst_reg[p1]) + r_out1); // L_new in Tile 1
                sfpi::dst_reg[p1]      = r_out1;                                               // R_new in Tile 0
            }
        }

        // Zero padding / scratch registers
        sfpi::dst_reg[base_r + 6] = 0.0f;
        sfpi::dst_reg[base_r + 7] = 0.0f;
        sfpi::dst_reg[32 + base_r + 6] = 0.0f;
        sfpi::dst_reg[32 + base_r + 7] = 0.0f;
    }
#endif

    _llk_math_eltwise_sfpu_done_();
#endif

    tile_regs_commit();
    tile_regs_wait();

    // Pack R_new (Dst reg 0) to cb_r_msg
    cb_reserve_back(cb_r_msg, 1);
    pack_init(cb_r_msg);
    pack_tile(0, cb_r_msg, 0);
    cb_push_back(cb_r_msg, 1);

    // Pack L_new to cb_out
    cb_reserve_back(cb_out, 1);
    pack_init(cb_out);
#if LDPC_TEST_PASSTHROUGH_MODE
    pack_tile(0, cb_out, 0);
#else
    pack_tile(1, cb_out, 0);
#endif
    cb_push_back(cb_out, 1);

    tile_regs_release();
}

// ------------------------------------------------------------------------------
// Vector xoshiro128+ PRNG Generator on SFPU (32 Lanes in Parallel)
// ------------------------------------------------------------------------------
sfpi_inline sfpi::vUInt arx_hash(sfpi::vUInt x, uint32_t c) {
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
    sfpi::vUInt s0;
    sfpi::vUInt s1;
    sfpi::vUInt s2;
    sfpi::vUInt s3;

    sfpi_inline void init(sfpi::vUInt seed_val) {
        s0 = arx_hash(seed_val, 0x9E3779B9U);
        s1 = arx_hash(seed_val, 0x85EBCA6BU);
        s2 = arx_hash(seed_val, 0xC2B2AE35U);
        s3 = arx_hash(seed_val, 0x27D4EB2FU);
        v_if ((s0 | s1 | s2 | s3) == 0) {
            s0 = 0x13579BDFU;
        } v_endif;
    }

    sfpi_inline sfpi::vUInt next() {
        sfpi::vUInt res = s0 + s3;
        sfpi::vUInt t = s1 << 9;
        s2 ^= s0;
        s3 ^= s1;
        s1 ^= s2;
        s0 ^= s3;
        s2 ^= t;
        s3 = (s3 << 11) | (s3 >> 21);
        return res;
    }
};

// ------------------------------------------------------------------------------
// 32-Lane Vector SFPU Channel Observation Generator
// Generates 10 tiles (8 AWGN / Benchmark tiles + 2 Punctured zero tiles)
// Directly streams into CB_VNODE_A in native Tensix face quadrant tile format
// ------------------------------------------------------------------------------
void __attribute__((noinline)) generate_channel_tiles(
    uint32_t cb_out, float mu_val, float sigma_val, bool benchmark_mode, uint32_t invocation_id, uint32_t batch
) {
    pack_init(cb_out);

    for (uint32_t t = 0; t < 10; ++t) {
        tile_regs_acquire();

#ifdef TRISC_MATH
        _llk_math_eltwise_sfpu_start_(0);

#if LDPC_TEST_PASSTHROUGH_MODE
        for (int r = 0; r < 32; ++r) {
            sfpi::dst_reg[r] = 0.0f;
        }
#else
        if (t < 8) {
            if (benchmark_mode) {
                sfpi::vFloat v_mu = mu_val;
                for (int r = 0; r < 32; ++r) {
                    sfpi::dst_reg[r] = v_mu;
                }
            } else {
                // Phase 1: Vector xoshiro128+ Uniform Draws staged directly into DEST
                {
                    VectorXoshiro128Plus rng;
                    uint32_t seed_base = 0x12345678U + (invocation_id * 1013904223U) + batch * 31337U + t * 7919U;
                    sfpi::vUInt lane_seed = static_cast<uint32_t>(seed_base) + sfpi::as<sfpi::vUInt>(sfpi::vConstTileId);
                    rng.init(lane_seed);

                    for (int r = 0; r < 32; ++r) {
                        sfpi::vUInt raw = rng.next();
                        sfpi::vUInt u_bits = ((raw >> 9) & 0x007FFFFF) | 0x3F800000;
                        sfpi::vFloat u = sfpi::as<sfpi::vFloat>(u_bits) - 1.0f;
                        sfpi::dst_reg[r] = sfpi::max(u, 1e-7f);
                    }
                }

                // Phase 2: Transcendental Box-Muller Transformation (PRNG is dead, 0 LReg pressure)
                sfpi::vFloat v_mu = mu_val;
                sfpi::vFloat v_sigma = sigma_val;
                for (int p = 0; p < 16; ++p) {
                    sfpi::vFloat u1 = sfpi::vFloat(sfpi::dst_reg[2 * p + 0]);
                    sfpi::vFloat ln_u1 = sfpu_fast_ln(u1);
                    sfpi::vFloat r = sfpu_fast_sqrt(ln_u1 * -2.0f);

                    sfpi::vFloat u2 = sfpi::vFloat(sfpi::dst_reg[2 * p + 1]);
                    sfpi::vFloat theta = u2 * 6.2831853f - 3.14159265f;
                    sfpi::vFloat s_th = theta * theta;

                    // Cosine series & write Row 2*p + 0
                    sfpi::vFloat cos_val = sfpi::vFloat(sfpi::sFloat16b(0.0000248f));
                    cos_val = cos_val * s_th - sfpi::vFloat(sfpi::sFloat16b(0.0013889f));
                    cos_val = cos_val * s_th + sfpi::vFloat(sfpi::sFloat16b(0.0416667f));
                    cos_val = cos_val * s_th - 0.5f;
                    cos_val = cos_val * s_th + 1.0f;
                    sfpi::dst_reg[2 * p + 0] = v_mu + v_sigma * (r * cos_val);

                    // Sine series & write Row 2*p + 1
                    sfpi::vFloat sin_val = sfpi::vFloat(sfpi::sFloat16b(0.00000276f));
                    sin_val = sin_val * s_th - sfpi::vFloat(sfpi::sFloat16b(0.0001984f));
                    sin_val = sin_val * s_th + sfpi::vFloat(sfpi::sFloat16b(0.0083333f));
                    sin_val = sin_val * s_th - sfpi::vFloat(sfpi::sFloat16b(0.1666667f));
                    sin_val = sin_val * s_th + 1.0f;
                    sfpi::dst_reg[2 * p + 1] = v_mu + v_sigma * (r * (sin_val * theta));
                }
            }
        } else {
            // Stage 2: Generate 2 Punctured Zero Tiles (2 x 1024 = 2,048 Zeros)
            for (int r = 0; r < 32; ++r) {
                sfpi::dst_reg[r] = 0.0f;
            }
        }
#endif

        _llk_math_eltwise_sfpu_done_();
#endif

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out, 0);
        cb_push_back(cb_out, 1);

        tile_regs_release();
    }
}

// Main Tri-TRISC Kernel Execution
// ==============================================================================
void kernel_main() {
    uint32_t num_batches   = get_arg_val<uint32_t>(0);
    uint32_t max_iter      = get_arg_val<uint32_t>(1);
    uint32_t num_tiles     = get_arg_val<uint32_t>(2);
    uint32_t mu_bits       = get_arg_val<uint32_t>(3);
    uint32_t sigma_bits    = get_arg_val<uint32_t>(4);
    uint32_t invocation_id = get_arg_val<uint32_t>(5);
    uint32_t start_batch   = get_arg_val<uint32_t>(6);
    if (invocation_id == 0) invocation_id = 1;
    uint32_t epoch = invocation_id << 16;

    bool benchmark_mode = (sigma_bits == 0);
    union { uint32_t u; float f; } mu_conv, sigma_conv;
    mu_conv.u = mu_bits;
    sigma_conv.u = sigma_bits;
    float mu_val = mu_conv.f;
    float sigma_val = sigma_conv.f;

    constexpr uint32_t cb_vnode       = TileMatrixCB::CB_VNODE_A;
    constexpr uint32_t cb_scratch_in  = TileMatrixCB::CB_SCRATCH_GATHER_0;
    constexpr uint32_t cb_scratch_out = TileMatrixCB::CB_SCRATCH_SCATTER_0;
    constexpr uint32_t cb_r_msg       = TileMatrixCB::CB_R_MSG;
    constexpr uint32_t cb_h_offset    = TileMatrixCB::CB_H_OFFSET;
    constexpr uint32_t cb_telemetry   = TileMatrixCB::CB_TELEMETRY;

    compute_kernel_hw_startup(cb_scratch_in, cb_scratch_out);
    copy_init(cb_scratch_in);

    uint32_t stats_l1 = ckernel::get_tile_address(cb_telemetry, 0);
    volatile uint32_t* sync_flags = reinterpret_cast<volatile uint32_t*>(stats_l1);

    volatile uint32_t* const wall_clock_reg = reinterpret_cast<volatile uint32_t*>(0xFFB121F0);
    uint32_t t_start = *wall_clock_reg;

    uint32_t global_iter = 0;

    // --------------------------------------------------------------------------
    // Main Simulation Loop
    // --------------------------------------------------------------------------
    for (uint32_t batch = 0; batch < num_batches; ++batch) {
        // Step 1: Generate Channel Observations for CB_VNODE_A on 32-Lane SFPU
        generate_channel_tiles(cb_vnode, mu_val, sigma_val, benchmark_mode, invocation_id, start_batch + batch);

        // Step 2: Tri-core batch rendezvous barrier
#ifdef TRISC_PACK
        sync_flags[SyncSlot::BATCH_BARRIER_TRISC] = epoch + batch + 1;
#endif
        while (sync_flags[SyncSlot::BATCH_BARRIER_NCRISC] < epoch + batch + 1) { }
        while (sync_flags[SyncSlot::BATCH_BARRIER_BRISC] < epoch + batch + 1) { }
        while (sync_flags[SyncSlot::BATCH_BARRIER_TRISC] < epoch + batch + 1) { }

        for (uint32_t iter = 0; iter < max_iter; ++iter) {
            global_iter++;
            for (uint32_t t = 0; t < num_tiles; ++t) {
                uint16_t deg = sync_flags[64 + t];

                // Execute Vector Check Node Compute (Approximate-Min* Spline)
                if (deg == 6) {
                    compute_tile_deg6(cb_scratch_in, cb_scratch_out, cb_r_msg, iter);
                } else {
                    compute_tile_deg3(cb_scratch_in, cb_scratch_out, cb_r_msg, iter);
                }
            }

            // Synchronize with NCRISC iteration check for early termination
            while (sync_flags[SyncSlot::ITER_DONE] < epoch + global_iter) { }
            if (sync_flags[SyncSlot::EARLY_TERMINATION]) {
                break;
            }
        }

#ifdef TRISC_UNPACK
        for (uint32_t t = 0; t < num_tiles; ++t) {
            cb_wait_front(cb_r_msg, 1);
            cb_pop_front(cb_r_msg, 1);
        }
#endif
    }

    uint32_t t_end = *wall_clock_reg;
#ifdef TRISC_PACK
    sync_flags[SyncSlot::TRISC_CYCLES] = t_end - t_start;
    sync_flags[SyncSlot::BATCH_BARRIER_TRISC] = epoch + num_batches + 1;
#endif
}
