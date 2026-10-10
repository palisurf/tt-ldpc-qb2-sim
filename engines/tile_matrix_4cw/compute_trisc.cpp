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

    const int face_bases[4] = {0, 8, 16, 24};

    #pragma GCC unroll 4
    for (int f = 0; f < 4; ++f) {
        int base_r = face_bases[f];

        // If iter > 0: compute Q = L - R_old on SFPU; if iter == 0: Q = L
        if (iter > 0) {
            #pragma GCC unroll 6
            for (int r = 0; r < 6; ++r) {
                sfpi::vFloat l_val = sfpi::dst_reg[base_r + r];
                sfpi::vFloat r_old = sfpi::dst_reg[32 + base_r + r];
                sfpi::dst_reg[base_r + r] = l_val - r_old;
            }
        }

        // 1. Total Sign (via native XOR), Min Mag, Min Idx
        sfpi::vFloat q0 = sfpi::dst_reg[base_r + 0];
        sfpi::vInt total_sign_bits = sfpi::as<sfpi::vInt>(q0);
        sfpi::vFloat min_mag = sfpi::abs(q0);
        sfpi::vInt min_idx = 0;

        #pragma GCC unroll 5
        for (int r = 1; r < 6; ++r) {
            sfpi::vFloat q = sfpi::dst_reg[base_r + r];
            total_sign_bits ^= sfpi::as<sfpi::vInt>(q);
            sfpi::vFloat m = sfpi::abs(q);
            v_if (m < min_mag) {
                min_mag = m;
                min_idx = r;
            } v_endif;
        }

        // 2. M_others accumulation (Neutral-Element Substitution)
        sfpi::vFloat q0_s2 = sfpi::dst_reg[base_r + 0];
        sfpi::vFloat m_others = sfpi::abs(q0_s2);
        v_if (min_idx == 0) {
            m_others = 1000.0f;
        } v_endif;

        #pragma GCC unroll 5
        for (int r = 1; r < 6; ++r) {
            sfpi::vFloat q = sfpi::dst_reg[base_r + r];
            sfpi::vFloat m = sfpi::abs(q);
            v_if (min_idx == r) {
                m = 1000.0f;
            } v_endif;
            m_others = sfpi_amin_star_branchless(m_others, m);
        }

        // 3. M_all (Final branchless AMin* with min_mag)
        sfpi::vFloat m_all = sfpi_amin_star_branchless(m_others, min_mag);

        // 4. Check Message Update and Posterior LLR Update (Fused Step 2B)
        // Store R_new in Tile 0 (dst_reg) and L_new = Q + R_new in Tile 1 (dst_reg 32..)
        #pragma GCC unroll 6
        for (int r = 0; r < 6; ++r) {
            sfpi::vFloat q = sfpi::dst_reg[base_r + r];
            sfpi::vFloat m_d = m_all;
            v_if (min_idx == r) {
                m_d = m_others;
            } v_endif;
            sfpi::vInt edge_sign_bits = total_sign_bits ^ sfpi::as<sfpi::vInt>(q);
            sfpi::vFloat r_out = sfpi::copysgn(m_d, edge_sign_bits);
            sfpi::dst_reg[32 + base_r + r] = q + r_out; // Tile 1: L_new
            sfpi::dst_reg[base_r + r]      = r_out;     // Tile 0: R_new
        }

        // Zero out unused padding registers (regs 6, 7 of this face)
        sfpi::dst_reg[base_r + 6] = 0.0f;
        sfpi::dst_reg[base_r + 7] = 0.0f;
        sfpi::dst_reg[32 + base_r + 6] = 0.0f;
        sfpi::dst_reg[32 + base_r + 7] = 0.0f;
    }

    _llk_math_eltwise_sfpu_done_();
#endif

    tile_regs_commit();
    tile_regs_wait();

    // Pack R_new (Dst reg 0) to cb_r_msg
    cb_reserve_back(cb_r_msg, 1);
    pack_init(cb_r_msg);
    pack_tile(0, cb_r_msg, 0);
    cb_push_back(cb_r_msg, 1);

    // Pack L_new (Dst reg 1) to cb_out
    cb_reserve_back(cb_out, 1);
    pack_init(cb_out);
    pack_tile(1, cb_out, 0);
    cb_push_back(cb_out, 1);

    tile_regs_release();
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

    const int deg3_bases[8] = {0, 3, 8, 11, 16, 19, 24, 27};

    #pragma GCC unroll 8
    for (int b = 0; b < 8; ++b) {
        int base_r = deg3_bases[b];

        // If iter > 0: compute Q = L - R_old on SFPU; if iter == 0: Q = L
        if (iter > 0) {
            #pragma GCC unroll 3
            for (int r = 0; r < 3; ++r) {
                sfpi::vFloat l_val = sfpi::dst_reg[base_r + r];
                sfpi::vFloat r_old = sfpi::dst_reg[32 + base_r + r];
                sfpi::dst_reg[base_r + r] = l_val - r_old;
            }
        }

        // 1. Total Sign, Min Mag, Min Idx
        sfpi::vFloat q0 = sfpi::dst_reg[base_r + 0];
        sfpi::vInt total_sign_bits = sfpi::as<sfpi::vInt>(q0);
        sfpi::vFloat min_mag = sfpi::abs(q0);
        sfpi::vInt min_idx = 0;

        #pragma GCC unroll 2
        for (int r = 1; r < 3; ++r) {
            sfpi::vFloat q = sfpi::dst_reg[base_r + r];
            total_sign_bits ^= sfpi::as<sfpi::vInt>(q);
            sfpi::vFloat m = sfpi::abs(q);
            v_if (m < min_mag) {
                min_mag = m;
                min_idx = r;
            } v_endif;
        }

        // 2. M_others accumulation
        sfpi::vFloat q0_s2 = sfpi::dst_reg[base_r + 0];
        sfpi::vFloat m_others = sfpi::abs(q0_s2);
        v_if (min_idx == 0) {
            m_others = 1000.0f;
        } v_endif;

        #pragma GCC unroll 2
        for (int r = 1; r < 3; ++r) {
            sfpi::vFloat q = sfpi::dst_reg[base_r + r];
            sfpi::vFloat m = sfpi::abs(q);
            v_if (min_idx == r) {
                m = 1000.0f;
            } v_endif;
            m_others = sfpi_amin_star_branchless(m_others, m);
        }

        // 3. M_all
        sfpi::vFloat m_all = sfpi_amin_star_branchless(m_others, min_mag);

        // 4. Check Message Update and Posterior LLR Update (Fused Step 2B)
        // Store R_new in Tile 0 (dst_reg) and L_new = Q + R_new in Tile 1 (dst_reg 32..)
        #pragma GCC unroll 3
        for (int r = 0; r < 3; ++r) {
            sfpi::vFloat q = sfpi::dst_reg[base_r + r];
            sfpi::vFloat m_d = m_all;
            v_if (min_idx == r) {
                m_d = m_others;
            } v_endif;
            sfpi::vInt edge_sign_bits = total_sign_bits ^ sfpi::as<sfpi::vInt>(q);
            sfpi::vFloat r_out = sfpi::copysgn(m_d, edge_sign_bits);
            sfpi::dst_reg[32 + base_r + r] = q + r_out; // Tile 1: L_new
            sfpi::dst_reg[base_r + r]      = r_out;     // Tile 0: R_new
        }
    }

    // Zero out unused padding registers (regs 6, 7 of each face)
    const int pad_bases[4] = {0, 8, 16, 24};
    #pragma GCC unroll 4
    for (int f = 0; f < 4; ++f) {
        int pr = pad_bases[f];
        sfpi::dst_reg[pr + 6] = 0.0f;
        sfpi::dst_reg[pr + 7] = 0.0f;
        sfpi::dst_reg[32 + pr + 6] = 0.0f;
        sfpi::dst_reg[32 + pr + 7] = 0.0f;
    }

    _llk_math_eltwise_sfpu_done_();
#endif

    tile_regs_commit();
    tile_regs_wait();

    // Pack R_new (Dst reg 0) to cb_r_msg
    cb_reserve_back(cb_r_msg, 1);
    pack_init(cb_r_msg);
    pack_tile(0, cb_r_msg, 0);
    cb_push_back(cb_r_msg, 1);

    // Pack L_new (Dst reg 1) to cb_out
    cb_reserve_back(cb_out, 1);
    pack_init(cb_out);
    pack_tile(1, cb_out, 0);
    cb_push_back(cb_out, 1);

    tile_regs_release();
}

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

        if (t < 8) {
            if (benchmark_mode) {
                sfpi::vFloat v_mu = mu_val;
                for (int r = 0; r < 32; ++r) {
                    sfpi::dst_reg[r] = v_mu;
                }
            } else {
                uint32_t seed_base = 0x12345678U + (invocation_id * 1013904223U) + batch * 31337U + t * 7919U;
                sfpi::vUInt prng_state = static_cast<uint32_t>(seed_base) + sfpi::as<sfpi::vUInt>(sfpi::vConstTileId);
                prng_state = (prng_state << 5) ^ sfpi::as<sfpi::vUInt>(sfpi::vConstTileId) ^ 0x55AA55AAU;
                v_if (prng_state == 0) {
                    prng_state = 0x13579BDFU;
                } v_endif;

                sfpi::vFloat v_mu = mu_val;
                sfpi::vFloat v_sigma = sigma_val;
                for (int p = 0; p < 16; ++p) {
                    // Draw U1 in [1e-7, 1.0)
                    prng_state ^= prng_state << 13;
                    prng_state ^= prng_state >> 17;
                    prng_state ^= prng_state << 5;
                    sfpi::vUInt u1_bits = ((prng_state >> 9) & 0x007FFFFF) | 0x3F800000;
                    sfpi::vFloat u1 = sfpi::as<sfpi::vFloat>(u1_bits) - 1.0f;
                    u1 = sfpi::max(u1, 1e-7f);

                    // Compute Radius R = sqrt(-2 * ln(U1))
                    sfpi::vFloat ln_u1 = sfpu_fast_ln(u1);
                    sfpi::vFloat r = sfpu_fast_sqrt(ln_u1 * -2.0f);

                    // Draw U2 in [0.0, 1.0)
                    prng_state ^= prng_state << 13;
                    prng_state ^= prng_state >> 17;
                    prng_state ^= prng_state << 5;
                    sfpi::vUInt u2_bits = ((prng_state >> 9) & 0x007FFFFF) | 0x3F800000;
                    sfpi::vFloat u2 = sfpi::as<sfpi::vFloat>(u2_bits) - 1.0f;

                    // Compute theta in [-PI, PI)
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
        generate_channel_tiles(cb_vnode, mu_val, sigma_val, benchmark_mode, invocation_id, batch);

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
    sync_flags[SyncSlot::TRISC_CYCLES] = t_end - t_start;
#ifdef TRISC_PACK
    sync_flags[SyncSlot::BATCH_BARRIER_TRISC] = epoch + num_batches + 1;
#endif
}
