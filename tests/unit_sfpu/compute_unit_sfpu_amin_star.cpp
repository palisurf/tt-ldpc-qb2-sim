#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"
#include "sfpi.h"
#include "llk_math_eltwise_sfpu_common.h"

// ==============================================================================
// TRISC SFPU Unit Test Kernel: Approximate-Min* Verification
// Tests:
//   1. Binary sfpi_amin_star_branchless(a, b) across 32 lanes
//   2. Degree-6 Check Node computation (exact formulation from compute_trisc.cpp)
//   3. Degree-3 Check Node computation (exact formulation from compute_trisc.cpp)
// ==============================================================================

#if defined(LDPC_USE_SFPU_LUT)
sfpi_inline sfpi::vFloat sfpi_amin_star_lut(sfpi::vFloat a, sfpi::vFloat b) {
    auto [min_v, max_v] = sfpi::min_max(a, b);
    sfpi::vFloat diff   = max_v - min_v;
    sfpi::vLut16ss s01(-0.4000f, -0.3280f);
    sfpi::vLut16ii i01(0.69315f, 0.65715f);
    sfpi::vLut16ss s23(-0.3280f, -0.3280f);
    sfpi::vLut16ii i23(0.65715f, 0.65715f);
    sfpi::vLut16ss s45(0.0f, 0.0f);
    sfpi::vLut16ii i45(0.0f, 0.0f);

    sfpi::vFloat corr = sfpi::lut<sfpi::LutMode::Fp16x6_HWM3>(
        diff, s01, i01, s23, i23, s45, i45, sfpi::LutSign::Update);
    return sfpi::max(min_v - corr, 0.0f);
}
#define sfpi_amin_star sfpi_amin_star_lut
#else
sfpi_inline sfpi::vFloat sfpi_amin_star_branchless(sfpi::vFloat a, sfpi::vFloat b) {
    auto [min_v, max_v] = sfpi::min_max(a, b);
    sfpi::vFloat diff   = max_v - min_v;
    sfpi::vFloat line1  = sfpi::vFloat(sfpi::sFloat16b(0.69315f)) + sfpi::vFloat(sfpi::sFloat16b(-0.4000f)) * diff;
    sfpi::vFloat line2  = sfpi::vFloat(sfpi::sFloat16b(0.65715f)) + sfpi::vFloat(sfpi::sFloat16b(-0.3280f)) * diff;
    sfpi::vFloat corr   = sfpi::max(sfpi::max(line1, line2), 0.0f);
    return sfpi::max(min_v - corr, 0.0f);
}
#define sfpi_amin_star sfpi_amin_star_branchless
#endif

void kernel_main() {
    constexpr uint32_t cb_in    = tt::CBIndex::c_0;
    constexpr uint32_t cb_out   = tt::CBIndex::c_16;
    constexpr uint32_t cb_stats = tt::CBIndex::c_17;

    compute_kernel_hw_startup(cb_in, cb_out);
    copy_init(cb_in);

    volatile uint32_t* const wall_clock_reg = reinterpret_cast<volatile uint32_t*>(0xFFB121F0);
    uint32_t c_start = 0, c_end = 0;

    tile_regs_acquire();

    cb_wait_front(cb_in, 1);
    copy_tile(cb_in, 0, 0);

#ifdef TRISC_MATH
    _llk_math_eltwise_sfpu_start_(0);
    c_start = *wall_clock_reg;

    // --------------------------------------------------------------------------
    // Test 1: Direct Binary Approximate-Min* across 32 lanes
    // Inputs: dst_reg[0] = a, dst_reg[1] = b
    // Output: dst_reg[0] = AMin*(a, b)
    // --------------------------------------------------------------------------
    {
        sfpi::vFloat a = sfpi::dst_reg[0];
        sfpi::vFloat b = sfpi::dst_reg[1];
        sfpi::dst_reg[0] = sfpi_amin_star(a, b);
    }

    // --------------------------------------------------------------------------
    // Test 2: Degree-6 Check Node SFPU processing across 32 lanes
    // Inputs: dst_reg[2..7] = q[0..5]
    // Outputs: dst_reg[2..7] = r_out[0..5], dst_reg[16..21] = L_out[0..5]
    // --------------------------------------------------------------------------
    {
        // 1. Total sign XOR, min mag, min idx
        sfpi::vFloat q0 = sfpi::dst_reg[2];
        sfpi::vInt total_sign_bits = sfpi::as<sfpi::vInt>(q0);
        sfpi::vFloat min_mag = sfpi::abs(q0);
        sfpi::vInt min_idx = 0;

        #pragma GCC unroll 5
        for (int r = 1; r < 6; ++r) {
            sfpi::vFloat q = sfpi::dst_reg[2 + r];
            total_sign_bits ^= sfpi::as<sfpi::vInt>(q);
            sfpi::vFloat m = sfpi::abs(q);
            v_if (m < min_mag) {
                min_mag = m;
                min_idx = r;
            } v_endif;
        }

        // 2. M_others accumulation (Neutral-Element Substitution: 1000.0f)
        sfpi::vFloat q0_s2 = sfpi::dst_reg[2];
        sfpi::vFloat m_others = sfpi::abs(q0_s2);
        v_if (min_idx == 0) {
            m_others = 1000.0f;
        } v_endif;

        #pragma GCC unroll 5
        for (int r = 1; r < 6; ++r) {
            sfpi::vFloat q = sfpi::dst_reg[2 + r];
            sfpi::vFloat m = sfpi::abs(q);
            v_if (min_idx == r) {
                m = 1000.0f;
            } v_endif;
            m_others = sfpi_amin_star(m_others, m);
        }

        // 3. M_all
        sfpi::vFloat m_all = sfpi_amin_star(m_others, min_mag);

        // 4. Updates: r_out and L_out = q + r_out
        #pragma GCC unroll 6
        for (int r = 0; r < 6; ++r) {
            sfpi::vFloat q = sfpi::dst_reg[2 + r];
            sfpi::vFloat m_d = m_all;
            v_if (min_idx == r) {
                m_d = m_others;
            } v_endif;
            sfpi::vInt edge_sign_bits = total_sign_bits ^ sfpi::as<sfpi::vInt>(q);
            sfpi::vFloat r_out = sfpi::copysgn(m_d, edge_sign_bits);
            sfpi::dst_reg[16 + r] = q + r_out; // Posterior L_out
            sfpi::dst_reg[2 + r]  = r_out;     // Extrinsic check message r_out
        }
    }

    // --------------------------------------------------------------------------
    // Test 3: Degree-3 Check Node SFPU processing across 32 lanes
    // Inputs: dst_reg[8..10] = q[0..2]
    // Outputs: dst_reg[8..10] = r_out[0..2], dst_reg[24..26] = L_out[0..2]
    // --------------------------------------------------------------------------
    {
        sfpi::vFloat q0 = sfpi::dst_reg[8];
        sfpi::vInt total_sign_bits = sfpi::as<sfpi::vInt>(q0);
        sfpi::vFloat min_mag = sfpi::abs(q0);
        sfpi::vInt min_idx = 0;

        #pragma GCC unroll 2
        for (int r = 1; r < 3; ++r) {
            sfpi::vFloat q = sfpi::dst_reg[8 + r];
            total_sign_bits ^= sfpi::as<sfpi::vInt>(q);
            sfpi::vFloat m = sfpi::abs(q);
            v_if (m < min_mag) {
                min_mag = m;
                min_idx = r;
            } v_endif;
        }

        sfpi::vFloat q0_s2 = sfpi::dst_reg[8];
        sfpi::vFloat m_others = sfpi::abs(q0_s2);
        v_if (min_idx == 0) {
            m_others = 1000.0f;
        } v_endif;

        #pragma GCC unroll 2
        for (int r = 1; r < 3; ++r) {
            sfpi::vFloat q = sfpi::dst_reg[8 + r];
            sfpi::vFloat m = sfpi::abs(q);
            v_if (min_idx == r) {
                m = 1000.0f;
            } v_endif;
            m_others = sfpi_amin_star(m_others, m);
        }

        sfpi::vFloat m_all = sfpi_amin_star(m_others, min_mag);

        #pragma GCC unroll 3
        for (int r = 0; r < 3; ++r) {
            sfpi::vFloat q = sfpi::dst_reg[8 + r];
            sfpi::vFloat m_d = m_all;
            v_if (min_idx == r) {
                m_d = m_others;
            } v_endif;
            sfpi::vInt edge_sign_bits = total_sign_bits ^ sfpi::as<sfpi::vInt>(q);
            sfpi::vFloat r_out = sfpi::copysgn(m_d, edge_sign_bits);
            sfpi::dst_reg[24 + r] = q + r_out;
            sfpi::dst_reg[8 + r]  = r_out;
        }
    }

    c_end = *wall_clock_reg;
    _llk_math_eltwise_sfpu_done_();
#endif

    tile_regs_commit();
    tile_regs_wait();

    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);
    cb_pop_front(cb_in, 1);
    tile_regs_release();
    cb_push_back(cb_out, 1);

    cb_reserve_back(cb_stats, 1);
    uint32_t stats_addr = get_tile_address(cb_stats, 0);
    auto* stats = reinterpret_cast<volatile uint32_t*>(stats_addr);
    stats[0] = c_end - c_start;
    stats[1] = 0xAA55AA55;
    cb_push_back(cb_stats, 1);
}
