#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"
#include "sfpi.h"
#include "llk_math_eltwise_sfpu_common.h"

// ==============================================================================
// Phase 2B Sandbox: Degree-6 32-Lane Tile Block Kernel (Physical Blackhole)
// Target: 8 Checks x 4 Parallel Codewords (32 lanes in lockstep)
// True Recursive MILCOM 2003 Formulation (Dr. Christopher R. Jones)
//
// Register Allocation within dst_reg (Tile 0 scratchpad):
//   dst_reg[0..5] : 6 check edges (q_d -> updated l_new)
//   dst_reg[6]    : total_sign
//   dst_reg[7]    : min_mag
//   dst_reg[8]    : min_idx
//   dst_reg[9]    : m_others
//   dst_reg[10]   : m_all
// ==============================================================================

sfpi_inline sfpi::vFloat sfpi_amin_star_binary(sfpi::vFloat a, sfpi::vFloat b) {
    sfpi::vFloat diff  = sfpi::abs(a - b);
    sfpi::vFloat min_v = sfpi::min(a, b);
    sfpi::vFloat corr  = 0.0f;

    v_if (diff < 0.5f) {
        corr = 0.69315f - 0.4000f * diff;
    } v_elseif (diff < 2.0f) {
        corr = 0.49315f - 0.3280f * (diff - 0.5f);
    } v_endif;

    sfpi::vFloat res = sfpi::max(min_v - corr, 0.0f);
    return res;
}

void kernel_main() {
    constexpr uint32_t cb_in    = tt::CBIndex::c_0;
    constexpr uint32_t cb_out   = tt::CBIndex::c_16;
    constexpr uint32_t cb_stats = tt::CBIndex::c_17;

    compute_kernel_hw_startup(cb_in, cb_out);
    copy_init(cb_in);

    volatile uint32_t* const wall_clock_reg = reinterpret_cast<volatile uint32_t*>(0xFFB121F0);
    uint32_t elapsed_cycles = 0;

    tile_regs_acquire();

    cb_wait_front(cb_in, 1);
    copy_tile(cb_in, 0, 0);

#ifdef TRISC_MATH
    _llk_math_eltwise_sfpu_start_(0);

    uint32_t c_start = *wall_clock_reg;

    // --------------------------------------------------------------------------
    // 1. Calculate Total Sign, Find Minimum Magnitude and min_idx
    // --------------------------------------------------------------------------
    {
        sfpi::vFloat total_sign = 1.0f;
        sfpi::vFloat min_mag = 9999.0f;
        sfpi::vFloat min_idx = 0.0f;

        for (int r = 0; r < 6; ++r) {
            sfpi::vFloat q = sfpi::dst_reg[r];
            v_if (q < 0.0f) {
                total_sign = -total_sign;
            } v_endif;
            sfpi::vFloat m = sfpi::abs(q);
            v_if (m < min_mag) {
                min_mag = m;
                min_idx = static_cast<float>(r);
            } v_endif;
        }

        sfpi::dst_reg[6] = total_sign;
        sfpi::dst_reg[7] = min_mag;
        sfpi::dst_reg[8] = min_idx;
    }

    // --------------------------------------------------------------------------
    // 2. True Recursive Approximate-Min* Accumulation for M_others
    // --------------------------------------------------------------------------
    {
        sfpi::vFloat min_idx = sfpi::dst_reg[8];
        sfpi::vFloat m_others = 1000.0f;
        for (int r = 0; r < 6; ++r) {
            sfpi::vFloat q = sfpi::dst_reg[r];
            sfpi::vFloat m = sfpi::abs(q);
            sfpi::vFloat next_m = sfpi_amin_star_binary(m_others, m);
            v_if (min_idx != static_cast<float>(r)) {
                m_others = next_m;
            } v_endif;
        }
        sfpi::dst_reg[9] = m_others;
    }

    // --------------------------------------------------------------------------
    // 3. Final Approximate-Min* for M_all = AMin*(M_others, min_mag)
    // --------------------------------------------------------------------------
    {
        sfpi::vFloat m_others = sfpi::dst_reg[9];
        sfpi::vFloat min_mag  = sfpi::dst_reg[7];
        sfpi::dst_reg[10]     = sfpi_amin_star_binary(m_others, min_mag);
    }

    // --------------------------------------------------------------------------
    // 4. In-Place Posterior LLR Update for Rows 0..5
    // --------------------------------------------------------------------------
    {
        sfpi::vFloat total_sign = sfpi::dst_reg[6];
        sfpi::vFloat min_idx    = sfpi::dst_reg[8];
        sfpi::vFloat m_others   = sfpi::dst_reg[9];
        sfpi::vFloat m_all      = sfpi::dst_reg[10];

        for (int r = 0; r < 6; ++r) {
            sfpi::vFloat q = sfpi::dst_reg[r];
            sfpi::vFloat s = 1.0f;
            v_if (q < 0.0f) {
                s = -1.0f;
            } v_endif;
            sfpi::vFloat m_d = m_all;
            v_if (min_idx == static_cast<float>(r)) {
                m_d = m_others;
            } v_endif;
            sfpi::dst_reg[r] = q + total_sign * s * m_d;
        }
    }

    uint32_t c_end = *wall_clock_reg;
    elapsed_cycles = c_end - c_start;

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
    stats[0] = elapsed_cycles;
    stats[1] = 0xAA55AA55;
    cb_push_back(cb_stats, 1);
}
