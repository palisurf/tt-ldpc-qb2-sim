#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"
#include "sfpi.h"
#include "llk_math_eltwise_sfpu_common.h"

// ==============================================================================
// Phase 2C Full-Tile Kernel: 5 Blocks (40 Checks x 4 Parallel Codewords)
// Target: Native 32x32 Systolic Tile (30 active rows = 960 elements)
// True Recursive MILCOM 2003 Formulation (Dr. Christopher R. Jones)
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
    uint32_t c_init = 0, c_start = 0, c_end = 0, c_done = 0;
    uint32_t sum_step1 = 0, sum_step2 = 0, sum_step3 = 0, sum_step4 = 0;
    uint32_t block_cycles[5] = {0, 0, 0, 0, 0};

    tile_regs_acquire();

    cb_wait_front(cb_in, 1);
    copy_tile(cb_in, 0, 0);

#ifdef TRISC_MATH
    c_init = *wall_clock_reg;
    _llk_math_eltwise_sfpu_start_(0);
    c_start = *wall_clock_reg;

    // Loop over all 5 blocks (40 check nodes * 4 codewords = 160 check nodes)
    for (int b = 0; b < 5; ++b) {
        uint32_t tb0 = *wall_clock_reg;
        int base_r = b * 6;

        // 1. Total Sign, Min Mag, Min Idx
        uint32_t t1_start = *wall_clock_reg;
        {
            sfpi::vFloat total_sign = 1.0f;
            sfpi::vFloat min_mag    = 9999.0f;
            sfpi::vFloat min_idx    = 0.0f;

            for (int r = 0; r < 6; ++r) {
                sfpi::vFloat q = sfpi::dst_reg[base_r + r];
                total_sign = total_sign * sfpi::copysgn(sfpi::vFloat(1.0f), q);
                sfpi::vFloat m = sfpi::abs(q);
                v_if (m < min_mag) {
                    min_mag = m;
                    min_idx = static_cast<float>(r);
                } v_endif;
            }

            sfpi::dst_reg[30] = sfpi::copysgn(min_mag, total_sign);
            sfpi::dst_reg[31] = min_idx;
        }
        uint32_t t1_end = *wall_clock_reg;

        // 2. M_others accumulation
        sfpi::vFloat m_others = 1000.0f;
        uint32_t t2_start = *wall_clock_reg;
        {
            sfpi::vFloat min_idx_v = sfpi::dst_reg[31];
            for (int r = 0; r < 6; ++r) {
                sfpi::vFloat q = sfpi::dst_reg[base_r + r];
                sfpi::vFloat m = sfpi::abs(q);
                sfpi::vFloat next_m = sfpi_amin_star_binary(m_others, m);
                v_if (min_idx_v != static_cast<float>(r)) {
                    m_others = next_m;
                } v_endif;
            }
        }
        uint32_t t2_end = *wall_clock_reg;

        // 3. M_all
        uint32_t t3_start = *wall_clock_reg;
        sfpi::vFloat signed_min_mag = sfpi::dst_reg[30];
        sfpi::vFloat m_all = sfpi_amin_star_binary(m_others, sfpi::abs(signed_min_mag));
        uint32_t t3_end = *wall_clock_reg;

        // 4. In-Place Posterior Update
        uint32_t t4_start = *wall_clock_reg;
        {
            sfpi::vFloat total_sign_v   = sfpi::copysgn(sfpi::vFloat(1.0f), signed_min_mag);
            sfpi::vFloat min_idx_v      = sfpi::dst_reg[31];

            for (int r = 0; r < 6; ++r) {
                sfpi::vFloat q = sfpi::dst_reg[base_r + r];
                sfpi::vFloat s = sfpi::copysgn(sfpi::vFloat(1.0f), q);
                sfpi::vFloat m_d = m_all;
                v_if (min_idx_v == static_cast<float>(r)) {
                    m_d = m_others;
                } v_endif;
                sfpi::dst_reg[base_r + r] = q + total_sign_v * s * m_d;
            }
        }
        uint32_t t4_end = *wall_clock_reg;

        sum_step1 += (t1_end - t1_start);
        sum_step2 += (t2_end - t2_start);
        sum_step3 += (t3_end - t3_start);
        sum_step4 += (t4_end - t4_start);
        block_cycles[b] = (t4_end - tb0);
    }

    c_end = *wall_clock_reg;
    elapsed_cycles = c_end - c_start;
    _llk_math_eltwise_sfpu_done_();
    c_done = *wall_clock_reg;
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
    stats[2] = sum_step1;
    stats[3] = sum_step2;
    stats[4] = sum_step3;
    stats[5] = sum_step4;
    stats[6] = c_start - c_init; // sfpu_start overhead
    stats[7] = c_done - c_end;   // sfpu_done overhead
    stats[8] = block_cycles[0];
    stats[9] = block_cycles[1];
    stats[10] = block_cycles[2];
    stats[11] = block_cycles[3];
    stats[12] = block_cycles[4];
    cb_push_back(cb_stats, 1);
}
