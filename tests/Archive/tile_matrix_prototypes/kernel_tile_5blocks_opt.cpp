#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"
#include "sfpi.h"
#include "llk_math_eltwise_sfpu_common.h"

// ==============================================================================
// Phase 2C Optimized Full-Tile Kernel: 5 Blocks (40 Checks x 4 Parallel Codewords)
// Target: Native 32x32 Systolic Tile (30 active rows = 960 elements)
// Innovations:
//   1. 100% Branchless Convex-Envelope Spline (sfpi_amin_star_branchless)
//   2. Fused Neutral-Element Substitution (m = 1000.0f) for M_others Accumulation
//   3. Single-Pass Sign & Parity Tracking via vector copysgn
//   4. Fused In-Place Posterior Addition (L_new = Q + R')
//   5. Strict LREG budgeting (<= 4 live vector registers) with Zero Stack Spills
// ==============================================================================

sfpi_inline sfpi::vFloat sfpi_amin_star_branchless(sfpi::vFloat a, sfpi::vFloat b) {
    auto [min_v, max_v] = sfpi::min_max(a, b);
    sfpi::vFloat diff   = max_v - min_v;
    sfpi::vFloat line1  = sfpi::vFloat(sfpi::sFloat16b(0.69315f)) + sfpi::vFloat(sfpi::sFloat16b(-0.4000f)) * diff;
    sfpi::vFloat line2  = sfpi::vFloat(sfpi::sFloat16b(0.65715f)) + sfpi::vFloat(sfpi::sFloat16b(-0.3280f)) * diff;
    sfpi::vFloat corr   = sfpi::max(sfpi::max(line1, line2), 0.0f);
    return sfpi::max(min_v - corr, 0.0f);
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
    #pragma GCC unroll 5
    for (int b = 0; b < 5; ++b) {
        uint32_t tb0 = *wall_clock_reg;
        int base_r = b * 6;

        // 1. Total Sign (via native XOR), Min Mag, Min Idx
        sfpi::vFloat q0 = sfpi::dst_reg[base_r + 0];
        sfpi::vInt total_sign_bits = sfpi::as<sfpi::vInt>(q0);
        sfpi::vFloat min_mag = sfpi::abs(q0);
        sfpi::vFloat min_idx = 0.0f;

        #pragma GCC unroll 5
        for (int r = 1; r < 6; ++r) {
            sfpi::vFloat q = sfpi::dst_reg[base_r + r];
            total_sign_bits ^= sfpi::as<sfpi::vInt>(q);
            sfpi::vFloat m = sfpi::abs(q);
            v_if (m < min_mag) {
                min_mag = m;
                min_idx = static_cast<float>(r);
            } v_endif;
        }

        // 2. M_others accumulation (Branchless AMin* with Neutral-Element Substitution)
        sfpi::vFloat q0_s2 = sfpi::dst_reg[base_r + 0];
        sfpi::vFloat m_others = sfpi::abs(q0_s2);
        v_if (min_idx == 0.0f) {
            m_others = 1000.0f;
        } v_endif;

        #pragma GCC unroll 5
        for (int r = 1; r < 6; ++r) {
            sfpi::vFloat q = sfpi::dst_reg[base_r + r];
            sfpi::vFloat m = sfpi::abs(q);
            v_if (min_idx == static_cast<float>(r)) {
                m = 1000.0f;
            } v_endif;
            m_others = sfpi_amin_star_branchless(m_others, m);
        }

        // 3. M_all (Final branchless AMin* with min_mag)
        sfpi::vFloat m_all = sfpi_amin_star_branchless(m_others, min_mag);

        // 4. In-Place Posterior Update (L_new = Q + R')
        #pragma GCC unroll 6
        for (int r = 0; r < 6; ++r) {
            sfpi::vFloat q = sfpi::dst_reg[base_r + r];
            sfpi::vFloat m_d = m_all;
            v_if (min_idx == static_cast<float>(r)) {
                m_d = m_others;
            } v_endif;
            sfpi::vInt edge_sign_bits = total_sign_bits ^ sfpi::as<sfpi::vInt>(q);
            sfpi::vFloat r_out = sfpi::copysgn(m_d, edge_sign_bits);
            sfpi::dst_reg[base_r + r] = q + r_out;
        }
        block_cycles[b] = *wall_clock_reg - tb0;
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
    stats[6] = c_start - c_init;
    stats[7] = c_done - c_end;
    stats[8] = block_cycles[0];
    stats[9] = block_cycles[1];
    stats[10] = block_cycles[2];
    stats[11] = block_cycles[3];
    stats[12] = block_cycles[4];
    cb_push_back(cb_stats, 1);
}
