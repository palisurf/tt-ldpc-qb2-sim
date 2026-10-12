#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"
#include "sfpi.h"
#include "llk_math_eltwise_sfpu_common.h"

void kernel_main() {
    constexpr uint32_t cb_in    = tt::CBIndex::c_0;
    constexpr uint32_t cb_out   = tt::CBIndex::c_16;
    constexpr uint32_t cb_r_msg = tt::CBIndex::c_17;

    compute_kernel_hw_startup(cb_in, cb_out);
    copy_init(cb_in);

    // Iteration 0: cb_in has tile (10.0f).
    // Push R_new = 5.0f into cb_r_msg.
    cb_wait_front(cb_in, 1);
    tile_regs_acquire();
    copy_tile(cb_in, 0, 0); // Dst 0 = 10.0f

#ifdef TRISC_MATH
    _llk_math_eltwise_sfpu_start_(0);
    sfpi::dst_reg[32] = 5.0f; // R_new = 5.0f
    _llk_math_eltwise_sfpu_done_();
#endif
    tile_regs_commit();
    tile_regs_wait();

    // Pack R_new to cb_r_msg
    cb_reserve_back(cb_r_msg, 1);
    pack_init(cb_r_msg);
    pack_tile(1, cb_r_msg, 0); // reg 32
    cb_push_back(cb_r_msg, 1);

    tile_regs_release();

    // -------------------------------------------------------------
    // Iteration 1: Unpack R_old from cb_r_msg (was 5.0f)
    // Compute Dst 0 = 10.0f + 5.0f = 15.0f
    // Pack Dst 0 to cb_out
    // -------------------------------------------------------------
    cb_wait_front(cb_r_msg, 1);
    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);       // Dst 0 = 10.0f
    copy_tile(cb_r_msg, 0, 1);    // Dst 1 (reg 32) = 5.0f from cb_r_msg!
    cb_pop_front(cb_r_msg, 1);

#ifdef TRISC_MATH
    _llk_math_eltwise_sfpu_start_(0);
    sfpi::vFloat v0 = sfpi::dst_reg[0];
    sfpi::vFloat vr = sfpi::dst_reg[32];
    sfpi::dst_reg[0] = v0 + vr; // 10.0 + 5.0 = 15.0f!
    _llk_math_eltwise_sfpu_done_();
#endif
    tile_regs_commit();
    tile_regs_wait();

    // Pack result to cb_out
    cb_reserve_back(cb_out, 1);
    pack_init(cb_out);
    pack_tile(0, cb_out, 0);
    cb_push_back(cb_out, 1);

    cb_pop_front(cb_in, 1);
    tile_regs_release();
}
