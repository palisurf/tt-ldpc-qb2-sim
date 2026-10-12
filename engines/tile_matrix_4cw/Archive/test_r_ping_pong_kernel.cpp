#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"
#include "sfpi.h"
#include "llk_math_eltwise_sfpu_common.h"

void kernel_main() {
    constexpr uint32_t cb_in   = tt::CBIndex::c_0;
    constexpr uint32_t cb_r_in = tt::CBIndex::c_2;   // Unpack reads Buffer B
    constexpr uint32_t cb_r_out= tt::CBIndex::c_18;  // Pack writes Buffer B
    constexpr uint32_t cb_out  = tt::CBIndex::c_16;  // Pack writes to output

    compute_kernel_hw_startup(cb_in, cb_out);
    copy_init(cb_in);

    // -------------------------------------------------------------
    // Iteration 0:
    // cb_in has input (10.0f).
    // Push R_new = 5.0f into Buffer B (cb_r_out = c_18).
    // -------------------------------------------------------------
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

    // Pack R_new to Buffer B
    cb_reserve_back(cb_r_out, 1);
    pack_init(cb_r_out);
    pack_tile(1, cb_r_out, 0); // reg 32
    cb_push_back(cb_r_out, 1);

    tile_regs_release();

    // -------------------------------------------------------------
    // Iteration 1:
    // Unpack R_old from Buffer B (cb_r_in = c_2)
    // Dst 0 = 10.0f + 5.0f = 15.0f
    // Pack Dst 0 to cb_out (c_16)
    // -------------------------------------------------------------
    cb_wait_front(cb_r_in, 1);
    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);       // Dst 0 = 10.0f
    copy_tile(cb_r_in, 0, 1);     // Dst 1 (reg 32) = 5.0f from cb_r_in!
    cb_pop_front(cb_r_in, 1);

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
