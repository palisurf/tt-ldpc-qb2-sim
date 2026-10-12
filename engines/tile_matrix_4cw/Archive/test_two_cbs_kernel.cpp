#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"
#include "sfpi.h"
#include "llk_math_eltwise_sfpu_common.h"

void kernel_main() {
    constexpr uint32_t cb_in0  = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1  = tt::CBIndex::c_1;
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
    constexpr uint32_t cb_out1 = tt::CBIndex::c_17;

    compute_kernel_hw_startup(cb_in0, cb_out0);
    copy_init(cb_in0);

    cb_wait_front(cb_in0, 1);
    cb_wait_front(cb_in1, 1);

    tile_regs_acquire();
    copy_tile(cb_in0, 0, 0); // copy into Dst slot 0 (10.0f)
    copy_tile(cb_in1, 0, 1); // copy into Dst slot 1 (20.0f)

#ifdef TRISC_MATH
    _llk_math_eltwise_sfpu_start_(0);
    sfpi::vFloat v0 = sfpi::dst_reg[0];
    sfpi::vFloat v1 = sfpi::dst_reg[32];
    sfpi::dst_reg[0]  = v0 + 1.0f; // 10.0 + 1.0 = 11.0f
    sfpi::dst_reg[32] = v1 + 2.0f; // 20.0 + 2.0 = 22.0f
    _llk_math_eltwise_sfpu_done_();
#endif

    tile_regs_commit();
    tile_regs_wait();

    // Pack DST 0 to cb_out0
    cb_reserve_back(cb_out0, 1);
    pack_tile(0, cb_out0, 0);
    cb_push_back(cb_out0, 1);

    // Pack DST 1 (reg 32) to cb_out1
    cb_reserve_back(cb_out1, 1);
    pack_init(cb_out1);
    pack_tile(1, cb_out1, 0);
    cb_push_back(cb_out1, 1);

    cb_pop_front(cb_in0, 1);
    cb_pop_front(cb_in1, 1);
    tile_regs_release();
}
