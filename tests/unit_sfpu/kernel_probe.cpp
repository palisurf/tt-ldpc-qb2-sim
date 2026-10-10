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
    constexpr uint32_t cb_stats = tt::CBIndex::c_17;

    compute_kernel_hw_startup(cb_in, cb_out);
    copy_init(cb_in);

    tile_regs_acquire();
    cb_wait_front(cb_in, 1);
    copy_tile(cb_in, 0, 0);

#ifdef TRISC_MATH
    _llk_math_eltwise_sfpu_start_(0);

    for (int r = 0; r < 32; ++r) {
        sfpi::dst_reg[r] = sfpi::vFloat(static_cast<float>(r + 1));
    }

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
    cb_push_back(cb_stats, 1);
}
