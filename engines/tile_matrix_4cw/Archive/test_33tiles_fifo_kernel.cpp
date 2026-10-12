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

    constexpr uint32_t NUM_TILES   = 33;
    constexpr uint32_t NUM_ITERS   = 3;
    constexpr uint32_t NUM_BATCHES = 2;

    for (uint32_t batch = 0; batch < NUM_BATCHES; ++batch) {
        if (batch > 0) {
            for (uint32_t t = 0; t < NUM_TILES; ++t) {
                cb_pop_front(cb_r_msg, 1);
            }
        }

        for (uint32_t iter = 0; iter < NUM_ITERS; ++iter) {
        for (uint32_t t = 0; t < NUM_TILES; ++t) {
            if (iter == 0) {
                // Iteration 0: Generate initial R_new = (t + 1) * 1.0f and push to cb_r_msg
                tile_regs_acquire();
#ifdef TRISC_MATH
                _llk_math_eltwise_sfpu_start_(0);
                sfpi::dst_reg[32] = static_cast<float>(t + 1);
                _llk_math_eltwise_sfpu_done_();
#endif
                tile_regs_commit();
                tile_regs_wait();

                cb_reserve_back(cb_r_msg, 1);
                pack_init(cb_r_msg);
                pack_tile(1, cb_r_msg, 0); // reg 32
                cb_push_back(cb_r_msg, 1);

                tile_regs_release();
            } else {
                // Iteration > 0: Pop old R (was R_old), double it, and push back to cb_r_msg
                cb_wait_front(cb_r_msg, 1);
                tile_regs_acquire();
                copy_tile(cb_r_msg, 0, 1); // Dst 1 (reg 32)
                cb_pop_front(cb_r_msg, 1);

#ifdef TRISC_MATH
                _llk_math_eltwise_sfpu_start_(0);
                sfpi::vFloat v = sfpi::dst_reg[32];
                sfpi::dst_reg[32] = v * 2.0f;
                _llk_math_eltwise_sfpu_done_();
#endif
                tile_regs_commit();
                tile_regs_wait();

                // Push updated R back to FIFO
                cb_reserve_back(cb_r_msg, 1);
                pack_init(cb_r_msg);
                pack_tile(1, cb_r_msg, 0);
                cb_push_back(cb_r_msg, 1);

                // On final iteration, if this is tile 32 (last tile), pack to output
                if (iter == NUM_ITERS - 1 && t == NUM_TILES - 1) {
                    cb_reserve_back(cb_out, 1);
                    pack_init(cb_out);
                    pack_tile(1, cb_out, 0);
                    cb_push_back(cb_out, 1);
                }

                tile_regs_release();
            }
        }
    }
    }
}
