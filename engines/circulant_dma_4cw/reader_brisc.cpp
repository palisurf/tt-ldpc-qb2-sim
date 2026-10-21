#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/waypoint.h"
#include "ldpc_circulant_dma_core.h"

// ==============================================================================
// Tenstorrent QuietBox 2: 4-Codeword Circulant DMA LDPC Decoder Engine
// Component: Data Movement Kernel 0 (BRISC - Worker 0: Dual Gather & Dual Scatter)
// Architecture: Centralized TRISC1 Maestro with Bounded Satellite Autonomy (DAE)
// Target: Pure 4-Codeword Engine (Single Batch)
//
// Key Responsibilities:
// 1. Contiguous 512-bit DMA Gather & Scatter of Face 0 and Face 2 via compile-time
//    analytical descriptors TILE_SLICES_W0_F0 and TILE_SLICES_W0_F2.
// 2. Information Node Parity Scan: On Tile 39, scans sign bits across 1,024 info
//    bits for CW0..3 and reports the 4-bit error mask to SyncSlot::PARITY_REPORT_W0.
// 3. Autonomous Dual-Worker Synchronization: Only synchronizes with Worker 1 (NCRISC).
// ==============================================================================

static inline uint32_t get_wall_clock() {
    volatile uint32_t* const wall_clock_reg = reinterpret_cast<volatile uint32_t*>(0xFFB121F0);
    return *wall_clock_reg;
}

void kernel_main() {
    uint32_t num_batches   = get_arg_val<uint32_t>(0); // Number of 4-codeword batches
    uint32_t max_iter      = get_arg_val<uint32_t>(1);
    uint32_t num_tiles     = get_arg_val<uint32_t>(2);
    uint32_t invocation_id = get_arg_val<uint32_t>(3);
    uint32_t core_id       = get_arg_val<uint32_t>(4);
    (void)core_id;

    if (invocation_id == 0) invocation_id = 1;
    uint32_t epoch = invocation_id << 16;

    constexpr uint32_t cb_scratch_in  = TileMatrixCB::CB_SCRATCH_GATHER;
    constexpr uint32_t cb_telem       = TileMatrixCB::CB_TELEMETRY;

    uint32_t telem_l1 = get_write_ptr(cb_telem);
    volatile uint32_t* sync_flags = reinterpret_cast<volatile uint32_t*>(telem_l1);

    // Handshake with Worker 1 (NCRISC) and Worker 2..4 (TRISC)
    WAYPOINT("B_01");
    while (sync_flags[SyncSlot::NCRISC_READY] < epoch) {
        asm volatile("" ::: "memory");
    }

    sync_flags[SyncSlot::BRISC_READY] = epoch;
    asm volatile("fence rw, rw" ::: "memory");
    WAYPOINT("B_02");

    while (sync_flags[SyncSlot::TRISC_READY] < epoch) {
        asm volatile("" ::: "memory");
    }
    WAYPOINT("B_03");

    uint32_t cycles_gather = 0;
    uint32_t cycles_scatter = 0;
    uint32_t cycles_brisc_wait = 0;
    uint32_t max_steps = max_iter * num_tiles;

    for (uint32_t batch = 0; batch < num_batches; ++batch) {
        // Tri-core batch rendezvous barrier
        sync_flags[SyncSlot::BATCH_BARRIER_BRISC] = epoch + batch + 1;
        asm volatile("fence rw, rw" ::: "memory");
        WAYPOINT("B_03");
        while (sync_flags[SyncSlot::BATCH_BARRIER_NCRISC] < epoch + batch + 1) {
            asm volatile("" ::: "memory");
        }
        WAYPOINT("B_04");
        while (sync_flags[SyncSlot::BATCH_BARRIER_TRISC] < epoch + batch + 1) {
            asm volatile("" ::: "memory");
        }
        WAYPOINT("B_05");

        // Wait for channel observations / patterned data to be ready in L1
        while (sync_flags[SyncSlot::BATCH_READY] < epoch + batch + 1) {
            asm volatile("" ::: "memory");
        }
        WAYPOINT("B_06");

        uint32_t vnode_l1 = sync_flags[SyncSlot::VNODE_L1_ADDR];
        char* vnode_ptr = reinterpret_cast<char*>(vnode_l1);

        // Strict Sequential Lockstep Step Loop: Gather -> Compute -> Scatter
        for (uint32_t step = 0; step < max_steps; ++step) {
            uint32_t bar = batch * max_steps + step + 1;
            uint32_t t_step = step % num_tiles;

            sync_flags[SyncSlot::DEBUG_BRISC_STEP] = (step << 8) | 1;
            uint8_t active_mask = static_cast<uint8_t>(sync_flags[SyncSlot::ACTIVE_MASK]);

            // ------------------------------------------------------------------
            // Phase 1: GATHER Stage for Tile t_step (Face 0 & Face 2)
            // ------------------------------------------------------------------
            WAYPOINT("B_RS");
            cb_reserve_back(cb_scratch_in, 1);
            uint32_t scratch_in_addr = get_write_ptr(cb_scratch_in);

            sync_flags[SyncSlot::SCRATCH_GATHER_PTR] = scratch_in_addr;
            sync_flags[SyncSlot::T0_GATHER_START]    = get_wall_clock();
            sync_flags[SyncSlot::GATHER_START]       = bar;
            asm volatile("fence rw, rw" ::: "memory");
            WAYPOINT("B_GT");

            if (active_mask != 0) {
                uint32_t t_g0 = get_wall_clock();
                #pragma GCC unroll 3
                for (uint32_t d = 0; d < 3; ++d) {
                    gather_slice_l1(scratch_in_addr + 0 + d * 128, QC_LDPC_4CW::TILE_SLICES_W0_F0[t_step][d], vnode_ptr);
                    gather_slice_l1(scratch_in_addr + 1024 + d * 128, QC_LDPC_4CW::TILE_SLICES_W0_F2[t_step][d], vnode_ptr);
                }
                noc_async_read_barrier();
                cycles_gather += (get_wall_clock() - t_g0);
            }

            WAYPOINT("B_WG");
            while (sync_flags[SyncSlot::GATHER_DONE_W1] < bar) {
                asm volatile("" ::: "memory");
            }
            WAYPOINT("B_PS");
            cb_push_back(cb_scratch_in, 1);
            sync_flags[SyncSlot::T1_GATHER_DONE] = get_wall_clock();
            sync_flags[SyncSlot::GATHER_DONE_W0] = bar;
            asm volatile("fence rw, rw" ::: "memory");
            WAYPOINT("B_GD");

            // ------------------------------------------------------------------
            // Phase 2: Wait for SCATTER Stage (Compute executes in parallel on TRISC)
            // ------------------------------------------------------------------
            uint32_t t_w0 = get_wall_clock();
            WAYPOINT("B_WS");
            while (sync_flags[SyncSlot::SCATTER_START] < bar) {
                asm volatile("" ::: "memory");
            }
            cycles_brisc_wait += (get_wall_clock() - t_w0);

            uint32_t scratch_out_addr = sync_flags[SyncSlot::SCRATCH_SCATTER_PTR];

            // ------------------------------------------------------------------
            // Phase 3: SCATTER Stage for Tile t_step (Face 0 & Face 2)
            // Blind Unmasked DMA Scatter: All 4 codewords are written unconditionally.
            // Codewords that have converged are protected by having their convergence
            // iteration and 0-error status already locked; no post-convergence errors
            // are ever counted.
            // ------------------------------------------------------------------
            WAYPOINT("B_SC");
            uint32_t t_s0 = get_wall_clock();
            if (active_mask != 0) {
                #pragma GCC unroll 3
                for (uint32_t d = 0; d < 3; ++d) {
                    scatter_slice_l1(QC_LDPC_4CW::TILE_SLICES_W0_F0[t_step][d], scratch_out_addr + 0 + d * 128, vnode_ptr);
                    scatter_slice_l1(QC_LDPC_4CW::TILE_SLICES_W0_F2[t_step][d], scratch_out_addr + 1024 + d * 128, vnode_ptr);
                }
                noc_async_write_barrier();
            }
            cycles_scatter += (get_wall_clock() - t_s0);

            // Information Node Parity Scan (Tile 39)
#if !LDPC_TEST_PASSTHROUGH_MODE
            if (t_step == num_tiles - 1) {
                uint32_t mask_err01 = ((active_mask & 1) ? 0x00008000 : 0) | ((active_mask & 2) ? 0x80000000 : 0);
                uint32_t mask_err23 = ((active_mask & 4) ? 0x00008000 : 0) | ((active_mask & 8) ? 0x80000000 : 0);

                const auto* p0 = reinterpret_cast<const uint32_t*>(vnode_ptr);
                uint32_t brisc_err01 = 0;
                uint32_t brisc_err23 = 0;
                if (mask_err01 != 0 || mask_err23 != 0) {
                    #pragma GCC unroll 2
                    for (uint32_t i = 0; i < 2048; i += 16) {
                        if (mask_err01) {
                            brisc_err01 |= (p0[i + 0] | p0[i + 2] | p0[i + 4] | p0[i + 6] |
                                            p0[i + 8] | p0[i + 10] | p0[i + 12] | p0[i + 14]);
                        }
                        if (mask_err23) {
                            brisc_err23 |= (p0[i + 1] | p0[i + 3] | p0[i + 5] | p0[i + 7] |
                                            p0[i + 9] | p0[i + 11] | p0[i + 13] | p0[i + 15]);
                        }
                        if ((rv32_andn(mask_err01, brisc_err01) | rv32_andn(mask_err23, brisc_err23)) == 0) {
                            break;
                        }
                    }
                }
                uint32_t info_err_mask = 0;
                if ((brisc_err01 & 0x00008000) && (active_mask & 1)) info_err_mask |= 1;
                if ((brisc_err01 & 0x80000000) && (active_mask & 2)) info_err_mask |= 2;
                if ((brisc_err23 & 0x00008000) && (active_mask & 4)) info_err_mask |= 4;
                if ((brisc_err23 & 0x80000000) && (active_mask & 8)) info_err_mask |= 8;
                sync_flags[SyncSlot::PARITY_REPORT_W0] = info_err_mask;
                asm volatile("fence rw, rw" ::: "memory");
            }
#endif

            sync_flags[SyncSlot::SCATTER_DONE_W0] = bar;
            asm volatile("fence rw, rw" ::: "memory");
            WAYPOINT("B_SD");

            // Wait for Worker 1 (NCRISC) to finish scatter and commit L1 writeback
            while (sync_flags[SyncSlot::SCATTER_DONE_W1] < bar) {
                asm volatile("" ::: "memory");
            }

            // Early termination check at iteration boundary
            if (t_step == num_tiles - 1) {
                if (sync_flags[SyncSlot::EARLY_TERMINATION] != 0) {
                    break;
                }
            }
        }
    }

    sync_flags[SyncSlot::BRISC_CYCLES]        = get_wall_clock();
    sync_flags[TelemSlot::BR_GATHER]          = cycles_gather;
    sync_flags[TelemSlot::BR_WAIT]            = cycles_brisc_wait;
    sync_flags[TelemSlot::BR_SCATTER]         = cycles_scatter;
    sync_flags[SyncSlot::BATCH_BARRIER_BRISC] = epoch + num_batches + 1;
}
