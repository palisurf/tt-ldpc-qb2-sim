#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/waypoint.h"
#include "ldpc_circulant_dma_core.h"

// ==============================================================================
// Tenstorrent QuietBox 2: 4-Codeword Circulant DMA LDPC Decoder Engine
// Component: Data Movement Kernel 0 (BRISC - Worker 0: Dual Gather & Dual Scatter)
// Architecture: Centralized TRISC1 Maestro with Bounded Satellite Autonomy (DAE)
// Target: Tenstorrent Blackhole (440 Tensix Cores, 2,200 RISC-V Processors)
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
    uint32_t num_batch_pairs = get_arg_val<uint32_t>(0); // Number of 8-codeword pairs
    uint32_t max_iter        = get_arg_val<uint32_t>(1);
    uint32_t num_tiles       = get_arg_val<uint32_t>(2);
    uint32_t invocation_id   = get_arg_val<uint32_t>(3);
    uint32_t core_id         = get_arg_val<uint32_t>(4);
    (void)core_id;

    if (invocation_id == 0) invocation_id = 1;
    uint32_t epoch = invocation_id << 16;

    constexpr uint32_t cb_scratch_in_a = TileMatrixCB::CB_SCRATCH_GATHER_0;
    constexpr uint32_t cb_scratch_in_b = TileMatrixCB::CB_SCRATCH_GATHER_1;
    constexpr uint32_t cb_vnode_a      = TileMatrixCB::CB_VNODE_A;
    constexpr uint32_t cb_vnode_b      = TileMatrixCB::CB_VNODE_B;
    constexpr uint32_t cb_telem        = TileMatrixCB::CB_TELEMETRY;

    uint32_t telem_l1 = get_write_ptr(cb_telem);
    volatile uint32_t* sync_flags = reinterpret_cast<volatile uint32_t*>(telem_l1);

    // Wait until Worker 1 (NCRISC) has arrived for this epoch
    WAYPOINT("B_01");
    while (sync_flags[63] < epoch) {
        asm volatile("" ::: "memory");
    }

    // Signal Worker 1 (NCRISC) that initialization is complete
    sync_flags[0] = epoch;
    asm volatile("fence rw, rw" ::: "memory");
    WAYPOINT("B_02");

    uint32_t cycles_gather = 0;
    uint32_t cycles_scatter = 0;
    uint32_t cycles_brisc_wait = 0;
    uint32_t max_steps = max_iter * num_tiles;
    uint32_t total_steps = 2 * max_steps + 2;

    uint32_t cb_scat_in[2] = {cb_scratch_in_a, cb_scratch_in_b};

    // --------------------------------------------------------------------------
    // Main Simulation Loop (Dual-Batch Pairs)
    // --------------------------------------------------------------------------
    for (uint32_t pair = 0; pair < num_batch_pairs; ++pair) {
        // Tri-core batch rendezvous barrier
        sync_flags[SyncSlot::BATCH_BARRIER_BRISC] = epoch + pair + 1;
        asm volatile("fence rw, rw" ::: "memory");
        WAYPOINT("B_03");
        while (sync_flags[SyncSlot::BATCH_BARRIER_NCRISC] < epoch + pair + 1) {
            asm volatile("" ::: "memory");
        }
        WAYPOINT("B_04");
        while (sync_flags[SyncSlot::BATCH_BARRIER_TRISC] < epoch + pair + 1) {
            asm volatile("" ::: "memory");
        }
        WAYPOINT("B_05");

        // Wait for channel observations to be ready in L1
        while (sync_flags[SyncSlot::BATCH_READY] < epoch + pair + 1) {
            asm volatile("" ::: "memory");
        }
        WAYPOINT("B_06");

        // Pre-Step Gate: Wait for TRISC1 Maestro to launch Step 0
        while (sync_flags[SyncSlot::TRISC1_CMD] != static_cast<uint32_t>(MaestroCmd::CMD_STEP_EXEC) ||
               sync_flags[SyncSlot::TRISC1_GRANT_STEP] < pair * total_steps + 1) {
            asm volatile("" ::: "memory");
        }
        WAYPOINT("B_07");

        uint32_t vnode_l1_a = sync_flags[SyncSlot::VNODE_L1_ADDR_A];
        uint32_t vnode_l1_b = sync_flags[SyncSlot::VNODE_L1_ADDR_B];
        uint32_t vnode_cbs[2] = {vnode_l1_a, vnode_l1_b};

        // Decoupled Access-Execute Step Loop
        for (uint32_t step = 0; step < total_steps; ++step) {
            WAYPOINT("B_08");
            uint32_t bar = pair * total_steps + step + 1;
            uint32_t b_dae = step & 1;
            uint32_t s_dae = step >> 1;

            sync_flags[SyncSlot::DEBUG_BRISC_STEP] = (step << 8) | 1; // Start step

            uint32_t vnode_l1 = vnode_cbs[b_dae];
            const char* vnode_bytes_ptr = reinterpret_cast<const char*>(vnode_l1);
            char* vnode_out_ptr = reinterpret_cast<char*>(vnode_l1);

            // 1. Wait for TRISC1 Macro Step Grant
            sync_flags[SyncSlot::DEBUG_BRISC_STEP] = (step << 8) | 2; // Wait TRISC1 grant
            WAYPOINT("B_WT");
            while (sync_flags[SyncSlot::TRISC1_GRANT_STEP] < bar) {
                asm volatile("" ::: "memory");
                if (sync_flags[SyncSlot::TRISC1_CMD] >= static_cast<uint32_t>(MaestroCmd::CMD_DRAIN_RESIDUAL)) {
                    break;
                }
            }
            if (sync_flags[SyncSlot::TRISC1_CMD] >= static_cast<uint32_t>(MaestroCmd::CMD_DRAIN_RESIDUAL)) {
                sync_flags[SyncSlot::DEBUG_BRISC_STEP] = (step << 8) | 0xEE; // Drain exit
                WAYPOINT("B_DR");
                break;
            }
            sync_flags[SyncSlot::DEBUG_BRISC_STEP] = (step << 8) | 3; // Grant received
            WAYPOINT("B_GR");

            uint8_t active_mask = static_cast<uint8_t>(
                (b_dae == 0) ? sync_flags[SyncSlot::ACTIVE_MASK_A] : sync_flags[SyncSlot::ACTIVE_MASK_B]
            );

            // 2. Autonomous Scatter Stage: Scatter previous tile (s_dae - 1) of Batch b_dae
            if (s_dae >= 1 && (s_dae - 1) < max_steps) {
                uint32_t t_w0 = get_wall_clock();
                sync_flags[SyncSlot::DEBUG_BRISC_STEP] = (step << 8) | 4; // Wait SCATTER_START
                WAYPOINT("B_WS");
                while (sync_flags[SyncSlot::SCATTER_START] < bar) {
                    asm volatile("" ::: "memory");
                }
                cycles_brisc_wait += (get_wall_clock() - t_w0);

                uint32_t scratch_out_addr = sync_flags[SyncSlot::SCRATCH_SCATTER_PTR];

                uint32_t s_scat = s_dae - 1;
                uint32_t t_scat = s_scat % num_tiles;

                sync_flags[SyncSlot::DEBUG_BRISC_STEP] = (step << 8) | 5; // Scattering
                WAYPOINT("B_SC");

                // Worker 0 scatters Face 0 and Face 2 via Contiguous Circulant DMA Bursts
                uint32_t t_s0 = get_wall_clock();
                if (active_mask != 0) {
                    #pragma GCC unroll 3
                    for (uint32_t d = 0; d < 3; ++d) {
                        scatter_slice_masked_l1(QC_LDPC_4CW::TILE_SLICES_W0_F0[t_scat][d], scratch_out_addr + 0 + d * 128, vnode_out_ptr, active_mask);
                        scatter_slice_masked_l1(QC_LDPC_4CW::TILE_SLICES_W0_F2[t_scat][d], scratch_out_addr + 1024 + d * 128, vnode_out_ptr, active_mask);
                    }
                    if (active_mask == 0xF) {
                        noc_async_write_barrier();
                    }
                }
                cycles_scatter += (get_wall_clock() - t_s0);

                // Information Node Parity Scan (Tile 39)
                if (t_scat == num_tiles - 1) {
                    uint32_t mask_err01 = ((active_mask & 1) ? 0x00008000 : 0) | ((active_mask & 2) ? 0x80000000 : 0);
                    uint32_t mask_err23 = ((active_mask & 4) ? 0x00008000 : 0) | ((active_mask & 8) ? 0x80000000 : 0);

                    const auto* p0 = reinterpret_cast<const uint32_t*>(vnode_bytes_ptr);
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

                // Mandatory Invariant: Signal SCATTER_DONE_W0 strictly AFTER writing PARITY_REPORT_W0
                sync_flags[SyncSlot::SCATTER_DONE_W0] = bar;
                asm volatile("fence rw, rw" ::: "memory");
                WAYPOINT("B_SD");
                sync_flags[SyncSlot::DEBUG_BRISC_STEP] = (step << 8) | 6; // SCATTER_DONE_W0 signaled
            }

            // 3. Autonomous Gather Stage: Gather current tile s_dae of Batch b_dae
            if (s_dae < max_steps) {
                uint32_t active_in_cb = cb_scat_in[b_dae];
                sync_flags[SyncSlot::DEBUG_BRISC_STEP] = (step << 8) | 7; // In gather reserve
                WAYPOINT("B_RS");
                cb_reserve_back(active_in_cb, 1);
                uint32_t scratch_in_addr = get_write_ptr(active_in_cb);

                sync_flags[SyncSlot::SCRATCH_GATHER_PTR] = scratch_in_addr;
                sync_flags[SyncSlot::GATHER_START] = bar;
                asm volatile("fence rw, rw" ::: "memory");
                WAYPOINT("B_GT");

                if (active_mask != 0) {
                    uint32_t t_g0 = get_wall_clock();
                    uint32_t t_gath = s_dae % num_tiles;
                    #pragma GCC unroll 3
                    for (uint32_t d = 0; d < 3; ++d) {
                        gather_slice_l1(scratch_in_addr + 0 + d * 128, QC_LDPC_4CW::TILE_SLICES_W0_F0[t_gath][d], vnode_bytes_ptr);
                        gather_slice_l1(scratch_in_addr + 1024 + d * 128, QC_LDPC_4CW::TILE_SLICES_W0_F2[t_gath][d], vnode_bytes_ptr);
                    }
                    noc_async_read_barrier();
                    cycles_gather += (get_wall_clock() - t_g0);
                }

                sync_flags[SyncSlot::DEBUG_BRISC_STEP] = (step << 8) | 8; // Wait GATHER_DONE_W1
                WAYPOINT("B_WG");
                while (sync_flags[SyncSlot::GATHER_DONE_W1] < bar) {
                    asm volatile("" ::: "memory");
                }
                WAYPOINT("B_PS");
                cb_push_back(active_in_cb, 1);
                sync_flags[SyncSlot::GATHER_DONE_W0] = bar;
                asm volatile("fence rw, rw" ::: "memory");
                WAYPOINT("B_GD");
                sync_flags[SyncSlot::DEBUG_BRISC_STEP] = (step << 8) | 9; // Step finished
            }
        }
    }

    sync_flags[SyncSlot::BRISC_CYCLES]        = get_wall_clock();
    sync_flags[TelemSlot::BR_GATHER]          = cycles_gather;
    sync_flags[TelemSlot::BR_WAIT]            = cycles_brisc_wait;
    sync_flags[TelemSlot::BR_SCATTER]         = cycles_scatter;
    sync_flags[SyncSlot::BATCH_BARRIER_BRISC] = epoch + num_batch_pairs + 1;
}
