#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/waypoint.h"
#include "ldpc_circulant_dma_core.h"

// ==============================================================================
// Tenstorrent QuietBox 2: 4-Codeword Circulant DMA LDPC Decoder Engine
// Component: Data Movement Kernel 1 (NCRISC - Worker 1: Dual Gather & Dual Scatter)
// Architecture: Centralized TRISC1 Maestro with Bounded Satellite Autonomy (DAE)
// Target: Tenstorrent Blackhole (440 Tensix Cores, 2,200 RISC-V Processors)
//
// Key Responsibilities:
// 1. Contiguous 512-bit DMA Gather & Scatter of Face 1 and Face 3 via compile-time
//    analytical descriptors TILE_SLICES_W1_F1 and TILE_SLICES_W1_F3.
// 2. Parity Node Early Termination Scan: On Tile 39, scans sign bits across 1,024
//    parity bits for CW0..3 and reports the 4-bit error mask to SyncSlot::PARITY_REPORT_W1.
// 3. Batch Telemetry & Inter-Pair Control: Resets TRISC1_CMD to CMD_IDLE at pair
//    boundaries, publishes BATCH_READY, and records final cumulative telemetry to stats.
// ==============================================================================

static inline uint32_t get_wall_clock() {
    volatile uint32_t* const wall_clock_reg = reinterpret_cast<volatile uint32_t*>(0xFFB121F0);
    return *wall_clock_reg;
}

void kernel_main() {
    uint32_t num_batch_pairs = get_arg_val<uint32_t>(0); // Number of 8-codeword pairs
    uint32_t max_iter        = get_arg_val<uint32_t>(1);
    uint32_t num_tiles       = get_arg_val<uint32_t>(2);
    uint32_t stats_dram_addr = get_arg_val<uint32_t>(3);
    uint32_t dram_bank       = get_arg_val<uint32_t>(4);
    uint32_t stats_bytes     = get_arg_val<uint32_t>(5);
    (void)stats_bytes;
    uint32_t mu_bits         = get_arg_val<uint32_t>(6);
    uint32_t sigma_bits      = get_arg_val<uint32_t>(7);
    uint32_t global_seed     = get_arg_val<uint32_t>(8);
    (void)global_seed;
    uint32_t invocation_id   = get_arg_val<uint32_t>(9);
    uint32_t diff_test_mode  = get_arg_val<uint32_t>(10);
    (void)diff_test_mode;
    uint32_t start_batch     = get_arg_val<uint32_t>(11);
    (void)start_batch;
    uint32_t core_id         = get_arg_val<uint32_t>(12);
    (void)core_id;

    if (invocation_id == 0) invocation_id = 1;
    uint32_t epoch = invocation_id << 16;

    constexpr uint32_t cb_scratch_out_a = TileMatrixCB::CB_SCRATCH_SCATTER_0;
    constexpr uint32_t cb_scratch_out_b = TileMatrixCB::CB_SCRATCH_SCATTER_1;
    constexpr uint32_t cb_vnode_a       = TileMatrixCB::CB_VNODE_A;
    constexpr uint32_t cb_vnode_b       = TileMatrixCB::CB_VNODE_B;
    constexpr uint32_t cb_telemetry     = TileMatrixCB::CB_TELEMETRY;

    uint32_t stats_l1 = get_write_ptr(cb_telemetry);
    volatile uint32_t* sync_flags = reinterpret_cast<volatile uint32_t*>(stats_l1);
    auto* stats = reinterpret_cast<uint32_t*>(stats_l1);

    // Zero out telemetry and mailboxes EXCEPT inter-worker rendezvous semaphores
    for (uint32_t i = 1; i < 63; ++i) {
        sync_flags[i] = 0;
    }
    for (uint32_t i = 64; i < 1024; ++i) {
        sync_flags[i] = 0;
    }

    // Announce NCRISC ready for this epoch
    sync_flags[63] = epoch;
    asm volatile("fence rw, rw" ::: "memory");
    WAYPOINT("N_01");

    // Wait until Worker 0 (BRISC) is ready
    while (sync_flags[0] < epoch) {
        asm volatile("" ::: "memory");
    }
    WAYPOINT("N_02");

    uint32_t total_be = 0;
    uint32_t total_fe = 0;
    uint32_t total_iters = 0;
    uint32_t total_tiles_sim = 0;
    uint32_t first_failing_batch = 0xFFFFFFFF;
    uint32_t first_failing_cw    = 0;
    uint32_t first_failing_be    = 0;
    uint32_t first_failing_mask  = 0;

    uint32_t c_start = get_wall_clock();
    uint32_t cycles_gather = 0;
    uint32_t cycles_scatter = 0;
    uint32_t cycles_wait = 0;
    uint32_t cycles_channel = 0;
    uint32_t cycles_error_scan = 0;

    uint32_t max_steps = max_iter * num_tiles;
    uint32_t total_steps = 2 * max_steps + 2;

    uint32_t vnode_l1_a = get_read_ptr(cb_vnode_a);
    uint32_t vnode_l1_b = get_read_ptr(cb_vnode_b);
    uint32_t vnodes[2] = {vnode_l1_a, vnode_l1_b};
    uint32_t cb_scat_out[2] = {cb_scratch_out_a, cb_scratch_out_b};

    // --------------------------------------------------------------------------
    // Main Simulation Loop (Dual-Batch Pairs: 8 Codewords per Pair)
    // --------------------------------------------------------------------------
    for (uint32_t pair = 0; pair < num_batch_pairs; ++pair) {
        // Wait for TRISC SFPU to finish generating and packing channel observations
        WAYPOINT("N_03");
        uint32_t t_ch0 = get_wall_clock();
        for (uint32_t t = 0; t < 10; ++t) {
            cb_wait_front(cb_vnode_a, 1);
            cb_pop_front(cb_vnode_a, 1);
        }
        for (uint32_t t = 0; t < 10; ++t) {
            cb_wait_front(cb_vnode_b, 1);
            cb_pop_front(cb_vnode_b, 1);
        }
        cycles_channel += (get_wall_clock() - t_ch0);

        if (pair == 0) {
            const auto* pa = reinterpret_cast<const uint32_t*>(vnode_l1_a);
            stats[TelemSlot::MU_BITS]    = mu_bits;
            stats[TelemSlot::SIGMA_BITS] = sigma_bits;
            uint32_t na0 = 0, na1 = 0, na2 = 0, na3 = 0;
            for (uint32_t i = 0; i < 2048; i += 2) {
                uint32_t w0 = pa[i + 0];
                uint32_t w1 = pa[i + 1];
                na0 += ((w0 >> 15) & 1); na1 += (w0 >> 31);
                na2 += ((w1 >> 15) & 1); na3 += (w1 >> 31);
            }
            stats[TelemSlot::LAST_BATCH_BE_CW0 + 0] = na0;
            stats[TelemSlot::LAST_BATCH_BE_CW0 + 1] = na1;
            stats[TelemSlot::LAST_BATCH_BE_CW0 + 2] = na2;
            stats[TelemSlot::LAST_BATCH_BE_CW0 + 3] = na3;
        }

        // Reset early termination flag, active masks, and scan flags
        sync_flags[SyncSlot::EARLY_TERMINATION] = 0;
        sync_flags[SyncSlot::ACTIVE_MASK_A]     = 0x0F;
        sync_flags[SyncSlot::ACTIVE_MASK_B]     = 0x0F;
        sync_flags[SyncSlot::PARITY_REPORT_W0]  = 0;
        sync_flags[SyncSlot::PARITY_REPORT_W1]  = 0;
        sync_flags[SyncSlot::VNODE_L1_ADDR_A]   = vnode_l1_a;
        sync_flags[SyncSlot::VNODE_L1_ADDR_B]   = vnode_l1_b;
        sync_flags[SyncSlot::TRISC1_CMD]        = static_cast<uint32_t>(MaestroCmd::CMD_IDLE);
        sync_flags[SyncSlot::DRAIN_ACK_W1]      = pair;
        asm volatile("fence rw, rw" ::: "memory");

        for (int c = 0; c < 8; ++c) {
            stats[TelemSlot::LAST_BATCH_ITER_CW0 + c] = 0;
            stats[TelemSlot::LAST_BATCH_BE_CW0 + c]   = 0;
        }

        // Tri-core batch rendezvous barrier
        sync_flags[SyncSlot::BATCH_BARRIER_NCRISC] = epoch + pair + 1;
        asm volatile("fence rw, rw" ::: "memory");
        WAYPOINT("N_04");
        while (sync_flags[SyncSlot::BATCH_BARRIER_BRISC] < epoch + pair + 1) {
            asm volatile("" ::: "memory");
        }
        WAYPOINT("N_05");
        while (sync_flags[SyncSlot::BATCH_BARRIER_TRISC] < epoch + pair + 1) {
            asm volatile("" ::: "memory");
        }
        WAYPOINT("N_06");

        uint8_t active_mask_a = 0x0F;
        uint8_t active_mask_b = 0x0F;
        uint32_t iter_converged_a[4] = {max_iter, max_iter, max_iter, max_iter};
        uint32_t iter_converged_b[4] = {max_iter, max_iter, max_iter, max_iter};
        uint32_t cw_errs_a[4] = {0, 0, 0, 0};
        uint32_t cw_errs_b[4] = {0, 0, 0, 0};

        // Publish variable node base addresses and active masks before signaling batch ready
        sync_flags[SyncSlot::VNODE_L1_ADDR_A] = vnode_l1_a;
        sync_flags[SyncSlot::VNODE_L1_ADDR_B] = vnode_l1_b;
        sync_flags[SyncSlot::ACTIVE_MASK_A]   = 0x0F;
        sync_flags[SyncSlot::ACTIVE_MASK_B]   = 0x0F;
        sync_flags[SyncSlot::BATCH_READY]     = epoch + pair + 1;
        asm volatile("fence rw, rw" ::: "memory");
        WAYPOINT("N_07");

        // Pre-Step Gate: Wait for TRISC1 Maestro to launch Step 0
        while (sync_flags[SyncSlot::TRISC1_CMD] != static_cast<uint32_t>(MaestroCmd::CMD_STEP_EXEC) ||
               sync_flags[SyncSlot::TRISC1_GRANT_STEP] < pair * total_steps + 1) {
            asm volatile("" ::: "memory");
        }
        WAYPOINT("N_08");

        uint32_t last_completed_k = 0;

        // Decoupled Access-Execute Step Loop
        for (uint32_t k = 0; k < total_steps; ++k) {
            uint32_t bar = pair * total_steps + k + 1;
            uint32_t b_dae = k & 1;
            uint32_t s_dae = k >> 1;

            uint32_t vnode_l1 = vnodes[b_dae];
            const char* vnode_bytes_ptr = reinterpret_cast<const char*>(vnode_l1);
            char* vnode_out_ptr = reinterpret_cast<char*>(vnode_l1);

            sync_flags[SyncSlot::DEBUG_NCRISC_STEP] = (k << 8) | 1; // Start step

            sync_flags[SyncSlot::DEBUG_NCRISC_STEP] = (k << 8) | 2; // Wait TRISC1 grant
            WAYPOINT("N_WT");
            while (sync_flags[SyncSlot::TRISC1_GRANT_STEP] < bar) {
                asm volatile("" ::: "memory");
                if (sync_flags[SyncSlot::TRISC1_CMD] >= static_cast<uint32_t>(MaestroCmd::CMD_DRAIN_RESIDUAL)) {
                    break;
                }
            }
            if (sync_flags[SyncSlot::TRISC1_CMD] >= static_cast<uint32_t>(MaestroCmd::CMD_DRAIN_RESIDUAL)) {
                sync_flags[SyncSlot::DEBUG_NCRISC_STEP] = (k << 8) | 0xEE; // Drain exit
                WAYPOINT("N_DR");
                break;
            }
            sync_flags[SyncSlot::DEBUG_NCRISC_STEP] = (k << 8) | 3; // Grant received
            WAYPOINT("N_GR");

            uint8_t active_mask = static_cast<uint8_t>(
                (b_dae == 0) ? sync_flags[SyncSlot::ACTIVE_MASK_A] : sync_flags[SyncSlot::ACTIVE_MASK_B]
            );

            // 1. Autonomous Scatter Stage: Scatter previous tile (s_dae - 1) of Batch b_dae
            if (s_dae >= 1 && (s_dae - 1) < max_steps) {
                uint32_t t_w0 = get_wall_clock();
                uint32_t active_out_cb = cb_scat_out[b_dae];
                sync_flags[SyncSlot::DEBUG_NCRISC_STEP] = (k << 8) | 4; // Wait cb_wait_front active_out_cb
                WAYPOINT("N_WF");
                cb_wait_front(active_out_cb, 1);
                cycles_wait += (get_wall_clock() - t_w0);

                uint32_t scratch_out_addr = get_read_ptr(active_out_cb);
                sync_flags[SyncSlot::SCRATCH_SCATTER_PTR] = scratch_out_addr;
                sync_flags[SyncSlot::SCATTER_START] = bar;
                asm volatile("fence rw, rw" ::: "memory");

                uint32_t s_scat = s_dae - 1;
                uint32_t t_scat = s_scat % num_tiles;

                sync_flags[SyncSlot::DEBUG_NCRISC_STEP] = (k << 8) | 5; // Scattering
                WAYPOINT("N_SC");

                // Worker 1 scatters Face 1 and Face 3 via Contiguous Circulant DMA Bursts
                uint32_t t_s0 = get_wall_clock();
                if (active_mask != 0) {
                    #pragma GCC unroll 3
                    for (uint32_t d = 0; d < 3; ++d) {
                        scatter_slice_masked_l1(QC_LDPC_4CW::TILE_SLICES_W1_F1[t_scat][d], scratch_out_addr + 512 + d * 128, vnode_out_ptr, active_mask);
                        scatter_slice_masked_l1(QC_LDPC_4CW::TILE_SLICES_W1_F3[t_scat][d], scratch_out_addr + 1536 + d * 128, vnode_out_ptr, active_mask);
                    }
                    if (active_mask == 0xF) {
                        noc_async_write_barrier();
                    }
                }
                cycles_scatter += (get_wall_clock() - t_s0);

                // Parity Node Early Termination Scan (Tile 39)
                if (t_scat == num_tiles - 1) {
                    uint32_t t_es0 = get_wall_clock();
                    uint32_t mask_err01 = ((active_mask & 1) ? 0x00008000 : 0) | ((active_mask & 2) ? 0x80000000 : 0);
                    uint32_t mask_err23 = ((active_mask & 4) ? 0x00008000 : 0) | ((active_mask & 8) ? 0x80000000 : 0);

                    const auto* p1 = reinterpret_cast<const uint32_t*>(vnode_bytes_ptr + 8192);
                    uint32_t ncrisc_err01 = 0;
                    uint32_t ncrisc_err23 = 0;
                    if (mask_err01 != 0 || mask_err23 != 0) {
                        #pragma GCC unroll 2
                        for (uint32_t i = 0; i < 2048; i += 16) {
                            if (mask_err01) {
                                ncrisc_err01 |= (p1[i + 0] | p1[i + 2] | p1[i + 4] | p1[i + 6] |
                                                 p1[i + 8] | p1[i + 10] | p1[i + 12] | p1[i + 14]);
                            }
                            if (mask_err23) {
                                ncrisc_err23 |= (p1[i + 1] | p1[i + 3] | p1[i + 5] | p1[i + 7] |
                                                 p1[i + 9] | p1[i + 11] | p1[i + 13] | p1[i + 15]);
                            }
                            if ((rv32_andn(mask_err01, ncrisc_err01) | rv32_andn(mask_err23, ncrisc_err23)) == 0) {
                                break;
                            }
                        }
                    }

                    uint32_t parity_err_mask = 0;
                    if ((ncrisc_err01 & 0x00008000) && (active_mask & 1)) parity_err_mask |= 1;
                    if ((ncrisc_err01 & 0x80000000) && (active_mask & 2)) parity_err_mask |= 2;
                    if ((ncrisc_err23 & 0x00008000) && (active_mask & 4)) parity_err_mask |= 4;
                    if ((ncrisc_err23 & 0x80000000) && (active_mask & 8)) parity_err_mask |= 8;
                    sync_flags[SyncSlot::PARITY_REPORT_W1] = parity_err_mask;
                    asm volatile("fence rw, rw" ::: "memory");
                    cycles_error_scan += (get_wall_clock() - t_es0);
                }

                sync_flags[SyncSlot::DEBUG_NCRISC_STEP] = (k << 8) | 6; // Wait SCATTER_DONE_W0
                WAYPOINT("N_WS");
                // Wait for Worker 0 to finish scattering Face 0 & 2 before releasing scratch page
                while (sync_flags[SyncSlot::SCATTER_DONE_W0] < bar) {
                    asm volatile("" ::: "memory");
                }
                WAYPOINT("N_PP");
                cb_pop_front(active_out_cb, 1);

                // Mandatory Invariant: Signal SCATTER_DONE_W1 strictly AFTER writing PARITY_REPORT_W1
                sync_flags[SyncSlot::SCATTER_DONE_W1] = bar;
                asm volatile("fence rw, rw" ::: "memory");
                WAYPOINT("N_SD");
            }

            // 2. Autonomous Gather Stage: Gather current tile s_dae of Batch b_dae
            if (s_dae < max_steps) {
                sync_flags[SyncSlot::DEBUG_NCRISC_STEP] = (k << 8) | 7; // Wait GATHER_START
                WAYPOINT("N_WG");
                while (sync_flags[SyncSlot::GATHER_START] < bar) {
                    asm volatile("" ::: "memory");
                }
                WAYPOINT("N_GT");
                uint32_t scratch_in_addr = sync_flags[SyncSlot::SCRATCH_GATHER_PTR];

                if (active_mask != 0) {
                    total_tiles_sim++;
                    uint32_t t_g0 = get_wall_clock();
                    uint32_t t_gath = s_dae % num_tiles;
                    #pragma GCC unroll 3
                    for (uint32_t d = 0; d < 3; ++d) {
                        gather_slice_l1(scratch_in_addr + 512 + d * 128, QC_LDPC_4CW::TILE_SLICES_W1_F1[t_gath][d], vnode_bytes_ptr);
                        gather_slice_l1(scratch_in_addr + 1536 + d * 128, QC_LDPC_4CW::TILE_SLICES_W1_F3[t_gath][d], vnode_bytes_ptr);
                    }
                    noc_async_read_barrier();
                    cycles_gather += (get_wall_clock() - t_g0);
                }

                sync_flags[SyncSlot::GATHER_DONE_W1] = bar;
                asm volatile("fence rw, rw" ::: "memory");
                WAYPOINT("N_GD");
                sync_flags[SyncSlot::DEBUG_NCRISC_STEP] = (k << 8) | 8; // Gather done
            }
            sync_flags[SyncSlot::DEBUG_NCRISC_STEP] = (k << 8) | 9; // Step finished
            last_completed_k = k;
        }

        // Drain Residual Speculative Scatter Tile upon Early Termination
        if (sync_flags[SyncSlot::TRISC1_CMD] == static_cast<uint32_t>(MaestroCmd::CMD_DRAIN_RESIDUAL)) {
            if (last_completed_k >= 1 && last_completed_k <= 2 * max_steps) {
                uint32_t b_prev = (last_completed_k - 1) % 2;
                uint32_t res_out = cb_scat_out[b_prev];
                cb_wait_front(res_out, 1);
                cb_pop_front(res_out, 1);
            }
        }
        sync_flags[SyncSlot::DRAIN_ACK_W1] = pair + 1;

        // Residual Bit Error Check on Information Bits
        active_mask_a = static_cast<uint8_t>(sync_flags[SyncSlot::ACTIVE_MASK_A]);
        active_mask_b = static_cast<uint8_t>(sync_flags[SyncSlot::ACTIVE_MASK_B]);
        for (int cw = 0; cw < 4; ++cw) {
            uint32_t it_a = stats[TelemSlot::LAST_BATCH_ITER_CW0 + cw];
            if (it_a != 0) iter_converged_a[cw] = it_a;
            uint32_t it_b = stats[TelemSlot::LAST_BATCH_ITER_CW0 + 4 + cw];
            if (it_b != 0) iter_converged_b[cw] = it_b;
        }

        if (active_mask_a != 0) {
            const auto* p32 = reinterpret_cast<const uint32_t*>(vnode_l1_a);
            uint32_t e0 = 0, e1 = 0, e2 = 0, e3 = 0;
            #pragma GCC unroll 4
            for (uint32_t i = 0; i < 2048; i += 2) {
                uint32_t w0 = p32[i + 0];
                uint32_t w1 = p32[i + 1];
                e0 += ((w0 >> 15) & 1); e1 += (w0 >> 31);
                e2 += ((w1 >> 15) & 1); e3 += (w1 >> 31);
            }
            if (active_mask_a & 1) cw_errs_a[0] = e0;
            if (active_mask_a & 2) cw_errs_a[1] = e1;
            if (active_mask_a & 4) cw_errs_a[2] = e2;
            if (active_mask_a & 8) cw_errs_a[3] = e3;
        }

        if (active_mask_b != 0) {
            const auto* p32 = reinterpret_cast<const uint32_t*>(vnode_l1_b);
            uint32_t e0 = 0, e1 = 0, e2 = 0, e3 = 0;
            #pragma GCC unroll 4
            for (uint32_t i = 0; i < 2048; i += 2) {
                uint32_t w0 = p32[i + 0];
                uint32_t w1 = p32[i + 1];
                e0 += ((w0 >> 15) & 1); e1 += (w0 >> 31);
                e2 += ((w1 >> 15) & 1); e3 += (w1 >> 31);
            }
            if (active_mask_b & 1) cw_errs_b[0] = e0;
            if (active_mask_b & 2) cw_errs_b[1] = e1;
            if (active_mask_b & 4) cw_errs_b[2] = e2;
            if (active_mask_b & 8) cw_errs_b[3] = e3;
        }

        for (int cw = 0; cw < 4; ++cw) {
            stats[TelemSlot::LAST_BATCH_BE_CW0 + cw] = cw_errs_a[cw];
            stats[TelemSlot::LAST_BATCH_ITER_CW0 + cw] = iter_converged_a[cw];
            total_be += cw_errs_a[cw];
            if (cw_errs_a[cw] > 0) total_fe++;
            total_iters += iter_converged_a[cw];
        }
        for (int cw = 0; cw < 4; ++cw) {
            stats[TelemSlot::LAST_BATCH_BE_CW0 + 4 + cw] = cw_errs_b[cw];
            stats[TelemSlot::LAST_BATCH_ITER_CW0 + 4 + cw] = iter_converged_b[cw];
            total_be += cw_errs_b[cw];
            if (cw_errs_b[cw] > 0) total_fe++;
            total_iters += iter_converged_b[cw];
        }

        if (first_failing_batch == 0xFFFFFFFF) {
            for (int cw = 0; cw < 4; ++cw) {
                if (cw_errs_a[cw] > 0) {
                    first_failing_batch = 2 * pair;
                    first_failing_cw    = cw;
                    first_failing_be    = cw_errs_a[cw];
                    first_failing_mask  = active_mask_a;
                    break;
                }
            }
            if (first_failing_batch == 0xFFFFFFFF) {
                for (int cw = 0; cw < 4; ++cw) {
                    if (cw_errs_b[cw] > 0) {
                        first_failing_batch = 2 * pair + 1;
                        first_failing_cw    = 4 + cw;
                        first_failing_be    = cw_errs_b[cw];
                        first_failing_mask  = active_mask_b;
                        break;
                    }
                }
            }
        }
    }

    uint32_t c_end = get_wall_clock();
    uint32_t elapsed_cycles = c_end - c_start;

    // Wait for BRISC and TRISC to finish cycle measurements
    sync_flags[SyncSlot::BATCH_BARRIER_NCRISC] = epoch + num_batch_pairs + 1;
    while (sync_flags[SyncSlot::BATCH_BARRIER_BRISC] < epoch + num_batch_pairs + 1) { }
    while (sync_flags[SyncSlot::BATCH_BARRIER_TRISC] < epoch + num_batch_pairs + 1) { }

    // Stream Final Telemetry to DRAM via NoC DMA
    stats[TelemSlot::TOTAL_BE]          = total_be;
    stats[TelemSlot::TOTAL_FE]          = total_fe;
    stats[TelemSlot::TOTAL_ITERS]       = total_iters;
    stats[TelemSlot::TOTAL_CWS]         = num_batch_pairs * 8;
    stats[TelemSlot::ELAPSED_CYCLES]    = elapsed_cycles;
    stats[TelemSlot::CYCLES_PER_BATCH]  = elapsed_cycles / (num_batch_pairs > 0 ? num_batch_pairs : 1);
    stats[TelemSlot::TRISC_CYCLES]      = sync_flags[SyncSlot::TRISC_CYCLES];
    stats[TelemSlot::TRISC_MATH_CYCLES] = sync_flags[SyncSlot::TRISC_MATH_CYCLES];
    stats[TelemSlot::BRISC_CYCLES]      = sync_flags[SyncSlot::BRISC_CYCLES];
    stats[TelemSlot::CYCLES_CHANNEL]    = cycles_channel;
    stats[TelemSlot::CYCLES_ERROR_SCAN] = cycles_error_scan;
    stats[TelemSlot::TOTAL_TILES_SIM]   = total_tiles_sim;
    stats[TelemSlot::TELEMETRY_L1_ADDR] = stats_l1;

    // NCRISC (Worker 1) per-tile breakdown
    stats[TelemSlot::NC_GATHER]         = cycles_gather;
    stats[TelemSlot::NC_WAIT]           = cycles_wait;
    stats[TelemSlot::NC_SCATTER]        = cycles_scatter;

    uint64_t dst_noc_addr = get_noc_addr_from_bank_id<true>(dram_bank, stats_dram_addr);
    noc_async_write(stats_l1, dst_noc_addr, 4096);
    noc_async_write_barrier();
}
