#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/waypoint.h"
#include "ldpc_circulant_dma_core.h"

// ==============================================================================
// Tenstorrent QuietBox 2: 4-Codeword Circulant DMA LDPC Decoder Engine
// Component: Data Movement Kernel 1 (NCRISC - Worker 1: Dual Gather & Dual Scatter)
// Architecture: Centralized TRISC1 Maestro with Bounded Satellite Autonomy (DAE)
// Target: Pure 4-Codeword Engine (Single Batch)
//
// Key Responsibilities:
// 1. Contiguous 512-bit DMA Gather & Scatter of Face 1 and Face 3 via compile-time
//    analytical descriptors TILE_SLICES_W1_F1 and TILE_SLICES_W1_F3.
// 2. Deterministic Patterned Data Initialization and Verification for PassThrough mode.
// 3. Autonomous Dual-Worker Synchronization with Worker 0 (BRISC).
// ==============================================================================

static inline uint32_t get_wall_clock() {
    volatile uint32_t* const wall_clock_reg = reinterpret_cast<volatile uint32_t*>(0xFFB121F0);
    return *wall_clock_reg;
}



void kernel_main() {
    uint32_t num_batches     = get_arg_val<uint32_t>(0); // Number of 4-codeword batches
    uint32_t max_iter        = get_arg_val<uint32_t>(1);
    uint32_t num_tiles       = get_arg_val<uint32_t>(2);
    uint32_t stats_dram_addr = get_arg_val<uint32_t>(3);
    uint32_t dram_bank       = get_arg_val<uint32_t>(4);
    uint32_t stats_bytes     = get_arg_val<uint32_t>(5);
    (void)stats_bytes;
    uint32_t mu_bits         = get_arg_val<uint32_t>(6);
    uint32_t sigma_bits      = get_arg_val<uint32_t>(7);
    uint32_t seed_val        = get_arg_val<uint32_t>(8);
    uint32_t invocation_id   = get_arg_val<uint32_t>(9);
    uint32_t diff_test_mode  = get_arg_val<uint32_t>(10);
    (void)diff_test_mode;
    uint32_t start_batch     = get_arg_val<uint32_t>(11);
    (void)start_batch;
    uint32_t core_id         = get_arg_val<uint32_t>(12);

    if (invocation_id == 0) invocation_id = 1;
    uint32_t epoch = invocation_id << 16;

    constexpr uint32_t cb_scratch_in  = TileMatrixCB::CB_SCRATCH_GATHER;
    constexpr uint32_t cb_scratch_out = TileMatrixCB::CB_SCRATCH_SCATTER;
    constexpr uint32_t cb_vnode_a     = TileMatrixCB::CB_VNODE_A;
    constexpr uint32_t cb_vnode_b     = TileMatrixCB::CB_VNODE_B;
    constexpr uint32_t cb_vnode       = TileMatrixCB::CB_VNODE;
    constexpr uint32_t cb_r_msg       = TileMatrixCB::CB_R_MSG;
    constexpr uint32_t cb_telem       = TileMatrixCB::CB_TELEMETRY;

    uint32_t stats_l1 = get_write_ptr(cb_telem);
    volatile uint32_t* sync_flags = reinterpret_cast<volatile uint32_t*>(stats_l1);
    volatile uint32_t* stats = reinterpret_cast<volatile uint32_t*>(stats_l1);

    // Clear stats buffer
    for (uint32_t i = 0; i < 256; ++i) stats[i] = 0;
    stats[TelemSlot::TELEMETRY_L1_ADDR] = stats_l1;
    asm volatile("fence rw, rw" ::: "memory");

    // Handshake with Worker 0 (BRISC) and Worker 2..4 (TRISC)
    sync_flags[SyncSlot::NCRISC_READY] = epoch;
    asm volatile("fence rw, rw" ::: "memory");
    WAYPOINT("N_01");

    while (sync_flags[SyncSlot::BRISC_READY] < epoch) {
        asm volatile("" ::: "memory");
    }
    WAYPOINT("N_02");

    while (sync_flags[SyncSlot::TRISC_READY] < epoch) {
        asm volatile("" ::: "memory");
    }
    WAYPOINT("N_03");

    uint32_t total_be = 0;
    uint32_t total_fe = 0;
    uint32_t total_iters = 0;
    uint32_t total_tiles_sim = 0;

    uint32_t cycles_gather = 0;
    uint32_t cycles_scatter = 0;
    uint32_t cycles_wait = 0;
    uint32_t cycles_channel = 0;
    uint32_t cycles_error_scan = 0;

    uint32_t stage_gather_acc = 0;
    uint32_t stage_unpack_acc = 0;
    uint32_t stage_math_acc = 0;
    uint32_t stage_pack_acc = 0;
    uint32_t stage_scatter_acc = 0;
    uint32_t stage_total_acc = 0;

    uint32_t max_steps = max_iter * num_tiles;

    uint32_t first_failing_batch = 0xFFFFFFFF;
    uint32_t first_failing_cw    = 0xFFFFFFFF;
    uint32_t first_failing_be    = 0;
    uint32_t first_failing_mask  = 0;

    uint32_t c_start = get_wall_clock();

    for (uint32_t batch = 0; batch < num_batches; ++batch) {
        // Reset control semaphores for this batch
        sync_flags[SyncSlot::EARLY_TERMINATION] = 0;
        sync_flags[SyncSlot::ACTIVE_MASK]       = 0x0F;
        sync_flags[SyncSlot::PARITY_REPORT_W0]  = 0;
        sync_flags[SyncSlot::PARITY_REPORT_W1]  = 0;
        sync_flags[SyncSlot::TRISC1_CMD]        = static_cast<uint32_t>(MaestroCmd::CMD_IDLE);
        sync_flags[SyncSlot::DRAIN_ACK_W1]      = batch;
        asm volatile("fence rw, rw" ::: "memory");

        // Tri-core batch rendezvous barrier
        sync_flags[SyncSlot::BATCH_BARRIER_NCRISC] = epoch + batch + 1;
        asm volatile("fence rw, rw" ::: "memory");
        WAYPOINT("N_04");
        while (sync_flags[SyncSlot::BATCH_BARRIER_BRISC] < epoch + batch + 1) {
            asm volatile("" ::: "memory");
        }
        WAYPOINT("N_05");
        while (sync_flags[SyncSlot::BATCH_BARRIER_TRISC] < epoch + batch + 1) {
            asm volatile("" ::: "memory");
        }
        WAYPOINT("N_06");

        uint32_t t_ch0 = get_wall_clock();
        uint32_t active_cb = cb_vnode_a;
        uint32_t vnode_l1 = get_read_ptr(active_cb);
        char* vnode_ptr = reinterpret_cast<char*>(vnode_l1);
        auto* vnodes = reinterpret_cast<Packed4CW_BF16*>(vnode_l1);
        sync_flags[SyncSlot::VNODE_L1_ADDR] = vnode_l1;
        asm volatile("fence rw, rw" ::: "memory");

#if LDPC_TEST_PASSTHROUGH_MODE
        // Deterministic Patterned Data for Gather/Scatter Symmetry Verification
        for (uint32_t v = 0; v < 2560; ++v) {
            for (uint32_t cw = 0; cw < 4; ++cw) {
                float val = 1024.0f + static_cast<float>(v) + 0.1f * cw;
                vnodes[v].cw[cw] = fp32_to_bf16(val);
            }
        }
#endif
        cycles_channel += (get_wall_clock() - t_ch0);

        // Snapshot initial variable node state into DRAM (Core 0, Batch 0 only for debugging)
        if (batch == 0 && core_id == 0) {
            uint64_t vnode_noc_addr = get_noc_addr_from_bank_id<true>(dram_bank, stats_dram_addr + 4096);
            noc_async_write(vnode_l1, vnode_noc_addr, 20480);
            noc_async_write_barrier();
        }

        // Signal channel ready
        sync_flags[SyncSlot::BATCH_READY] = epoch + batch + 1;
        asm volatile("fence rw, rw" ::: "memory");
        WAYPOINT("N_07");

        uint8_t active_mask = 0x0F;
        uint32_t cw_iters[4] = {max_iter, max_iter, max_iter, max_iter};

        // Strict Sequential Lockstep Step Loop: Gather -> Compute -> Scatter
        for (uint32_t step = 0; step < max_steps; ++step) {
            uint32_t bar = batch * max_steps + step + 1;
            uint32_t t_step = step % num_tiles;
            uint32_t iter = step / num_tiles;

            sync_flags[SyncSlot::DEBUG_NCRISC_STEP] = (step << 8) | 1;
            active_mask = static_cast<uint8_t>(sync_flags[SyncSlot::ACTIVE_MASK]);

            // ------------------------------------------------------------------
            // Phase 1: GATHER Stage for Tile t_step (Face 1 & Face 3)
            // ------------------------------------------------------------------
            WAYPOINT("N_WG");
            while (sync_flags[SyncSlot::GATHER_START] < bar) {
                asm volatile("" ::: "memory");
            }
            uint32_t scratch_in_addr = sync_flags[SyncSlot::SCRATCH_GATHER_PTR];

            if (active_mask != 0) {
                uint32_t t_g0 = get_wall_clock();
                #pragma GCC unroll 3
                for (uint32_t d = 0; d < 3; ++d) {
                    gather_slice_l1(scratch_in_addr + 512 + d * 128, QC_LDPC_4CW::TILE_SLICES_W1_F1[t_step][d], vnode_ptr);
                    gather_slice_l1(scratch_in_addr + 1536 + d * 128, QC_LDPC_4CW::TILE_SLICES_W1_F3[t_step][d], vnode_ptr);
                }
                noc_async_read_barrier();
                cycles_gather += (get_wall_clock() - t_g0);
            }

            sync_flags[SyncSlot::GATHER_DONE_W1] = bar;
            asm volatile("fence rw, rw" ::: "memory");
            WAYPOINT("N_GD");

            while (sync_flags[SyncSlot::GATHER_DONE_W0] < bar) {
                asm volatile("" ::: "memory");
            }

            // ------------------------------------------------------------------
            // Phase 2: Wait for Compute to complete and push to cb_scratch_out
            // ------------------------------------------------------------------
            uint32_t t_w0 = get_wall_clock();
            cb_wait_front(cb_scratch_out, 1);
            cycles_wait += (get_wall_clock() - t_w0);

            uint32_t scratch_out_addr = get_read_ptr(cb_scratch_out);
            sync_flags[SyncSlot::SCRATCH_SCATTER_PTR] = scratch_out_addr;
            sync_flags[SyncSlot::SCATTER_START] = bar;
            asm volatile("fence rw, rw" ::: "memory");

            // ------------------------------------------------------------------
            // Phase 3: SCATTER Stage for Tile t_step (Face 1 & Face 3)
            // Blind Unmasked DMA Scatter: All 4 codewords are written unconditionally.
            // Codewords that have converged are protected by having their convergence
            // iteration and 0-error status already locked; no post-convergence errors
            // are ever counted.
            // ------------------------------------------------------------------
            WAYPOINT("N_SC");
            uint32_t t_s0 = get_wall_clock();
            if (active_mask != 0) {
                #pragma GCC unroll 3
                for (uint32_t d = 0; d < 3; ++d) {
                    scatter_slice_l1(QC_LDPC_4CW::TILE_SLICES_W1_F1[t_step][d], scratch_out_addr + 512 + d * 128, vnode_ptr);
                    scatter_slice_l1(QC_LDPC_4CW::TILE_SLICES_W1_F3[t_step][d], scratch_out_addr + 1536 + d * 128, vnode_ptr);
                }
                noc_async_write_barrier();
            }
            cycles_scatter += (get_wall_clock() - t_s0);

            // Wait for Worker 0 (BRISC) to finish scatter before popping
            while (sync_flags[SyncSlot::SCATTER_DONE_W0] < bar) {
                asm volatile("" ::: "memory");
            }
            cb_pop_front(cb_scratch_out, 1);
            uint32_t t5 = get_wall_clock();
            sync_flags[SyncSlot::T5_SCATTER_DONE] = t5;

            uint32_t t0 = sync_flags[SyncSlot::T0_GATHER_START];
            uint32_t t1 = sync_flags[SyncSlot::T1_GATHER_DONE];
            uint32_t t2 = sync_flags[SyncSlot::T2_UNPACK_DONE];
            uint32_t t3 = sync_flags[SyncSlot::T3_MATH_DONE];
            uint32_t t4 = sync_flags[SyncSlot::T4_PACK_DONE];

            stage_gather_acc  += (t1 - t0);
            stage_unpack_acc  += (t2 - t1);
            stage_math_acc    += (t3 - t2);
            stage_pack_acc    += (t4 - t3);
            stage_scatter_acc += (t5 - t4);
            stage_total_acc   += (t5 - t0);
            total_tiles_sim++;

            if (step == 0) {
                stats[100] = t0;
                stats[101] = t1;
                stats[102] = t2;
                stats[103] = t3;
                stats[104] = t4;
                stats[105] = t5;
            }

            // Iteration-Boundary Parity & Termination Check
#if !LDPC_TEST_PASSTHROUGH_MODE
            if (t_step == num_tiles - 1) {
                uint32_t info_errs = sync_flags[SyncSlot::PARITY_REPORT_W0];
                for (uint32_t cw = 0; cw < 4; ++cw) {
                    if ((active_mask & (1 << cw)) && !(info_errs & (1 << cw))) {
                        cw_iters[cw] = iter + 1;
                        active_mask &= ~(1 << cw);
                    }
                }
                sync_flags[SyncSlot::ACTIVE_MASK] = active_mask;
                if (active_mask == 0) {
                    sync_flags[SyncSlot::EARLY_TERMINATION] = 1;
                }
                asm volatile("fence rw, rw" ::: "memory");
            }
#endif

            sync_flags[SyncSlot::SCATTER_DONE_W1] = bar;
            asm volatile("fence rw, rw" ::: "memory");
            WAYPOINT("N_SD");

            // Early termination check at iteration boundary
            if (t_step == num_tiles - 1) {
                if (sync_flags[SyncSlot::EARLY_TERMINATION] != 0) {
                    break;
                }
            }
        }

        // Final Classification / PassThrough Verification
#if LDPC_TEST_PASSTHROUGH_MODE
        uint32_t mismatches = 0;
        for (uint32_t v = 0; v < 2560; ++v) {
            for (uint32_t cw = 0; cw < 4; ++cw) {
                float exp_val = 1024.0f + static_cast<float>(v) + 0.1f * cw;
                uint16_t exp_bf = fp32_to_bf16(exp_val);
                if (vnodes[v].cw[cw] != exp_bf) {
                    mismatches++;
                }
            }
        }
        total_be += mismatches;
        if (mismatches > 0) total_fe++;
        total_iters += num_tiles;
#else
        uint32_t batch_fe = 0;
        uint32_t batch_be = 0;
        for (uint32_t cw = 0; cw < 4; ++cw) {
            uint32_t cw_errors = 0;
            // Convergence Latching Invariant:
            // Only codewords that NEVER converged during the batch have active_mask & (1 << cw) != 0.
            // Any codeword observed to converge on Tile 39 had its bit cleared from active_mask and its
            // cw_iters[cw] frozen at the iteration of convergence. Errors are NEVER measured or counted
            // for converged codewords, completely protecting against any subsequent-iteration divergence
            // while allowing blind unmasked DMA scatter to run at full hardware line-rate.
            if (active_mask & (1 << cw)) {
                for (uint32_t k = 0; k < 1024; ++k) {
                    uint16_t bf = vnodes[k].cw[cw];
                    if (bf & 0x8000) cw_errors++;
                }
            }
            if (cw_errors > 0) batch_fe++;
            batch_be += cw_errors;
            total_iters += cw_iters[cw];
        }
        total_be += batch_be;
        total_fe += batch_fe;

        if (batch_fe > 0 && first_failing_batch == 0xFFFFFFFF) {
            first_failing_batch = batch;
            first_failing_be = batch_be;
            first_failing_mask = active_mask;
            for (uint32_t cw = 0; cw < 4; ++cw) {
                if (active_mask & (1 << cw)) {
                    first_failing_cw = cw;
                    break;
                }
            }
        }
        // Snapshot final variable node state into DRAM (Core 0, Batch 0 or first failing batch)
        if (core_id == 0 && (batch == 0 || first_failing_batch == batch)) {
            uint64_t vnode_final_noc_addr = get_noc_addr_from_bank_id<true>(dram_bank, stats_dram_addr + 4096 + 20480);
            noc_async_write(vnode_l1, vnode_final_noc_addr, 20480);
            noc_async_write_barrier();
        }

#if !LDPC_TEST_PASSTHROUGH_MODE
        cb_pop_front(active_cb, 10);
#endif
#endif
    }

    uint32_t c_end = get_wall_clock();
    uint32_t elapsed_cycles = c_end - c_start;

    // Rendezvous before writing final telemetry
    sync_flags[SyncSlot::BATCH_BARRIER_NCRISC] = epoch + num_batches + 1;
    while (sync_flags[SyncSlot::BATCH_BARRIER_BRISC] < epoch + num_batches + 1) { }
    while (sync_flags[SyncSlot::BATCH_BARRIER_TRISC] < epoch + num_batches + 1) { }

    // Stream telemetry to DRAM
    stats[TelemSlot::TOTAL_BE]          = total_be;
    stats[TelemSlot::TOTAL_FE]          = total_fe;
    stats[TelemSlot::TOTAL_ITERS]       = total_iters;
    stats[TelemSlot::TOTAL_CWS]         = num_batches * 4;
    stats[TelemSlot::ELAPSED_CYCLES]    = elapsed_cycles;
    stats[TelemSlot::CYCLES_PER_BATCH]  = elapsed_cycles / (num_batches > 0 ? num_batches : 1);
    stats[TelemSlot::TRISC_CYCLES]      = sync_flags[SyncSlot::TRISC_CYCLES];
    stats[TelemSlot::TRISC_MATH_CYCLES] = sync_flags[SyncSlot::TRISC_MATH_CYCLES];
    stats[TelemSlot::BRISC_CYCLES]      = sync_flags[SyncSlot::BRISC_CYCLES];
    stats[TelemSlot::CYCLES_CHANNEL]    = cycles_channel;
    stats[TelemSlot::CYCLES_ERROR_SCAN] = cycles_error_scan;
    stats[TelemSlot::TOTAL_TILES_SIM]   = total_tiles_sim;

    stats[TelemSlot::NC_GATHER]         = cycles_gather;
    stats[TelemSlot::NC_WAIT]           = cycles_wait;
    stats[TelemSlot::NC_SCATTER]        = cycles_scatter;

    stats[TelemSlot::STAGE_GATHER_CYCLES]  = stage_gather_acc;
    stats[TelemSlot::STAGE_UNPACK_CYCLES]  = stage_unpack_acc;
    stats[TelemSlot::STAGE_MATH_CYCLES]    = stage_math_acc;
    stats[TelemSlot::STAGE_PACK_CYCLES]    = stage_pack_acc;
    stats[TelemSlot::STAGE_SCATTER_CYCLES] = stage_scatter_acc;
    stats[TelemSlot::STAGE_TOTAL_CYCLES]   = stage_total_acc;

    stats[TelemSlot::FIRST_FAILING_BATCH] = first_failing_batch;
    stats[TelemSlot::FIRST_FAILING_CW]    = first_failing_cw;
    stats[TelemSlot::FIRST_FAILING_BE]    = first_failing_be;
    stats[TelemSlot::FIRST_FAILING_MASK]  = first_failing_mask;

    uint64_t dst_noc_addr = get_noc_addr_from_bank_id<true>(dram_bank, stats_dram_addr);
    noc_async_write(stats_l1, dst_noc_addr, 4096);
    noc_async_write_barrier();
}
