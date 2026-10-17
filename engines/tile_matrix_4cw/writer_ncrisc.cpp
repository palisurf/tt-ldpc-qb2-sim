#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ldpc_tile_matrix_core.h"

// ==============================================================================
// Tenstorrent QuietBox 2: Tile Matrix 4-Codeword LDPC Decoder Engine
// Component: Data Movement Kernel 1 (NCRISC - Worker 1: Dual Gather & Dual Scatter)
// Architecture: Decoupled Access-Execute (DAE) with Native 32x32 Tensix Tiles
// Target: Tenstorrent Blackhole (440 Tensix Cores, 2,200 RISC-V Processors)
// ==============================================================================

static inline uint32_t get_wall_clock() {
    volatile uint32_t* const wall_clock_reg = reinterpret_cast<volatile uint32_t*>(0xFFB121F0);
    return *wall_clock_reg;
}

static inline uint32_t rv32_pack(uint32_t a, uint32_t b) {
    uint32_t res;
    asm volatile(
        ".option push\n\t"
        ".option arch, +zbkb\n\t"
        "pack %0, %1, %2\n\t"
        ".option pop"
        : "=r"(res) : "r"(a), "r"(b)
    );
    return res;
}

void kernel_main() {
    uint32_t num_batch_pairs = get_arg_val<uint32_t>(0); // Number of 8-codeword pairs
    uint32_t max_iter        = get_arg_val<uint32_t>(1);
    uint32_t num_tiles       = get_arg_val<uint32_t>(2);
    uint32_t stats_dram_addr = get_arg_val<uint32_t>(3);
    uint32_t dram_bank       = get_arg_val<uint32_t>(4);
    uint32_t stats_bytes     = get_arg_val<uint32_t>(5);
    uint32_t mu_bits         = get_arg_val<uint32_t>(6);
    uint32_t sigma_bits      = get_arg_val<uint32_t>(7);
    uint32_t global_seed     = get_arg_val<uint32_t>(8);
    uint32_t invocation_id   = get_arg_val<uint32_t>(9);
    uint32_t diff_test_mode  = get_arg_val<uint32_t>(10);
    uint32_t start_batch     = get_arg_val<uint32_t>(11);
    uint32_t core_id         = get_arg_val<uint32_t>(12);
    if (invocation_id == 0) invocation_id = 1;
    uint32_t epoch = invocation_id << 16;

    constexpr uint32_t cb_h_offset      = TileMatrixCB::CB_H_OFFSET;
    constexpr uint32_t cb_scratch_out_a = TileMatrixCB::CB_SCRATCH_SCATTER_0;
    constexpr uint32_t cb_scratch_out_b = TileMatrixCB::CB_SCRATCH_SCATTER_1;
    constexpr uint32_t cb_vnode_a       = TileMatrixCB::CB_VNODE_A;
    constexpr uint32_t cb_vnode_b       = TileMatrixCB::CB_VNODE_B;
    constexpr uint32_t cb_telemetry     = TileMatrixCB::CB_TELEMETRY;

    uint32_t h_offset_l1 = get_read_ptr(cb_h_offset);
    uint32_t stats_l1    = get_write_ptr(cb_telemetry);

    volatile uint32_t* sync_flags = reinterpret_cast<volatile uint32_t*>(stats_l1);
    auto* stats = reinterpret_cast<uint32_t*>(stats_l1);

    // Zero out all sync flags, mailboxes, and telemetry to guarantee pristine startup state
    for (uint32_t i = 0; i < 1024; ++i) {
        sync_flags[i] = 0;
    }

    // Announce NCRISC ready for this epoch
    sync_flags[63] = epoch;

    // Wait until Worker 0 (BRISC) has loaded H matrix from DRAM into L1 SRAM
    while (sync_flags[0] < epoch) { }

    const uint32_t* p_hoff_base = reinterpret_cast<const uint32_t*>(h_offset_l1);
    const uint16_t* meta_ptr = reinterpret_cast<const uint16_t*>(h_offset_l1 + TOTAL_TILES_PER_ITER * 96 * sizeof(uint32_t));
    for (uint32_t t = 0; t < num_tiles; ++t) {
        sync_flags[64 + t] = meta_ptr[t * 2 + 1];
    }

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
        // Wait for TRISC SFPU to finish generating and packing all 10 channel tiles for Batch A & B
        uint32_t t_ch0 = get_wall_clock();
        for (uint32_t t = 0; t < 10; ++t) {
            cb_wait_front(cb_vnode_a, 1);
            cb_pop_front(cb_vnode_a, 1);
            cb_wait_front(cb_vnode_b, 1);
            cb_pop_front(cb_vnode_b, 1);
        }
        cycles_channel += (get_wall_clock() - t_ch0);

        if (pair == 0) {
            const auto* p0 = reinterpret_cast<const uint32_t*>(vnode_l1_a);
            stats[TelemSlot::INIT_BF16_W01] = p0[0];
            stats[TelemSlot::INIT_BF16_W23] = p0[1];
            stats[TelemSlot::MU_BITS]       = mu_bits;
            stats[TelemSlot::SIGMA_BITS]    = sigma_bits;
            uint32_t neg0 = 0, neg1 = 0, neg2 = 0, neg3 = 0;
            for (uint32_t i = 0; i < 2048; i += 2) {
                uint32_t w0 = p0[i + 0];
                uint32_t w1 = p0[i + 1];
                neg0 += ((w0 >> 15) & 1);
                neg1 += (w0 >> 31);
                neg2 += ((w1 >> 15) & 1);
                neg3 += (w1 >> 31);
            }
            stats[TelemSlot::INIT_NEG_CW0 + 0] = neg0;
            stats[TelemSlot::INIT_NEG_CW0 + 1] = neg1;
            stats[TelemSlot::INIT_NEG_CW0 + 2] = neg2;
            stats[TelemSlot::INIT_NEG_CW0 + 3] = neg3;
        }

        // Reset early termination flag, active masks, and scan flags
        sync_flags[SyncSlot::EARLY_TERMINATION] = 0;
        sync_flags[SyncSlot::ACTIVE_MASK_A]     = 0x0F;
        sync_flags[SyncSlot::ACTIVE_MASK_B]     = 0x0F;
        sync_flags[SyncSlot::BRISC_SCAN_DONE_A] = 0;
        sync_flags[SyncSlot::BRISC_SCAN_DONE_B] = 0;
        sync_flags[SyncSlot::VNODE_L1_ADDR_A]   = vnode_l1_a;
        sync_flags[SyncSlot::VNODE_L1_ADDR_B]   = vnode_l1_b;

        // Tri-core batch rendezvous barrier
        sync_flags[SyncSlot::BATCH_BARRIER_NCRISC] = epoch + pair + 1;
        while (sync_flags[SyncSlot::BATCH_BARRIER_BRISC] < epoch + pair + 1) { }
        while (sync_flags[SyncSlot::BATCH_BARRIER_TRISC] < epoch + pair + 1) { }

        uint8_t active_mask_a = 0x0F;
        uint8_t active_mask_b = 0x0F;
        uint32_t iter_converged_a[4] = {max_iter, max_iter, max_iter, max_iter};
        uint32_t iter_converged_b[4] = {max_iter, max_iter, max_iter, max_iter};
        uint32_t cw_errs_a[4] = {0, 0, 0, 0};
        uint32_t cw_errs_b[4] = {0, 0, 0, 0};

        // Signal all workers that channel observations are 100% ready in L1
        sync_flags[SyncSlot::BATCH_READY] = epoch + pair + 1;

        // ----------------------------------------------------------------------
        // Interleaved Dual-Batch DAE Gather/Scatter Engine (Zero-RAW Direct Store)
        // ----------------------------------------------------------------------
        for (uint32_t k = 0; k < total_steps; ++k) {
            uint32_t bar = pair * total_steps + k + 1;
            uint32_t b_dae = k % 2;
            uint32_t s_dae = k / 2;
            uint32_t vnode_l1 = vnodes[b_dae];
            const char* vnode_bytes_ptr = reinterpret_cast<const char*>(vnode_l1);
            char* vnode_out_ptr = reinterpret_cast<char*>(vnode_l1);
            uint8_t active_mask = (b_dae == 0) ? active_mask_a : active_mask_b;

            // 1. Scatter Stage: Scatter previous tile (s_dae - 1) of Batch b_dae
            if (s_dae >= 1 && (s_dae - 1) < max_steps) {
                uint32_t t_w0 = get_wall_clock();
                uint32_t active_out_cb = cb_scat_out[b_dae];
                cb_wait_front(active_out_cb, 1);
                cycles_wait += (get_wall_clock() - t_w0);

                uint32_t t_s0 = get_wall_clock();
                uint32_t scratch_out_addr = get_read_ptr(active_out_cb);
                sync_flags[SyncSlot::SCRATCH_SCATTER_PTR] = scratch_out_addr;
                sync_flags[SyncSlot::SCATTER_START] = bar;

                uint32_t s_scat = s_dae - 1;
                uint32_t t_scat = s_scat % num_tiles;
                const uint32_t* tile_hoff_w1 = p_hoff_base + t_scat * 96 + 48;

                // Worker 1 scatters 24 pairs (Face 1) back to variable nodes with direct store
                if (active_mask != 0) {
                    scatter_face_24pairs(scratch_out_addr + 512, tile_hoff_w1, vnode_out_ptr);
                }

                while (sync_flags[SyncSlot::SCATTER_DONE_W0] < bar) { }
                while (sync_flags[SyncSlot::SCATTER_DONE_T0] < bar) { }
                while (sync_flags[SyncSlot::SCATTER_DONE_T2] < bar) { }
                cb_pop_front(active_out_cb, 1);
                cycles_scatter += (get_wall_clock() - t_s0);

                // Check node iteration completion scan on Parity Nodes
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

                    uint32_t cur_iter = s_scat / num_tiles;
                    if (b_dae == 0) {
                        while (sync_flags[SyncSlot::BRISC_SCAN_DONE_A] < bar) { }
                        uint32_t total_err_cw01 = (ncrisc_err01 & mask_err01) | sync_flags[SyncSlot::BRISC_GATHER_ERR_CW01_A];
                        uint32_t total_err_cw23 = (ncrisc_err23 & mask_err23) | sync_flags[SyncSlot::BRISC_GATHER_ERR_CW23_A];
                        if ((active_mask_a & 1) && (total_err_cw01 & 0x00008000) == 0) { iter_converged_a[0] = cur_iter + 1; active_mask_a &= ~1; }
                        if ((active_mask_a & 2) && (total_err_cw01 & 0x80000000) == 0) { iter_converged_a[1] = cur_iter + 1; active_mask_a &= ~2; }
                        if ((active_mask_a & 4) && (total_err_cw23 & 0x00008000) == 0) { iter_converged_a[2] = cur_iter + 1; active_mask_a &= ~4; }
                        if ((active_mask_a & 8) && (total_err_cw23 & 0x80000000) == 0) { iter_converged_a[3] = cur_iter + 1; active_mask_a &= ~8; }
                        sync_flags[SyncSlot::ACTIVE_MASK_A] = active_mask_a;
                        if (active_mask_a == 0 && active_mask_b == 0) {
                            sync_flags[SyncSlot::EARLY_TERMINATION] = 1;
                        }
                        sync_flags[SyncSlot::ITER_DONE_A] = bar;
                    } else {
                        while (sync_flags[SyncSlot::BRISC_SCAN_DONE_B] < bar) { }
                        uint32_t total_err_cw01 = (ncrisc_err01 & mask_err01) | sync_flags[SyncSlot::BRISC_GATHER_ERR_CW01_B];
                        uint32_t total_err_cw23 = (ncrisc_err23 & mask_err23) | sync_flags[SyncSlot::BRISC_GATHER_ERR_CW23_B];
                        if ((active_mask_b & 1) && (total_err_cw01 & 0x00008000) == 0) { iter_converged_b[0] = cur_iter + 1; active_mask_b &= ~1; }
                        if ((active_mask_b & 2) && (total_err_cw01 & 0x80000000) == 0) { iter_converged_b[1] = cur_iter + 1; active_mask_b &= ~2; }
                        if ((active_mask_b & 4) && (total_err_cw23 & 0x00008000) == 0) { iter_converged_b[2] = cur_iter + 1; active_mask_b &= ~4; }
                        if ((active_mask_b & 8) && (total_err_cw23 & 0x80000000) == 0) { iter_converged_b[3] = cur_iter + 1; active_mask_b &= ~8; }
                        sync_flags[SyncSlot::ACTIVE_MASK_B] = active_mask_b;
                        if (active_mask_a == 0 && active_mask_b == 0) {
                            sync_flags[SyncSlot::EARLY_TERMINATION] = 1;
                        }
                        sync_flags[SyncSlot::ITER_DONE_B] = bar;
                    }
                    cycles_error_scan += (get_wall_clock() - t_es0);
                }
            }

            // Early termination check: Release batch pair when both batches have converged
            if (sync_flags[SyncSlot::EARLY_TERMINATION]) {
                uint32_t comp_cb_out = (b_dae == 0) ? cb_scratch_out_b : cb_scratch_out_a;
                cb_wait_front(comp_cb_out, 1);
                cb_pop_front(comp_cb_out, 1);
                break;
            }

            // 2. Gather Stage: Gather current tile s_dae of Batch b_dae
            if (s_dae < max_steps) {
                uint32_t t_g0 = get_wall_clock();
                while (sync_flags[SyncSlot::GATHER_START] < bar) { }
                uint32_t scratch_in_addr = sync_flags[SyncSlot::SCRATCH_GATHER_PTR];

                if (active_mask != 0) {
                    total_tiles_sim++;
                    uint32_t t_gath = s_dae % num_tiles;
                    const uint32_t* tile_hoff_w1 = p_hoff_base + t_gath * 96 + 48;
                    gather_face_24pairs(scratch_in_addr + 512, tile_hoff_w1, vnode_bytes_ptr);
                }

                sync_flags[SyncSlot::GATHER_DONE_W1] = bar;
                cycles_gather += (get_wall_clock() - t_g0);
            }
        }

        // Residual Bit Error Check on Information Bits for Unconverged Codewords
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

    // --------------------------------------------------------------------------
    // Step 3: Stream Final Telemetry to DRAM via NoC DMA
    // --------------------------------------------------------------------------
    stats[TelemSlot::TOTAL_BE]          = total_be;
    stats[TelemSlot::TOTAL_FE]          = total_fe;
    stats[TelemSlot::TOTAL_ITERS]       = total_iters;
    stats[TelemSlot::TOTAL_CWS]         = num_batch_pairs * 8;
    stats[TelemSlot::ELAPSED_CYCLES]    = elapsed_cycles;
    stats[TelemSlot::CYCLES_PER_BATCH]  = elapsed_cycles / (num_batch_pairs > 0 ? num_batch_pairs : 1);
    stats[TelemSlot::TRISC_CYCLES]      = sync_flags[SyncSlot::TRISC_CYCLES];
    stats[TelemSlot::BRISC_CYCLES]      = sync_flags[SyncSlot::BRISC_CYCLES];
    stats[TelemSlot::CYCLES_CHANNEL]    = cycles_channel;
    stats[TelemSlot::CYCLES_ERROR_SCAN] = cycles_error_scan;
    stats[TelemSlot::TOTAL_TILES_SIM]   = total_tiles_sim;
    stats[TelemSlot::TELEMETRY_L1_ADDR] = stats_l1;

    stats[TelemSlot::FIRST_FAILING_BATCH] = first_failing_batch;
    stats[TelemSlot::FIRST_FAILING_CW]    = first_failing_cw;
    stats[TelemSlot::FIRST_FAILING_BE]    = first_failing_be;
    stats[TelemSlot::FIRST_FAILING_MASK]  = first_failing_mask;

    // NCRISC (Worker 1) per-tile breakdown
    stats[TelemSlot::NC_GATHER]         = cycles_gather;
    stats[TelemSlot::NC_WAIT]           = cycles_wait;
    stats[TelemSlot::NC_SCATTER]        = cycles_scatter;

    uint64_t dst_noc_addr = get_noc_addr_from_bank_id<true>(dram_bank, stats_dram_addr);
    noc_async_write(stats_l1, dst_noc_addr, 4096);
    noc_async_write_barrier();
}
