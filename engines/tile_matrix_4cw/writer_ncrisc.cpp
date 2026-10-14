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
    uint32_t num_batches     = get_arg_val<uint32_t>(0);
    uint32_t max_iter        = get_arg_val<uint32_t>(1);
    uint32_t num_tiles       = get_arg_val<uint32_t>(2);
    uint32_t stats_dram_addr = get_arg_val<uint32_t>(3);
    uint32_t dram_bank       = get_arg_val<uint32_t>(4);
    uint32_t stats_bytes     = get_arg_val<uint32_t>(5);
    uint32_t mu_bits         = get_arg_val<uint32_t>(6);
    uint32_t sigma_bits      = get_arg_val<uint32_t>(7);
    uint32_t seed            = get_arg_val<uint32_t>(8);
    uint32_t invocation_id   = get_arg_val<uint32_t>(9);
    uint32_t diff_test_mode  = get_arg_val<uint32_t>(10);
    uint32_t start_batch     = get_arg_val<uint32_t>(11);
    if (invocation_id == 0) invocation_id = 1;
    uint32_t epoch = invocation_id << 16;

    constexpr uint32_t cb_h_offset    = TileMatrixCB::CB_H_OFFSET;
    constexpr uint32_t cb_scratch_out = TileMatrixCB::CB_SCRATCH_SCATTER_0;
    constexpr uint32_t cb_telemetry   = TileMatrixCB::CB_TELEMETRY;

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
    for (uint32_t i = 0; i < 96; ++i) {
        stats[200 + i] = p_hoff_base[i];
    }

    uint32_t total_be = 0;
    uint32_t total_fe = 0;
    uint32_t total_iters = 0;
    uint32_t first_failing_batch = 0xFFFFFFFF;
    uint32_t first_failing_cw    = 0;
    uint32_t first_failing_be    = 0;
    uint32_t first_failing_mask  = 0;

    uint32_t c_start = get_wall_clock();
    uint32_t cycles_gather = 0;
    uint32_t cycles_scatter = 0;
    uint32_t cycles_wait = 0;
    uint32_t cycles_scat_start_bar = 0;
    uint32_t cycles_scat_loop = 0;
    uint32_t cycles_scat_rcopy = 0;
    uint32_t cycles_scat_end_bar = 0;
    uint32_t cycles_scat_pop = 0;
    uint32_t cycles_scat_tile0 = 0;
    uint32_t cycles_scat_other = 0;
    uint32_t global_iter = 0;
    uint32_t step_barrier = 0;

    Xoshiro128Plus rng;
    rng.init(seed);

    bool benchmark_mode = (sigma_bits == 0);
    float mu_llr = uint_as_float(mu_bits);
    float sigma_llr = uint_as_float(sigma_bits);
    uint16_t mu_bf16 = fp32_to_bf16(mu_llr);

    uint32_t cycles_channel = 0;
    uint32_t cycles_error_scan = 0;
    uint32_t cycles_ncrisc_gather_loop = 0;

    constexpr uint32_t active_cb = TileMatrixCB::CB_VNODE_A;
    uint32_t vnode_l1  = get_read_ptr(active_cb);

    // --------------------------------------------------------------------------
    // Main Simulation Loop (Batches x Iterations x Tiles)
    // --------------------------------------------------------------------------
    for (uint32_t batch = 0; batch < num_batches; ++batch) {
        // Wait for TRISC SFPU to finish generating and packing all 10 channel tiles
        uint32_t t_ch0 = get_wall_clock();
        for (uint32_t t = 0; t < 10; ++t) {
            cb_wait_front(active_cb, 1);
            cb_pop_front(active_cb, 1);
        }
        cycles_channel += (get_wall_clock() - t_ch0);

        // Reset early termination flag, active mask, and scan flags BEFORE releasing other workers into the new batch
        sync_flags[SyncSlot::EARLY_TERMINATION] = 0;
        sync_flags[SyncSlot::ACTIVE_MASK]       = 0x0F;
        sync_flags[SyncSlot::BRISC_SCAN_DONE]   = 0;

        uint32_t vnode_l1 = get_read_ptr(active_cb);
        sync_flags[SyncSlot::VNODE_L1_ADDR]     = vnode_l1;

        // Tri-core batch rendezvous barrier (ensures channel LLRs generated by TRISC SFPU are ready)
        sync_flags[SyncSlot::BATCH_BARRIER_NCRISC] = epoch + batch + 1;
        while (sync_flags[SyncSlot::BATCH_BARRIER_BRISC] < epoch + batch + 1) { }
        while (sync_flags[SyncSlot::BATCH_BARRIER_TRISC] < epoch + batch + 1) { }

        const char* vnode_bytes_ptr = reinterpret_cast<const char*>(vnode_l1);
        char* vnode_out_ptr = reinterpret_cast<char*>(vnode_l1);

        uint8_t active_mask = 0x0F;
        uint32_t iter_converged[4] = {max_iter, max_iter, max_iter, max_iter};
        uint32_t cw_errs[4] = {0, 0, 0, 0};

#if LDPC_TEST_PASSTHROUGH_MODE
        // Initialize all 2,560 Vnodes with counting numbers offset by 1024 (1024..3583)
        // This ensures the bfloat16 exponent bits [14:7] are non-zero, preventing hardware FTZ!
        for (uint32_t v = 0; v < 2560; ++v) {
            uint32_t off = get_tile_vnode_byte_offset(v);
            uint16_t* p = reinterpret_cast<uint16_t*>(vnode_l1 + off);
            uint16_t val = static_cast<uint16_t>(1024 + v);
            p[0] = val;
            p[1] = val;
            p[2] = val;
            p[3] = val;
        }
#elif LDPC_TEST_R_DATAFLOW_MODE
        // Initialize all 2,560 Vnodes with exact bfloat16 numbers: (-1)^(v/2) * (2.0 + 0.25 * (v % 8))
        for (uint32_t v = 0; v < 2560; ++v) {
            uint32_t off = get_tile_vnode_byte_offset(v);
            uint16_t* p = reinterpret_cast<uint16_t*>(vnode_l1 + off);
            float sign = ((v / 2) % 2 == 1) ? -1.0f : 1.0f;
            float mag  = 2.0f + 0.25f * static_cast<float>(v % 8);
            float fval = sign * mag;
            uint32_t u = float_as_uint(fval);
            uint16_t val = static_cast<uint16_t>(u >> 16);
            p[0] = val;
            p[1] = val;
            p[2] = val;
            p[3] = val;
        }
#endif

        // In differential co-verification mode, snapshot channel LLRs to DRAM
        if (diff_test_mode != 0 && batch == 0) {
            if (diff_test_mode == 1) {
                // Mode 1: Deterministic synthetic probing pattern
                for (uint32_t v = 0; v < 2560; ++v) {
                    uint32_t off = get_tile_vnode_byte_offset(v);
                    uint16_t* p = reinterpret_cast<uint16_t*>(vnode_l1 + off);
                    float sign = ((v / 2) % 2 == 1) ? -1.0f : 1.0f;
                    float mag  = 2.0f + 0.25f * static_cast<float>(v % 8);
                    float fval = sign * mag;
                    uint32_t u = float_as_uint(fval);
                    uint16_t val = static_cast<uint16_t>(u >> 16);
                    p[0] = val;
                    p[1] = val;
                    p[2] = val;
                    p[3] = val;
                }
            }
            // In both Mode 1 (synthetic) and Mode 2 (real AWGN noise), snapshot initial LLRs to DRAM
            for (uint32_t chk = 0; chk < 10; ++chk) {
                uint64_t dst_noc_initial = get_noc_addr_from_bank_id<true>(dram_bank, stats_dram_addr + 4096 + chk * 2048);
                noc_async_write(vnode_l1 + chk * 2048, dst_noc_initial, 2048);
            }
            noc_async_write_barrier();
        }

        if (batch == 0) {
            uint32_t init_neg[4] = {0, 0, 0, 0};
            const auto* vn32_init = reinterpret_cast<const uint32_t*>(vnode_l1);
            for (uint32_t w = 0; w < 1024; ++w) {
                uint32_t w0 = vn32_init[4 * w + 0];
                uint32_t w1 = vn32_init[4 * w + 1];
                uint32_t w2 = vn32_init[4 * w + 2];
                uint32_t w3 = vn32_init[4 * w + 3];

                init_neg[0] += ((w0 >> 15) & 1) + (w0 >> 31);
                init_neg[1] += ((w1 >> 15) & 1) + (w1 >> 31);
                init_neg[2] += ((w2 >> 15) & 1) + (w2 >> 31);
                init_neg[3] += ((w3 >> 15) & 1) + (w3 >> 31);
            }
            stats[TelemSlot::INIT_NEG_CW0 + 0] = init_neg[0];
            stats[TelemSlot::INIT_NEG_CW0 + 1] = init_neg[1];
            stats[TelemSlot::INIT_NEG_CW0 + 2] = init_neg[2];
            stats[TelemSlot::INIT_NEG_CW0 + 3] = init_neg[3];
            stats[TelemSlot::INIT_BF16_W01] = (static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(vnode_bytes_ptr + get_tile_vnode_byte_offset(0))[0]) << 16) |
                         reinterpret_cast<const uint16_t*>(vnode_bytes_ptr + get_tile_vnode_byte_offset(1))[0];
            stats[TelemSlot::INIT_BF16_W23] = (static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(vnode_bytes_ptr + get_tile_vnode_byte_offset(2))[0]) << 16) |
                         reinterpret_cast<const uint16_t*>(vnode_bytes_ptr + get_tile_vnode_byte_offset(3))[0];
            stats[TelemSlot::MU_BITS] = mu_bits;
            stats[TelemSlot::SIGMA_BITS] = sigma_bits;

            // Initial vnodes already dumped above under diff_test_mode
        }

        // Signal all workers that channel observations / synthetic pattern are 100% ready in L1
        sync_flags[SyncSlot::BATCH_READY] = epoch + batch + 1;

        for (uint32_t iter = 0; iter < max_iter; ++iter) {
            global_iter++;
            for (uint32_t t = 0; t < num_tiles; ++t) {
                step_barrier++;
                const uint32_t* tile_hoff_w1 = p_hoff_base + t * 96 + 48;

                // --------------------------------------------------------------
                // STAGE 1: 4-WORKER PARALLEL GATHER (Worker 1: Face 1)
                // --------------------------------------------------------------
                uint32_t t_g0 = get_wall_clock();
                while (sync_flags[SyncSlot::GATHER_START] < step_barrier) { }
                uint32_t scratch_in_addr = sync_flags[SyncSlot::SCRATCH_GATHER_PTR];

                // Worker 1 gathers 24 pairs into Face 1
                uint32_t t_g1 = get_wall_clock();
                gather_face_24pairs(scratch_in_addr + 512, tile_hoff_w1, vnode_bytes_ptr);
                uint32_t t_g2 = get_wall_clock();
                cycles_ncrisc_gather_loop += (t_g2 - t_g1);

                // Signal Worker 1 gather is complete
                sync_flags[SyncSlot::GATHER_DONE_W1] = step_barrier;
                cycles_gather += (get_wall_clock() - t_g0);
                if (diff_test_mode != 0 && iter == (max_iter - 1) && t == 0) {
                    while (sync_flags[SyncSlot::GATHER_ALL_DONE] < step_barrier) { }
                    const auto* p_nc_scat = reinterpret_cast<const uint32_t*>(scratch_in_addr);
                    stats[332] = p_nc_scat[0];
                    stats[333] = p_nc_scat[1];
                    stats[334] = scratch_in_addr;
                    uint64_t dst_noc_scat_in = get_noc_addr_from_bank_id<true>(dram_bank, stats_dram_addr + 47104);
                    noc_async_write(scratch_in_addr, dst_noc_scat_in, 2048);
                    noc_async_write_barrier();
                }

                // --------------------------------------------------------------
                // STAGE 3: 4-WORKER PARALLEL SCATTER (Worker 1: Face 1)
                // --------------------------------------------------------------
                uint32_t t_w0 = get_wall_clock();
                cb_wait_front(cb_scratch_out, 1);
                cycles_wait += (get_wall_clock() - t_w0);

                uint32_t t_s0 = get_wall_clock();
                uint32_t scratch_out_addr = get_read_ptr(cb_scratch_out);
                if (diff_test_mode != 0 && iter == (max_iter - 1) && t == 0) {
                    uint64_t dst_noc_scat_out = get_noc_addr_from_bank_id<true>(dram_bank, stats_dram_addr + 45056);
                    noc_async_write(scratch_out_addr, dst_noc_scat_out, 2048);
                    noc_async_write_barrier();
                }
                sync_flags[SyncSlot::SCRATCH_SCATTER_PTR] = scratch_out_addr;
                sync_flags[SyncSlot::SCATTER_START] = step_barrier;

                uint32_t t_s1 = get_wall_clock();
                cycles_scat_start_bar += (t_s1 - t_s0);

                // Worker 1 scatters 24 pairs from Face 1 back to variable nodes
                scatter_face_24pairs(scratch_out_addr + 512, tile_hoff_w1, vnode_out_ptr);
                uint32_t t_s2 = get_wall_clock();
                uint32_t scat_loop_cur = (t_s2 - t_s1);
                cycles_scat_loop += scat_loop_cur;
                if (t == 0) cycles_scat_tile0 += scat_loop_cur;
                else cycles_scat_other += scat_loop_cur;

                // Wait for Workers 0, 2, 3 to finish scatter
                while (sync_flags[SyncSlot::SCATTER_DONE_W0] < step_barrier) { }
                while (sync_flags[SyncSlot::SCATTER_DONE_T0] < step_barrier) { }
                while (sync_flags[SyncSlot::SCATTER_DONE_T2] < step_barrier) { }
                uint32_t t_s4 = get_wall_clock();
                cycles_scat_end_bar += (t_s4 - t_s2);

                cb_pop_front(cb_scratch_out, 1);
                uint32_t t_s5 = get_wall_clock();
                cycles_scat_pop += (t_s5 - t_s4);

                cycles_scatter += (t_s5 - t_s0);
            }

            // Dedicated Parallel Transmitted Variable Node Error Scan
            // Dedicated Parallel Transmitted Variable Node Error Scan
            // Worker 1 (NCRISC) scans Parity Nodes 1024..2047 (Tiles 4..7: bytes 8192..16383, 2048 uint32 words)
            // Codewords that have already converged are masked out (scan disabled)
            uint32_t t_es0 = get_wall_clock();
            uint32_t mask_err01 = ((active_mask & 1) ? 0x00008000 : 0) | ((active_mask & 2) ? 0x80000000 : 0);
            uint32_t mask_err23 = ((active_mask & 4) ? 0x00008000 : 0) | ((active_mask & 8) ? 0x80000000 : 0);

            const auto* p1 = reinterpret_cast<const uint32_t*>(vnode_bytes_ptr + 8192);
            uint32_t ncrisc_err01 = 0;
            uint32_t ncrisc_err23 = 0;
            if (mask_err01 != 0 || mask_err23 != 0) {
                #pragma GCC unroll 0
                for (uint32_t i = 0; i < 2048; i += 16) {
                    if (mask_err01) {
                        ncrisc_err01 |= (p1[i + 0] | p1[i + 2] | p1[i + 4] | p1[i + 6] |
                                         p1[i + 8] | p1[i + 10] | p1[i + 12] | p1[i + 14]);
                    }
                    if (mask_err23) {
                        ncrisc_err23 |= (p1[i + 1] | p1[i + 3] | p1[i + 5] | p1[i + 7] |
                                         p1[i + 9] | p1[i + 11] | p1[i + 13] | p1[i + 15]);
                    }
                }
            }
            // Wait for Worker 0 (BRISC) to complete scanning Information Nodes 0..1023
            while (sync_flags[SyncSlot::BRISC_SCAN_DONE] < epoch + global_iter) { }
            uint32_t total_err_cw01 = (ncrisc_err01 & mask_err01) | sync_flags[SyncSlot::BRISC_GATHER_ERR_CW01];
            uint32_t total_err_cw23 = (ncrisc_err23 & mask_err23) | sync_flags[SyncSlot::BRISC_GATHER_ERR_CW23];
            cycles_error_scan += (get_wall_clock() - t_es0);

            // Independent Codeword Convergence Check ONLY for unconverged codewords
            // CW0: bit 15 of word 0
            if ((active_mask & 1) && (total_err_cw01 & 0x00008000) == 0) {
                iter_converged[0] = iter + 1;
                active_mask &= ~(1 << 0);
            }
            // CW1: bit 31 of word 0
            if ((active_mask & 2) && (total_err_cw01 & 0x80000000) == 0) {
                iter_converged[1] = iter + 1;
                active_mask &= ~(1 << 1);
            }
            // CW2: bit 15 of word 1
            if ((active_mask & 4) && (total_err_cw23 & 0x00008000) == 0) {
                iter_converged[2] = iter + 1;
                active_mask &= ~(1 << 2);
            }
            // CW3: bit 31 of word 1
            if ((active_mask & 8) && (total_err_cw23 & 0x80000000) == 0) {
                iter_converged[3] = iter + 1;
                active_mask &= ~(1 << 3);
            }

            // Broadcast active_mask to Worker 0 (BRISC) for next iteration
            sync_flags[SyncSlot::ACTIVE_MASK] = active_mask;

            // Early Termination: Release batch when all 4 codewords converge (disabled in diff_test)
            if (diff_test_mode == 0 && active_mask == 0) {
                sync_flags[SyncSlot::EARLY_TERMINATION] = 1;
                sync_flags[SyncSlot::ITER_DONE] = epoch + global_iter;
                break;
            }
            // Signal BRISC and TRISC that iteration check is complete
            sync_flags[SyncSlot::ITER_DONE] = epoch + global_iter;
        }

        // Residual Bit Error Check on Information Bits ONLY for unconverged codewords
        cw_errs[0] = 0; cw_errs[1] = 0; cw_errs[2] = 0; cw_errs[3] = 0;
        if (active_mask != 0) {
            for (uint32_t v = 0; v < 1024; ++v) {
                uint32_t off = get_tile_vnode_byte_offset(v);
                const auto* p32 = reinterpret_cast<const uint32_t*>(vnode_bytes_ptr + off);
                uint32_t w0 = p32[0];
                uint32_t w1 = p32[1];
                if (active_mask & 1) cw_errs[0] += ((w0 >> 15) & 1);
                if (active_mask & 2) cw_errs[1] += (w0 >> 31);
                if (active_mask & 4) cw_errs[2] += ((w1 >> 15) & 1);
                if (active_mask & 8) cw_errs[3] += (w1 >> 31);
            }
        }

        for (int cw = 0; cw < 4; ++cw) {
            stats[TelemSlot::LAST_BATCH_BE_CW0 + cw] = cw_errs[cw];
            stats[TelemSlot::LAST_BATCH_ITER_CW0 + cw] = iter_converged[cw];
            total_be += cw_errs[cw];
            if (cw_errs[cw] > 0) {
                total_fe++;
                if (first_failing_batch == 0xFFFFFFFF) {
                    first_failing_batch = start_batch + batch;
                    first_failing_cw    = cw;
                    first_failing_be    = cw_errs[cw];
                    first_failing_mask  = (cw_errs[0] > 0 ? 1 : 0) | (cw_errs[1] > 0 ? 2 : 0) | (cw_errs[2] > 0 ? 4 : 0) | (cw_errs[3] > 0 ? 8 : 0);
                }
            }
            total_iters += iter_converged[cw];
        }
    }

    uint32_t c_end = get_wall_clock();
    uint32_t elapsed_cycles = c_end - c_start;

    // Wait for BRISC and TRISC to finish cycle measurements
    sync_flags[SyncSlot::BATCH_BARRIER_NCRISC] = epoch + num_batches + 1;
    while (sync_flags[SyncSlot::BATCH_BARRIER_BRISC] < epoch + num_batches + 1) { }
    while (sync_flags[SyncSlot::BATCH_BARRIER_TRISC] < epoch + num_batches + 1) { }

    const char* vnode_bytes_ptr = reinterpret_cast<const char*>(get_read_ptr(TileMatrixCB::CB_VNODE_A));
    uint32_t slot0 = TelemSlot::FAILING_CW0_BASE, slot1 = TelemSlot::FAILING_CW1_BASE;
    for (uint32_t v = 0; v < 1024; ++v) {
        uint32_t off = get_tile_vnode_byte_offset(v);
        const uint16_t* cw_ptrs = reinterpret_cast<const uint16_t*>(vnode_bytes_ptr + off);
        if ((cw_ptrs[0] & 0x8000) && slot0 < (TelemSlot::FAILING_CW0_BASE + 32)) stats[slot0++] = v;
        if ((cw_ptrs[1] & 0x8000) && slot1 < (TelemSlot::FAILING_CW1_BASE + 32)) stats[slot1++] = v;
    }
    stats[TelemSlot::FAILING_CW0_COUNT] = slot0 - TelemSlot::FAILING_CW0_BASE;
    stats[TelemSlot::FAILING_CW1_COUNT] = slot1 - TelemSlot::FAILING_CW1_BASE;

    // --------------------------------------------------------------------------
    // Step 3: Stream Final Telemetry to DRAM via NoC DMA
    // --------------------------------------------------------------------------
    stats[TelemSlot::TOTAL_BE]          = total_be;
    stats[TelemSlot::TOTAL_FE]          = total_fe;
    stats[TelemSlot::TOTAL_ITERS]       = total_iters;
    stats[TelemSlot::TOTAL_CWS]         = num_batches * 4;
    stats[TelemSlot::ELAPSED_CYCLES]    = elapsed_cycles;
    stats[TelemSlot::CYCLES_PER_BATCH]  = elapsed_cycles / (num_batches > 0 ? num_batches : 1);
    stats[TelemSlot::TRISC_CYCLES]      = sync_flags[SyncSlot::TRISC_CYCLES];
    stats[TelemSlot::BRISC_CYCLES]      = sync_flags[SyncSlot::BRISC_CYCLES];
    stats[TelemSlot::CYCLES_CHANNEL]    = cycles_channel;
    stats[TelemSlot::CYCLES_ERROR_SCAN] = cycles_error_scan;
    stats[TelemSlot::TOTAL_TILES_SIM]     = global_iter * num_tiles;
    stats[TelemSlot::TELEMETRY_L1_ADDR]   = stats_l1;
    stats[TelemSlot::FIRST_FAILING_BATCH] = first_failing_batch;
    stats[TelemSlot::FIRST_FAILING_CW]    = first_failing_cw;
    stats[TelemSlot::FIRST_FAILING_BE]    = first_failing_be;
    stats[TelemSlot::FIRST_FAILING_MASK]  = first_failing_mask;

    // NCRISC (Worker 1) per-tile breakdown
    stats[TelemSlot::NC_GATHER]         = cycles_gather;
    stats[TelemSlot::NC_WAIT]           = cycles_wait;
    stats[TelemSlot::NC_SCATTER]        = cycles_scatter;
    stats[TelemSlot::NC_SCAT_START_BAR] = cycles_scat_start_bar;
    stats[TelemSlot::NC_SCAT_LOOP]      = cycles_scat_loop;
    stats[TelemSlot::NC_SCAT_RCOPY]     = cycles_scat_rcopy;
    stats[TelemSlot::NC_SCAT_END_BAR]   = cycles_scat_end_bar;
    stats[TelemSlot::NC_SCAT_POP]       = cycles_scat_pop;
    stats[TelemSlot::NC_SCAT_TILE0]     = cycles_scat_tile0;
    stats[TelemSlot::NC_SCAT_OTHER]     = cycles_scat_other;
    stats[TelemSlot::NC_GATHER_LOOP]    = cycles_ncrisc_gather_loop;

    // Note: BRISC (Worker 0) per-tile breakdown is already written into stats[TelemSlot::BR_*] (48..53) by BRISC!

#if LDPC_TEST_PASSTHROUGH_MODE
    uint32_t mismatches = 0;
    uint32_t first_bad_v = 0;
    uint32_t first_bad_val = 0;
    for (uint32_t v = 0; v < 2560; ++v) {
        uint32_t off = get_tile_vnode_byte_offset(v);
        const uint16_t* p = reinterpret_cast<const uint16_t*>(vnode_l1 + off);
        for (int c = 0; c < 4; ++c) {
            if (p[c] != static_cast<uint16_t>(1024 + v)) {
                if (mismatches == 0) {
                    first_bad_v = v;
                    first_bad_val = p[c];
                }
                mismatches++;
            }
        }
    }
    stats[200] = mismatches;
    stats[201] = first_bad_v;
    stats[202] = first_bad_val;
#endif

    uint64_t dst_noc_addr = get_noc_addr_from_bank_id<true>(dram_bank, stats_dram_addr);
    noc_async_write(stats_l1, dst_noc_addr, 4096);
    if (diff_test_mode != 0) {
        for (uint32_t chk = 0; chk < 10; ++chk) {
            uint64_t dst_noc_final = get_noc_addr_from_bank_id<true>(dram_bank, stats_dram_addr + 4096 + 20480 + chk * 2048);
            noc_async_write(vnode_l1 + chk * 2048, dst_noc_final, 2048);
        }
    }
    noc_async_write_barrier();
}
