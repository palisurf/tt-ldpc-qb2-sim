#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ldpc_tile_matrix_core.h"

// ==============================================================================
// Tenstorrent QuietBox 2: Tile Matrix 4-Codeword LDPC Decoder Engine
// Component: Data Movement Kernel 0 (BRISC - Worker 0: Dual Gather & Dual Scatter)
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
    uint32_t num_batches   = get_arg_val<uint32_t>(0);
    uint32_t max_iter      = get_arg_val<uint32_t>(1);
    uint32_t num_tiles     = get_arg_val<uint32_t>(2);
    uint32_t h_dram_addr   = get_arg_val<uint32_t>(3);
    uint32_t dram_bank     = get_arg_val<uint32_t>(4);
    uint32_t h_bytes       = get_arg_val<uint32_t>(5);
    uint32_t invocation_id = get_arg_val<uint32_t>(6);
    uint32_t mu_bits       = get_arg_val<uint32_t>(7);
    uint32_t sigma_bits    = get_arg_val<uint32_t>(8);
    uint32_t seed          = get_arg_val<uint32_t>(9);
    if (invocation_id == 0) invocation_id = 1;
    uint32_t epoch = invocation_id << 16;

    float mu_llr = uint_as_float(mu_bits);
    float sigma_llr = uint_as_float(sigma_bits);

    Xoshiro128Plus rng;
    rng.init(seed);

    constexpr uint32_t cb_h_offset   = TileMatrixCB::CB_H_OFFSET;
    constexpr uint32_t cb_scratch_in = TileMatrixCB::CB_SCRATCH_GATHER_0;
    constexpr uint32_t cb_telem      = TileMatrixCB::CB_TELEMETRY;

    uint32_t h_offset_l1 = get_read_ptr(cb_h_offset);
    uint32_t telem_l1    = get_write_ptr(cb_telem);

    volatile uint32_t* sync_flags = reinterpret_cast<volatile uint32_t*>(telem_l1);

    // --------------------------------------------------------------------------
    // Step 1: Read H matrix offsets + tile schedule metadata from DRAM into CB 2
    // --------------------------------------------------------------------------
    uint64_t src_noc_addr = get_noc_addr_from_bank_id<true>(dram_bank, h_dram_addr);
    noc_async_read(src_noc_addr, h_offset_l1, h_bytes);
    noc_async_read_barrier();

    const uint32_t* p_hoff_base = reinterpret_cast<const uint32_t*>(h_offset_l1);

    // Wait until Worker 1 (NCRISC) has arrived for this epoch
    while (sync_flags[63] < epoch) { }

    // Signal Worker 1 (NCRISC) that H table setup is complete
    sync_flags[0] = epoch;
    sync_flags[SyncSlot::HOFF_L1_ADDR] = h_offset_l1;

    uint32_t b_start = get_wall_clock();
    uint32_t cycles_gather = 0;
    uint32_t cycles_brisc_gather_loop = 0;
    uint32_t cycles_scatter = 0;
    uint32_t cycles_brisc_wait = 0;
    uint32_t cycles_brisc_scat_start_bar = 0;
    uint32_t cycles_brisc_scat_loop = 0;
    uint32_t cycles_brisc_scat_rcopy = 0;
    uint32_t cycles_brisc_scat_end_bar = 0;
    uint32_t global_iter = 0;
    uint32_t step_barrier = 0;

    // --------------------------------------------------------------------------
    // Step 2: Main Simulation Loop (Batches x Iterations x Tiles)
    // --------------------------------------------------------------------------
    for (uint32_t batch = 0; batch < num_batches; ++batch) {
        // Tri-core batch rendezvous barrier (ensures all cores arrive at batch boundary)
        sync_flags[SyncSlot::BATCH_BARRIER_BRISC] = epoch + batch + 1;
        while (sync_flags[SyncSlot::BATCH_BARRIER_NCRISC] < epoch + batch + 1) { }
        while (sync_flags[SyncSlot::BATCH_BARRIER_TRISC] < epoch + batch + 1) { }

        // Wait for channel observations / synthetic pattern to be 100% ready in L1
        while (sync_flags[SyncSlot::BATCH_READY] < epoch + batch + 1) { }

        constexpr uint32_t active_cb = TileMatrixCB::CB_VNODE_A;
        uint32_t vnode_l1 = sync_flags[SyncSlot::VNODE_L1_ADDR];
        const char* vnode_bytes_ptr = reinterpret_cast<const char*>(vnode_l1);
        char* vnode_out_ptr = reinterpret_cast<char*>(vnode_l1);

        for (uint32_t iter = 0; iter < max_iter; ++iter) {
            global_iter++;
            for (uint32_t t = 0; t < num_tiles; ++t) {
                step_barrier++;
                const uint32_t* tile_hoff_w0 = p_hoff_base + t * 96;

                // --------------------------------------------------------------
                // STAGE 1: 4-WORKER PARALLEL GATHER (Worker 0: Face 0)
                // --------------------------------------------------------------
                uint32_t t_g0 = get_wall_clock();
                cb_reserve_back(cb_scratch_in, 1);
                uint32_t scratch_in_addr = get_write_ptr(cb_scratch_in);

                // Publish scratch pointer and broadcast start barrier to Workers 1, 2, 3
                sync_flags[SyncSlot::SCRATCH_GATHER_PTR] = scratch_in_addr;
                sync_flags[SyncSlot::GATHER_START] = step_barrier;

                // Worker 0 gathers 24 pairs into Face 0
                if (iter == 0 && t == 0) {
                    sync_flags[300] = tile_hoff_w0[0];
                    sync_flags[301] = reinterpret_cast<uint32_t>(tile_hoff_w0);
                    sync_flags[302] = reinterpret_cast<uint32_t>(vnode_bytes_ptr);
                    sync_flags[303] = *reinterpret_cast<const uint32_t*>(vnode_bytes_ptr + (tile_hoff_w0[0] & 0xFFFF));
                    sync_flags[304] = *reinterpret_cast<const uint32_t*>(vnode_bytes_ptr + ((tile_hoff_w0[0] & 0xFFFF) + 4));
                    sync_flags[305] = scratch_in_addr;

                    const auto* p_v0 = reinterpret_cast<const uint32_t*>(vnode_bytes_ptr);
                    for (int i = 0; i < 8; ++i) sync_flags[310 + i] = p_v0[i];

                    const auto* p_v571 = reinterpret_cast<const uint32_t*>(vnode_bytes_ptr + 0x10f8);
                    sync_flags[320] = p_v571[0];
                    sync_flags[321] = p_v571[1];

                    const auto* p_v43 = reinterpret_cast<const uint32_t*>(vnode_bytes_ptr + 0x00b8);
                    sync_flags[322] = p_v43[0];
                    sync_flags[323] = p_v43[1];
                }
                uint32_t t_g1 = get_wall_clock();
                gather_face_24pairs(scratch_in_addr + 0, tile_hoff_w0, vnode_bytes_ptr);
                if (iter == 0 && t == 0) {
                    const auto* p_scat = reinterpret_cast<const uint32_t*>(scratch_in_addr);
                    sync_flags[330] = p_scat[0];
                    sync_flags[331] = p_scat[1];
                }
                uint32_t t_g2 = get_wall_clock();
                cycles_brisc_gather_loop += (t_g2 - t_g1);

                // Wait for Workers 1, 2, 3 to finish gather
                while (sync_flags[SyncSlot::GATHER_DONE_W1] < step_barrier) { }
                while (sync_flags[SyncSlot::GATHER_DONE_T0] < step_barrier) { }
                while (sync_flags[SyncSlot::GATHER_DONE_T2] < step_barrier) { }
                sync_flags[SyncSlot::GATHER_ALL_DONE] = step_barrier;
                cb_push_back(cb_scratch_in, 1);
                cycles_gather += (get_wall_clock() - t_g0);

                // --------------------------------------------------------------
                // STAGE 3: 4-WORKER PARALLEL SCATTER (Worker 0: Face 0)
                // --------------------------------------------------------------
                uint32_t t_wait = get_wall_clock();
                while (sync_flags[SyncSlot::SCATTER_START] < step_barrier) { }
                cycles_brisc_wait += (get_wall_clock() - t_wait);

                uint32_t t_s0 = get_wall_clock();
                uint32_t scratch_out_addr = sync_flags[SyncSlot::SCRATCH_SCATTER_PTR];
                uint32_t t_s1 = get_wall_clock();
                cycles_brisc_scat_start_bar += (t_s1 - t_s0);

                // Worker 0 scatters 24 pairs from Face 0 back to variable nodes
                scatter_face_24pairs(scratch_out_addr + 0, tile_hoff_w0, vnode_out_ptr);
                uint32_t t_s2 = get_wall_clock();
                cycles_brisc_scat_loop += (t_s2 - t_s1);

                // Signal Worker 0 scatter is done
                sync_flags[SyncSlot::SCATTER_DONE_W0] = step_barrier;
                cycles_scatter += (t_s2 - t_s0);
            }

            // Dedicated Parallel Transmitted Variable Node Error Scan
            // Worker 0 (BRISC) scans Information Nodes 0..1023 (Tiles 0..3: bytes 0..8191, 2048 uint32 words)
            // Codewords that have already converged are masked out (scan disabled)
            uint8_t active_mask = static_cast<uint8_t>(sync_flags[SyncSlot::ACTIVE_MASK]);
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
                    // Fast short-circuit: Once all active codewords have shown >= 1 bit error,
                    // none can converge in this iteration. Stop scanning immediately!
                    if ((rv32_andn(mask_err01, brisc_err01) | rv32_andn(mask_err23, brisc_err23)) == 0) {
                        break;
                    }
                }
            }
            sync_flags[SyncSlot::BRISC_GATHER_ERR_CW01] = brisc_err01 & mask_err01;
            sync_flags[SyncSlot::BRISC_GATHER_ERR_CW23] = brisc_err23 & mask_err23;
            sync_flags[SyncSlot::BRISC_SCAN_DONE] = epoch + global_iter;

            // Wait for Worker 1 (NCRISC) to complete the iteration convergence check
            while (sync_flags[SyncSlot::ITER_DONE] < epoch + global_iter) { }

            // Early Termination check: Release batch when all 4 codewords converge
            if (sync_flags[SyncSlot::EARLY_TERMINATION]) {
                break;
            }
        }
    }

    uint32_t b_end = get_wall_clock();
    sync_flags[SyncSlot::BRISC_CYCLES]     = telem_l1;
    sync_flags[TelemSlot::BR_GATHER]       = cycles_gather;
    sync_flags[TelemSlot::BR_WAIT]         = cycles_brisc_wait;
    sync_flags[TelemSlot::BR_SCATTER]      = cycles_scatter;
    sync_flags[TelemSlot::BR_SCAT_LOOP]    = cycles_brisc_scat_loop;
    sync_flags[TelemSlot::BR_SCAT_RCOPY]   = cycles_brisc_scat_rcopy;
    sync_flags[TelemSlot::BR_SCAT_END_BAR] = cycles_brisc_scat_end_bar;
    sync_flags[TelemSlot::BR_GATHER_LOOP]  = cycles_brisc_gather_loop;
    sync_flags[SyncSlot::BATCH_BARRIER_BRISC] = epoch + num_batches + 1;
}
