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

    uint32_t b_start = get_wall_clock();
    uint32_t cycles_gather = 0;
    uint32_t cycles_scatter = 0;
    uint32_t cycles_brisc_wait = 0;
    uint32_t cycles_brisc_scat_start_bar = 0;
    uint32_t cycles_brisc_scat_loop = 0;
    uint32_t cycles_brisc_scat_rcopy = 0;
    uint32_t cycles_brisc_scat_end_bar = 0;
    uint32_t step_barrier = 0;
    uint32_t global_iter = 0;

    // --------------------------------------------------------------------------
    // Step 2: Main Simulation Loop (Batches x Iterations x Tiles)
    // --------------------------------------------------------------------------
    for (uint32_t batch = 0; batch < num_batches; ++batch) {
        // Tri-core batch rendezvous barrier (ensures all cores start the batch together)
        sync_flags[SyncSlot::BATCH_BARRIER_BRISC] = epoch + batch + 1;
        while (sync_flags[SyncSlot::BATCH_BARRIER_NCRISC] < epoch + batch + 1) { }
        while (sync_flags[SyncSlot::BATCH_BARRIER_TRISC] < epoch + batch + 1) { }

        constexpr uint32_t active_cb = TileMatrixCB::CB_VNODE_A;
        uint32_t vnode_l1  = get_read_ptr(active_cb);
        const char* vnode_bytes_ptr = reinterpret_cast<const char*>(vnode_l1);
        char* vnode_out_ptr = reinterpret_cast<char*>(vnode_l1);

        uint8_t active_mask = 0x0F;

        for (uint32_t iter = 0; iter < max_iter; ++iter) {
            global_iter++;
            active_mask = static_cast<uint8_t>(sync_flags[SyncSlot::ACTIVE_MASK]);
            for (uint32_t t = 0; t < num_tiles; ++t) {
                step_barrier++;
                const uint32_t* tile_hoff_w0 = p_hoff_base + t * 96;

                // --------------------------------------------------------------
                // STAGE 1: DUAL GATHER (Worker 0: Faces 0 & 2)
                // --------------------------------------------------------------
                uint32_t t_g0 = get_wall_clock();
                cb_reserve_back(cb_scratch_in, 1);
                uint32_t scratch_in_addr = get_write_ptr(cb_scratch_in);

                // Share scratch pointer with Worker 1 and signal ready
                sync_flags[SyncSlot::SCRATCH_GATHER_PTR] = scratch_in_addr;
                sync_flags[SyncSlot::GATHER_BARRIER_W0] = step_barrier;
                while (sync_flags[SyncSlot::GATHER_BARRIER_W1] != step_barrier) { }

                // Worker 0 gathers 48 pairs into Faces 0 & 2 using Compile-Time LUT
                #pragma GCC unroll 8
                for (int p = 0; p < 48; ++p) {
                    uint32_t off = tile_hoff_w0[p];
                    uint32_t off_even = off & 0xFFFF;
                    uint32_t off_odd  = off >> 16;
                    uint32_t s_even = SCRATCH_FACE_LUT.w0_even[p];
                    uint32_t s_odd  = SCRATCH_FACE_LUT.w0_odd[p];

                    *reinterpret_cast<uint64_t*>(scratch_in_addr + s_even) =
                        *reinterpret_cast<const uint64_t*>(vnode_bytes_ptr + off_even);
                    *reinterpret_cast<uint64_t*>(scratch_in_addr + s_odd) =
                        *reinterpret_cast<const uint64_t*>(vnode_bytes_ptr + off_odd);
                }

                // Zero out unused padding rows (rows 12..15 of Face 0 & Face 2)
                auto* pad_f0 = reinterpret_cast<uint32_t*>(scratch_in_addr + 384);
                auto* pad_f2 = reinterpret_cast<uint32_t*>(scratch_in_addr + 1408);
                #pragma GCC unroll 8
                for (int i = 0; i < 32; ++i) {
                    pad_f0[i] = 0;
                    pad_f2[i] = 0;
                }

                // Wait for Worker 1 to finish gather
                sync_flags[SyncSlot::GATHER_BARRIER_W0] = step_barrier + 0x80000000;
                while (sync_flags[SyncSlot::GATHER_BARRIER_W1] != (step_barrier + 0x80000000)) { }
                cb_push_back(cb_scratch_in, 1);
                cycles_gather += (get_wall_clock() - t_g0);

                // --------------------------------------------------------------
                // STAGE 3: DUAL SCATTER (Worker 0: Faces 0 & 2)
                // --------------------------------------------------------------
                uint32_t t_wait = get_wall_clock();
                while (sync_flags[SyncSlot::SCATTER_BARRIER_W1] != step_barrier) { }
                cycles_brisc_wait += (get_wall_clock() - t_wait);

                uint32_t t_s0 = get_wall_clock();
                uint32_t scratch_out_addr = sync_flags[SyncSlot::SCRATCH_SCATTER_PTR];
                sync_flags[SyncSlot::SCATTER_BARRIER_W0] = step_barrier;
                uint32_t t_s1 = get_wall_clock();
                cycles_brisc_scat_start_bar += (t_s1 - t_s0);

                // Worker 0 scatters 48 pairs from Faces 0 & 2 back to variable nodes
                if (active_mask == 0x0F) {
                    #pragma GCC unroll 8
                    for (int p = 0; p < 48; ++p) {
                        uint32_t off = tile_hoff_w0[p];
                        uint32_t off_even = off & 0xFFFF;
                        uint32_t off_odd  = off >> 16;
                        uint32_t s_even = SCRATCH_FACE_LUT.w0_even[p];
                        uint32_t s_odd  = SCRATCH_FACE_LUT.w0_odd[p];

                        *reinterpret_cast<uint64_t*>(vnode_out_ptr + off_even) =
                            *reinterpret_cast<const uint64_t*>(scratch_out_addr + s_even);
                        *reinterpret_cast<uint64_t*>(vnode_out_ptr + off_odd) =
                            *reinterpret_cast<const uint64_t*>(scratch_out_addr + s_odd);
                    }
                } else {
                    uint32_t mask_low  = ((active_mask & 1) ? 0x0000FFFF : 0) | ((active_mask & 2) ? 0xFFFF0000 : 0);
                    uint32_t mask_high = ((active_mask & 4) ? 0x0000FFFF : 0) | ((active_mask & 8) ? 0xFFFF0000 : 0);
                    uint32_t inv_low   = ~mask_low;
                    uint32_t inv_high  = ~mask_high;

                    #pragma GCC unroll 8
                    for (int p = 0; p < 48; ++p) {
                        uint32_t off = tile_hoff_w0[p];
                        uint32_t off_even = off & 0xFFFF;
                        uint32_t off_odd  = off >> 16;
                        uint32_t s_even = SCRATCH_FACE_LUT.w0_even[p];
                        uint32_t s_odd  = SCRATCH_FACE_LUT.w0_odd[p];

                        auto* d_even = reinterpret_cast<uint32_t*>(vnode_out_ptr + off_even);
                        const auto* s_ev = reinterpret_cast<const uint32_t*>(scratch_out_addr + s_even);
                        d_even[0] = (s_ev[0] & mask_low)  | (d_even[0] & inv_low);
                        d_even[1] = (s_ev[1] & mask_high) | (d_even[1] & inv_high);

                        auto* d_odd = reinterpret_cast<uint32_t*>(vnode_out_ptr + off_odd);
                        const auto* s_od = reinterpret_cast<const uint32_t*>(scratch_out_addr + s_odd);
                        d_odd[0] = (s_od[0] & mask_low)  | (d_odd[0] & inv_low);
                        d_odd[1] = (s_od[1] & mask_high) | (d_odd[1] & inv_high);
                    }
                }
                uint32_t t_s2 = get_wall_clock();
                cycles_brisc_scat_loop += (t_s2 - t_s1);

                // Signal Worker 1 that Worker 0 scatter is complete
                sync_flags[SyncSlot::SCATTER_BARRIER_W0] = step_barrier + 0x80000000;
                while (sync_flags[SyncSlot::SCATTER_BARRIER_W1] != (step_barrier + 0x80000000)) { }
                uint32_t t_s4 = get_wall_clock();
                cycles_brisc_scat_end_bar += (t_s4 - t_s2);

                cycles_scatter += (t_s4 - t_s0);
            }

            // Wait for Worker 1 (NCRISC) to complete the iteration convergence check
            while (sync_flags[SyncSlot::ITER_DONE] < epoch + global_iter) { }

            // Early Termination check: Release batch when all 4 codewords converge
            if (sync_flags[SyncSlot::EARLY_TERMINATION]) {
                break;
            }
        }
    }

    uint32_t b_end = get_wall_clock();
    sync_flags[SyncSlot::BRISC_CYCLES]     = b_end - b_start;
    sync_flags[TelemSlot::BR_GATHER]       = cycles_gather;
    sync_flags[TelemSlot::BR_WAIT]         = cycles_brisc_wait;
    sync_flags[TelemSlot::BR_SCATTER]      = cycles_scatter;
    sync_flags[TelemSlot::BR_SCAT_LOOP]    = cycles_brisc_scat_loop;
    sync_flags[TelemSlot::BR_SCAT_RCOPY]   = cycles_brisc_scat_rcopy;
    sync_flags[TelemSlot::BR_SCAT_END_BAR] = cycles_brisc_scat_end_bar;
    sync_flags[SyncSlot::BATCH_BARRIER_BRISC] = epoch + num_batches + 1;
}
