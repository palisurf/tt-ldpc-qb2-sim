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
    uint32_t num_batch_pairs = get_arg_val<uint32_t>(0); // Number of 8-codeword pairs
    uint32_t max_iter        = get_arg_val<uint32_t>(1);
    uint32_t num_tiles       = get_arg_val<uint32_t>(2);
    uint32_t h_dram_addr     = get_arg_val<uint32_t>(3);
    uint32_t dram_bank       = get_arg_val<uint32_t>(4);
    uint32_t h_bytes         = get_arg_val<uint32_t>(5);
    uint32_t invocation_id   = get_arg_val<uint32_t>(6);
    uint32_t mu_bits         = get_arg_val<uint32_t>(7);
    uint32_t sigma_bits      = get_arg_val<uint32_t>(8);
    uint32_t global_seed     = get_arg_val<uint32_t>(9);
    uint32_t core_id         = get_arg_val<uint32_t>(10);
    if (invocation_id == 0) invocation_id = 1;
    uint32_t epoch = invocation_id << 16;

    constexpr uint32_t cb_h_offset     = TileMatrixCB::CB_H_OFFSET;
    constexpr uint32_t cb_scratch_in_a = TileMatrixCB::CB_SCRATCH_GATHER_0;
    constexpr uint32_t cb_scratch_in_b = TileMatrixCB::CB_SCRATCH_GATHER_1;
    constexpr uint32_t cb_vnode_a      = TileMatrixCB::CB_VNODE_A;
    constexpr uint32_t cb_vnode_b      = TileMatrixCB::CB_VNODE_B;
    constexpr uint32_t cb_telem        = TileMatrixCB::CB_TELEMETRY;

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
    uint32_t cycles_scatter = 0;
    uint32_t cycles_brisc_wait = 0;
    uint32_t max_steps = max_iter * num_tiles;
    uint32_t total_steps = 2 * max_steps + 2;

    uint32_t cb_scat_in[2] = {cb_scratch_in_a, cb_scratch_in_b};

    // --------------------------------------------------------------------------
    // Step 2: Main Simulation Loop (Dual-Batch Pairs)
    // --------------------------------------------------------------------------
    for (uint32_t pair = 0; pair < num_batch_pairs; ++pair) {
        // Tri-core batch rendezvous barrier
        sync_flags[SyncSlot::BATCH_BARRIER_BRISC] = epoch + pair + 1;
        while (sync_flags[SyncSlot::BATCH_BARRIER_NCRISC] < epoch + pair + 1) { }
        while (sync_flags[SyncSlot::BATCH_BARRIER_TRISC] < epoch + pair + 1) { }

        // Wait for channel observations to be 100% ready in L1
        while (sync_flags[SyncSlot::BATCH_READY] < epoch + pair + 1) { }

        uint32_t vnode_l1_a = sync_flags[SyncSlot::VNODE_L1_ADDR_A];
        uint32_t vnode_l1_b = sync_flags[SyncSlot::VNODE_L1_ADDR_B];
        uint32_t vnodes[2] = {vnode_l1_a, vnode_l1_b};

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
            uint8_t active_mask = static_cast<uint8_t>(
                (b_dae == 0) ? sync_flags[SyncSlot::ACTIVE_MASK_A] : sync_flags[SyncSlot::ACTIVE_MASK_B]
            );

            // 1. Scatter Stage: Scatter previous tile (s_dae - 1) of Batch b_dae
            if (s_dae >= 1 && (s_dae - 1) < max_steps) {
                uint32_t t_w0 = get_wall_clock();
                while (sync_flags[SyncSlot::SCATTER_START] < bar) { }
                cycles_brisc_wait += (get_wall_clock() - t_w0);

                uint32_t t_s0 = get_wall_clock();
                uint32_t scratch_out_addr = sync_flags[SyncSlot::SCRATCH_SCATTER_PTR];

                uint32_t s_scat = s_dae - 1;
                uint32_t t_scat = s_scat % num_tiles;
                const uint32_t* tile_hoff_w0 = p_hoff_base + t_scat * 96;

                // Worker 0 scatters 48 pairs (Faces 0 & 2) back to variable nodes with direct store
                if (active_mask != 0) {
                    scatter_face_48pairs_w0(scratch_out_addr, tile_hoff_w0, vnode_out_ptr);
                }
                sync_flags[SyncSlot::SCATTER_DONE_W0] = bar;
                cycles_scatter += (get_wall_clock() - t_s0);

                // Check node iteration completion scan on Information Nodes
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
                    if (b_dae == 0) {
                        sync_flags[SyncSlot::BRISC_GATHER_ERR_CW01_A] = brisc_err01 & mask_err01;
                        sync_flags[SyncSlot::BRISC_GATHER_ERR_CW23_A] = brisc_err23 & mask_err23;
                        sync_flags[SyncSlot::BRISC_SCAN_DONE_A] = bar;
                        while (sync_flags[SyncSlot::ITER_DONE_A] < bar) { }
                    } else {
                        sync_flags[SyncSlot::BRISC_GATHER_ERR_CW01_B] = brisc_err01 & mask_err01;
                        sync_flags[SyncSlot::BRISC_GATHER_ERR_CW23_B] = brisc_err23 & mask_err23;
                        sync_flags[SyncSlot::BRISC_SCAN_DONE_B] = bar;
                        while (sync_flags[SyncSlot::ITER_DONE_B] < bar) { }
                    }
                }
            }

            // Early termination check: Release batch pair when both batches have converged
            if (sync_flags[SyncSlot::EARLY_TERMINATION]) {
                break;
            }

            // 2. Gather Stage: Gather current tile s_dae of Batch b_dae
            if (s_dae < max_steps) {
                uint32_t t_g0 = get_wall_clock();
                uint32_t active_in_cb = cb_scat_in[b_dae];
                cb_reserve_back(active_in_cb, 1);
                uint32_t scratch_in_addr = get_write_ptr(active_in_cb);

                sync_flags[SyncSlot::SCRATCH_GATHER_PTR] = scratch_in_addr;
                sync_flags[SyncSlot::GATHER_START] = bar;

                if (active_mask != 0) {
                    uint32_t t_gath = s_dae % num_tiles;
                    const uint32_t* tile_hoff_w0 = p_hoff_base + t_gath * 96;
                    gather_face_48pairs_w0(scratch_in_addr, tile_hoff_w0, vnode_bytes_ptr);
                }

                while (sync_flags[SyncSlot::GATHER_DONE_W1] < bar) { }
                cb_push_back(active_in_cb, 1);
                cycles_gather += (get_wall_clock() - t_g0);
            }
        }
    }

    uint32_t b_end = get_wall_clock();
    sync_flags[SyncSlot::BRISC_CYCLES]     = telem_l1;
    sync_flags[TelemSlot::BR_GATHER]       = cycles_gather;
    sync_flags[TelemSlot::BR_WAIT]         = cycles_brisc_wait;
    sync_flags[TelemSlot::BR_SCATTER]      = cycles_scatter;
    sync_flags[SyncSlot::BATCH_BARRIER_BRISC] = epoch + num_batch_pairs + 1;
}
