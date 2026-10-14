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
    uint32_t cycles_brisc_gather_loop = 0;
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
            active_mask = (iter == 0) ? 0x0F : static_cast<uint8_t>(sync_flags[SyncSlot::ACTIVE_MASK]);
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

                // Worker 0 gathers 48 pairs into Faces 0 & 2 via sequential 128-bit transfers
                uint32_t t_g1 = get_wall_clock();
                for (uint32_t f = 0; f < 2; ++f) {
                    uint32_t face_offset = (f == 0) ? 0 : 1024;
                    auto* dst_face = reinterpret_cast<uint32_t*>(scratch_in_addr + face_offset);
                    const uint32_t* p_hoff = tile_hoff_w0 + f * 24;

                    for (uint32_t d = 0; d < 6; ++d) {
                        auto* dst_even = dst_face + d * 16;
                        auto* dst_odd  = dst_even + 8;
                        const uint32_t* p_h = p_hoff + d * 4;

                        // Block 0: Pairs 0 and 1 (8 words: 4 even + 4 odd)
                        {
                            uint32_t off0 = p_h[0];
                            uint32_t off1 = p_h[1];

                            uint32_t off0_e = off0 & 0xFFFF;
                            uint32_t off0_o = off0 >> 16;
                            uint32_t off1_e = off1 & 0xFFFF;
                            uint32_t off1_o = off1 >> 16;

                            const auto* s0_e = reinterpret_cast<const uint32_t*>(vnode_bytes_ptr + off0_e);
                            const auto* s0_o = reinterpret_cast<const uint32_t*>(vnode_bytes_ptr + off0_o);
                            const auto* s1_e = reinterpret_cast<const uint32_t*>(vnode_bytes_ptr + off1_e);
                            const auto* s1_o = reinterpret_cast<const uint32_t*>(vnode_bytes_ptr + off1_o);

                            uint32_t e0 = s0_e[0]; uint32_t e1 = s0_e[1];
                            uint32_t e2 = s1_e[0]; uint32_t e3 = s1_e[1];
                            uint32_t o0 = s0_o[0]; uint32_t o1 = s0_o[1];
                            uint32_t o2 = s1_o[0]; uint32_t o3 = s1_o[1];

                            dst_even[0] = e0; dst_even[1] = e1; dst_even[2] = e2; dst_even[3] = e3;
                            dst_odd[0]  = o0; dst_odd[1]  = o1; dst_odd[2]  = o2; dst_odd[3]  = o3;
                        }

                        // Block 1: Pairs 2 and 3 (8 words: 4 even + 4 odd)
                        {
                            uint32_t off2 = p_h[2];
                            uint32_t off3 = p_h[3];

                            uint32_t off2_e = off2 & 0xFFFF;
                            uint32_t off2_o = off2 >> 16;
                            uint32_t off3_e = off3 & 0xFFFF;
                            uint32_t off3_o = off3 >> 16;

                            const auto* s2_e = reinterpret_cast<const uint32_t*>(vnode_bytes_ptr + off2_e);
                            const auto* s2_o = reinterpret_cast<const uint32_t*>(vnode_bytes_ptr + off2_o);
                            const auto* s3_e = reinterpret_cast<const uint32_t*>(vnode_bytes_ptr + off3_e);
                            const auto* s3_o = reinterpret_cast<const uint32_t*>(vnode_bytes_ptr + off3_o);

                            uint32_t e4 = s2_e[0]; uint32_t e5 = s2_e[1];
                            uint32_t e6 = s3_e[0]; uint32_t e7 = s3_e[1];
                            uint32_t o4 = s2_o[0]; uint32_t o5 = s2_o[1];
                            uint32_t o6 = s3_o[0]; uint32_t o7 = s3_o[1];

                            dst_even[4] = e4; dst_even[5] = e5; dst_even[6] = e6; dst_even[7] = e7;
                            dst_odd[4]  = o4; dst_odd[5]  = o5; dst_odd[6]  = o6; dst_odd[7]  = o7;
                        }
                    }
                }
                uint32_t t_g2 = get_wall_clock();
                cycles_brisc_gather_loop += (t_g2 - t_g1);

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
                if (__builtin_expect(active_mask == 0x0F, 1)) {
                    // Straight-line pipelined fast path (0 branches, 0 stack spills)
                    for (uint32_t f = 0; f < 2; ++f) {
                        uint32_t face_offset = (f == 0) ? 0 : 1024;
                        const auto* src_face = reinterpret_cast<const uint32_t*>(scratch_out_addr + face_offset);
                        const uint32_t* p_hoff = tile_hoff_w0 + f * 24;

                        for (uint32_t d = 0; d < 6; ++d) {
                            const auto* src_even = src_face + d * 16;
                            const auto* src_odd  = src_even + 8;
                            const uint32_t* p_h = p_hoff + d * 4;

                            #pragma GCC unroll 4
                            for (uint32_t p = 0; p < 4; ++p) {
                                uint32_t off = p_h[p];
                                uint32_t e0 = src_even[2 * p];
                                uint32_t e1 = src_even[2 * p + 1];
                                uint32_t o0 = src_odd[2 * p];
                                uint32_t o1 = src_odd[2 * p + 1];
                                auto* d_e = reinterpret_cast<uint32_t*>(vnode_out_ptr + (off & 0xFFFF));
                                auto* d_o = reinterpret_cast<uint32_t*>(vnode_out_ptr + (off >> 16));
                                d_e[0] = e0; d_e[1] = e1;
                                d_o[0] = o0; d_o[1] = o1;
                            }
                        }
                    }
                } else {
                    // Slow path: masked scatter for frozen codewords
                    uint32_t mask_low  = ((active_mask & 1) ? 0x0000FFFF : 0) | ((active_mask & 2) ? 0xFFFF0000 : 0);
                    uint32_t mask_high = ((active_mask & 4) ? 0x0000FFFF : 0) | ((active_mask & 8) ? 0xFFFF0000 : 0);
                    bool inv_low   = (mask_low  != 0xFFFFFFFF);
                    bool inv_high  = (mask_high != 0xFFFFFFFF);

                    for (uint32_t f = 0; f < 2; ++f) {
                        uint32_t face_offset = (f == 0) ? 0 : 1024;
                        const auto* src_face = reinterpret_cast<const uint32_t*>(scratch_out_addr + face_offset);
                        const uint32_t* p_hoff = tile_hoff_w0 + f * 24;

                        for (uint32_t d = 0; d < 6; ++d) {
                            const auto* src_even = src_face + d * 16;
                            const auto* src_odd  = src_even + 8;
                            const uint32_t* p_h = p_hoff + d * 4;

                            #pragma GCC unroll 4
                            for (uint32_t p = 0; p < 4; ++p) {
                                uint32_t off = p_h[p];
                                auto* d_e = reinterpret_cast<uint32_t*>(vnode_out_ptr + (off & 0xFFFF));
                                auto* d_o = reinterpret_cast<uint32_t*>(vnode_out_ptr + (off >> 16));
                                uint32_t e0 = src_even[2 * p];
                                uint32_t e1 = src_even[2 * p + 1];
                                uint32_t o0 = src_odd[2 * p];
                                uint32_t o1 = src_odd[2 * p + 1];

                                if (inv_low)  { e0 = (e0 & mask_low)  | (d_e[0] & ~mask_low);  o0 = (o0 & mask_low)  | (d_o[0] & ~mask_low); }
                                if (inv_high) { e1 = (e1 & mask_high) | (d_e[1] & ~mask_high); o1 = (o1 & mask_high) | (d_o[1] & ~mask_high); }

                                d_e[0] = e0; d_e[1] = e1;
                                d_o[0] = o0; d_o[1] = o1;
                            }
                        }
                    }
                }
                uint32_t t_s2 = get_wall_clock();
                cycles_brisc_scat_loop += (t_s2 - t_s1);

                sync_flags[SyncSlot::SCATTER_BARRIER_W0] = step_barrier + 0x80000000;
                while (sync_flags[SyncSlot::SCATTER_BARRIER_W1] != (step_barrier + 0x80000000)) { }
                uint32_t t_s4 = get_wall_clock();
                cycles_brisc_scat_end_bar += (t_s4 - t_s2);

                cycles_scatter += (t_s4 - t_s0);
            }

            // Dedicated Parallel Transmitted Variable Node Error Scan
            // Worker 0 (BRISC) scans Information Nodes 0..1023 (Tiles 0..3: bytes 0..8191, 2048 uint32 words)
            const auto* p0 = reinterpret_cast<const uint32_t*>(vnode_bytes_ptr);
            uint32_t brisc_err01 = 0;
            uint32_t brisc_err23 = 0;
            #pragma GCC unroll 0
            for (uint32_t i = 0; i < 2048; i += 16) {
                brisc_err01 |= (p0[i + 0] | p0[i + 2] | p0[i + 4] | p0[i + 6] |
                                p0[i + 8] | p0[i + 10] | p0[i + 12] | p0[i + 14]);
                brisc_err23 |= (p0[i + 1] | p0[i + 3] | p0[i + 5] | p0[i + 7] |
                                p0[i + 9] | p0[i + 11] | p0[i + 13] | p0[i + 15]);
            }
            sync_flags[SyncSlot::BRISC_GATHER_ERR_CW01] = brisc_err01 & 0x80008000;
            sync_flags[SyncSlot::BRISC_GATHER_ERR_CW23] = brisc_err23 & 0x80008000;
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
    sync_flags[SyncSlot::BRISC_CYCLES]     = b_end - b_start;
    sync_flags[TelemSlot::BR_GATHER]       = cycles_gather;
    sync_flags[TelemSlot::BR_WAIT]         = cycles_brisc_wait;
    sync_flags[TelemSlot::BR_SCATTER]      = cycles_scatter;
    sync_flags[TelemSlot::BR_SCAT_LOOP]    = cycles_brisc_scat_loop;
    sync_flags[TelemSlot::BR_SCAT_RCOPY]   = cycles_brisc_scat_rcopy;
    sync_flags[TelemSlot::BR_SCAT_END_BAR] = cycles_brisc_scat_end_bar;
    sync_flags[TelemSlot::BR_GATHER_LOOP]  = cycles_brisc_gather_loop;
    sync_flags[SyncSlot::BATCH_BARRIER_BRISC] = epoch + num_batches + 1;
}
