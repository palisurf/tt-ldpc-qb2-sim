#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ldpc_tile_matrix_core.h"

// ==============================================================================
// Working 440 code 8CW pipelined simulation, still needs orchestration optimization and FER performance analysis
//
// Tenstorrent QuietBox 2: Tile Matrix 4-Codeword LDPC Decoder Engine
// Component: Data Movement Kernel 0 (BRISC - Worker 0: Dual Gather & Dual Scatter)
// Architecture: Centralized TRISC1 Maestro with Bounded Satellite Autonomy (DAE)
// Target: Tenstorrent Blackhole (440 Tensix Cores, 2,200 RISC-V Processors)
// Specification: docs/MD_files/CENTRALIZED_TRISC1_ORCHESTRATOR_AND_BOUNDED_AUTONOMY_SPECIFICATION.md
//
// Key Responsibilities:
// 1. Face 0 Gather & Scatter: Autonomously gathers Face 0 (lanes 0..15, checks 0..3)
//    and scatters updated posterior LLRs directly into L1 SRAM variable node arrays.
// 2. Information Node Parity Scan: On Tile 39, scans sign bits across 1,024 info
//    bits for CW0..3 and reports the 4-bit error mask to SyncSlot::PARITY_REPORT_W0.
// 3. Bounded Autonomy: Executes all intra-tile memory operations autonomously, but
//    gates step advancement and pair launch strictly on TRISC1_GRANT_STEP and TRISC1_CMD.
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
    uint32_t total_steps = max_steps + 2;

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

        // Satellite Pre-Step Gate: Wait for TRISC1 Maestro to launch Step 0
        while (sync_flags[SyncSlot::TRISC1_CMD] != static_cast<uint32_t>(MaestroCmd::CMD_STEP_EXEC) ||
               sync_flags[SyncSlot::TRISC1_GRANT_STEP] < pair * total_steps + 1) { }

        uint32_t vnode_l1 = sync_flags[SyncSlot::VNODE_L1_BASE_ADDR];
        const char* vnode_bytes_ptr = reinterpret_cast<const char*>(vnode_l1);
        char* vnode_out_ptr = reinterpret_cast<char*>(vnode_l1);

        // ----------------------------------------------------------------------
        // Unified Monotonic Macro-Step Pipeline (Bounded Autonomy)
        // ----------------------------------------------------------------------
        for (uint32_t k = 0; k < total_steps; ++k) {
            uint32_t bar = pair * total_steps + k + 1;

            // 1. Wait for TRISC1 Macro Step Grant
            while (sync_flags[SyncSlot::TRISC1_GRANT_STEP] < bar) {
                if (sync_flags[SyncSlot::TRISC1_CMD] >= static_cast<uint32_t>(MaestroCmd::CMD_DRAIN_RESIDUAL)) {
                    break;
                }
            }
            if (sync_flags[SyncSlot::TRISC1_CMD] >= static_cast<uint32_t>(MaestroCmd::CMD_DRAIN_RESIDUAL)) {
                break;
            }

            // 2. Autonomous Scatter Stage: Scatter previous tile (k - 2) for both Batch A and Batch B
            if (k >= 2 && (k - 2) < max_steps) {
                uint32_t t_w0 = get_wall_clock();
                while (sync_flags[SyncSlot::SCATTER_START] < bar) { }
                cycles_brisc_wait += (get_wall_clock() - t_w0);

                uint32_t t_s0 = get_wall_clock();
                uint32_t scratch_out_addr_a = sync_flags[SyncSlot::SCRATCH_SCATTER_PTR_A];
                uint32_t scratch_out_addr_b = sync_flags[SyncSlot::SCRATCH_SCATTER_PTR_B];

                uint32_t s_scat = k - 2;
                uint32_t t_scat = s_scat % num_tiles;
                const uint32_t* tile_hoff_w0 = p_hoff_base + t_scat * 96;

                // Worker 0 scatters Face 0 (lanes 0..15, rows 0..15) back to variable nodes with 128-bit vector stores
                scatter_face_128bit(scratch_out_addr_a + 0, scratch_out_addr_b + 0, tile_hoff_w0 + 0, vnode_out_ptr, 1);
                sync_flags[SyncSlot::SCATTER_DONE_W0] = bar;
                cycles_scatter += (get_wall_clock() - t_s0);

                // Autonomous Check node iteration completion scan on Information Nodes (Tile 39) for CW 0..3 (Batch A)
                if (t_scat == num_tiles - 1) {
                    uint8_t active_mask_a = static_cast<uint8_t>(sync_flags[SyncSlot::ACTIVE_MASK_A]);
                    uint32_t mask_err01 = ((active_mask_a & 1) ? 0x00008000 : 0) | ((active_mask_a & 2) ? 0x80000000 : 0);
                    uint32_t mask_err23 = ((active_mask_a & 4) ? 0x00008000 : 0) | ((active_mask_a & 8) ? 0x80000000 : 0);

                    const auto* p_v = reinterpret_cast<const VNode128*>(vnode_bytes_ptr);
                    uint32_t brisc_err01 = 0;
                    uint32_t brisc_err23 = 0;
                    if (mask_err01 != 0 || mask_err23 != 0) {
                        #pragma GCC unroll 4
                        for (uint32_t i = 0; i < 1024; ++i) {
                            uint64_t w_a = p_v[i].batch_a();
                            uint32_t w0 = static_cast<uint32_t>(w_a);
                            uint32_t w1 = static_cast<uint32_t>(w_a >> 32);
                            brisc_err01 |= w0;
                            brisc_err23 |= w1;
                            if ((rv32_andn(mask_err01, brisc_err01) | rv32_andn(mask_err23, brisc_err23)) == 0) {
                                break;
                            }
                        }
                    }
                    uint32_t info_err_mask = 0;
                    if ((brisc_err01 & 0x00008000) && (active_mask_a & 1)) info_err_mask |= 1;
                    if ((brisc_err01 & 0x80000000) && (active_mask_a & 2)) info_err_mask |= 2;
                    if ((brisc_err23 & 0x00008000) && (active_mask_a & 4)) info_err_mask |= 4;
                    if ((brisc_err23 & 0x80000000) && (active_mask_a & 8)) info_err_mask |= 8;
                    sync_flags[SyncSlot::PARITY_REPORT_W0] = info_err_mask;
                }
            }

            // 3. Autonomous Gather Stage: Gather tile k for both Batch A and Batch B
            if (k < max_steps) {
                uint32_t t_g0 = get_wall_clock();
                cb_reserve_back(cb_scratch_in_a, 1);
                cb_reserve_back(cb_scratch_in_b, 1);
                uint32_t scratch_in_addr_a = get_write_ptr(cb_scratch_in_a);
                uint32_t scratch_in_addr_b = get_write_ptr(cb_scratch_in_b);

                sync_flags[SyncSlot::SCRATCH_GATHER_PTR_A] = scratch_in_addr_a;
                sync_flags[SyncSlot::SCRATCH_GATHER_PTR_B] = scratch_in_addr_b;
                sync_flags[SyncSlot::GATHER_START] = bar;

                uint32_t s_gath = k;
                uint32_t t_gath = s_gath % num_tiles;
                const uint32_t* tile_hoff_w0 = p_hoff_base + t_gath * 96;
                gather_face_128bit(scratch_in_addr_a + 0, scratch_in_addr_b + 0, tile_hoff_w0 + 0, vnode_bytes_ptr, 1);

                while (sync_flags[SyncSlot::GATHER_DONE_W1] < bar) { }
                while (sync_flags[SyncSlot::GATHER_DONE_T0] < bar) { }
                while (sync_flags[SyncSlot::GATHER_DONE_T2] < bar) { }
                cb_push_back(cb_scratch_in_a, 1);
                cb_push_back(cb_scratch_in_b, 1);
                sync_flags[SyncSlot::GATHER_DONE_W0] = bar;
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
