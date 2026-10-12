#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

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
    uint32_t h_dram_addr     = get_arg_val<uint32_t>(0);
    uint32_t h_dram_bank     = get_arg_val<uint32_t>(1);
    uint32_t vnode_dram_addr = get_arg_val<uint32_t>(2);
    uint32_t vnode_dram_bank = get_arg_val<uint32_t>(3);
    uint32_t stats_dram_addr = get_arg_val<uint32_t>(4);
    uint32_t stats_dram_bank = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_vnode    = tt::CBIndex::c_0;   // 20 KB
    constexpr uint32_t cb_h_offset = tt::CBIndex::c_2;   // 16 KB
    constexpr uint32_t cb_scratch  = tt::CBIndex::c_3;   // 4 KB
    constexpr uint32_t cb_stats    = tt::CBIndex::c_16;  // 2 KB
    constexpr uint32_t cb_telem    = tt::CBIndex::c_17;  // 2 KB Sync flags

    uint32_t vnode_l1    = get_read_ptr(cb_vnode);
    uint32_t h_offset_l1 = get_read_ptr(cb_h_offset);
    uint32_t scratch_l1  = get_write_ptr(cb_scratch);
    uint32_t stats_l1    = get_write_ptr(cb_stats);
    uint32_t telem_l1    = get_write_ptr(cb_telem);

    volatile uint32_t* sync = reinterpret_cast<volatile uint32_t*>(telem_l1);
    sync[0] = 0; sync[1] = 0; // Reset barriers

    // Worker 0 loads data from DRAM into L1 SRAM
    constexpr uint32_t h_bytes     = 16 * 1024;
    constexpr uint32_t vnode_bytes = 20 * 1024;
    uint64_t h_noc = get_noc_addr_from_bank_id<true>(h_dram_bank, h_dram_addr);
    noc_async_read(h_noc, h_offset_l1, h_bytes);
    uint64_t v_noc = get_noc_addr_from_bank_id<true>(vnode_dram_bank, vnode_dram_addr);
    noc_async_read(v_noc, vnode_l1, vnode_bytes);
    noc_async_read_barrier();

    const char* vnode_bytes_ptr = reinterpret_cast<const char*>(vnode_l1);
    const uint32_t* p_hoff_base = reinterpret_cast<const uint32_t*>(h_offset_l1);
    uint32_t* results = reinterpret_cast<uint32_t*>(stats_l1);

    // Worker 0 processes Checks 0..19: Edge Pairs 0..59 (60 pairs)
    constexpr uint32_t PAIRS_PER_WORKER = 60;
    const uint32_t* p_vnode_ptrs0[PAIRS_PER_WORKER * 2];
    for (uint32_t p = 0; p < PAIRS_PER_WORKER; ++p) {
        uint32_t off = p_hoff_base[p];
        p_vnode_ptrs0[2 * p + 0] = reinterpret_cast<const uint32_t*>(vnode_bytes_ptr + (off & 0xFFFF));
        p_vnode_ptrs0[2 * p + 1] = reinterpret_cast<const uint32_t*>(vnode_bytes_ptr + (off >> 16));
    }

    // Signal Worker 1 that setup is complete
    sync[0] = 1;
    while (sync[1] == 0) { /* wait for Worker 1 ready */ }

    // Worker 0 writes into byte partition [0..959] of the tile
    auto* dst32_base = reinterpret_cast<uint32_t*>(scratch_l1);
    constexpr uint32_t NUM_ITERATIONS = 1000;

    uint32_t t_start = get_wall_clock();
    for (uint32_t it = 0; it < NUM_ITERATIONS; ++it) {
        const uint32_t* const* ptrs = p_vnode_ptrs0;
        uint32_t* dst32 = dst32_base;

        // 7 chunks of 8 pairs = 56 pairs
        for (uint32_t chunk = 0; chunk < 7; ++chunk) {
            #pragma GCC unroll 8
            for (int k = 0; k < 8; ++k) {
                const uint32_t* pe = ptrs[2 * k + 0];
                const uint32_t* po = ptrs[2 * k + 1];
                uint32_t ve0 = pe[0], vo0 = po[0], ve1 = pe[1], vo1 = po[1];
                dst32[4 * k + 0] = rv32_pack(ve0, vo0);
                dst32[4 * k + 1] = rv32_pack(ve0 >> 16, vo0 >> 16);
                dst32[4 * k + 2] = rv32_pack(ve1, vo1);
                dst32[4 * k + 3] = rv32_pack(ve1 >> 16, vo1 >> 16);
            }
            ptrs += 16;
            dst32 += 32;
        }
        // Tail 4 pairs
        #pragma GCC unroll 4
        for (int k = 0; k < 4; ++k) {
            const uint32_t* pe = ptrs[2 * k + 0];
            const uint32_t* po = ptrs[2 * k + 1];
            uint32_t ve0 = pe[0], vo0 = po[0], ve1 = pe[1], vo1 = po[1];
            dst32[4 * k + 0] = rv32_pack(ve0, vo0);
            dst32[4 * k + 1] = rv32_pack(ve0 >> 16, vo0 >> 16);
            dst32[4 * k + 2] = rv32_pack(ve1, vo1);
            dst32[4 * k + 3] = rv32_pack(ve1 >> 16, vo1 >> 16);
        }
    }
    uint32_t t_end = get_wall_clock();
    results[0] = (t_end - t_start) / NUM_ITERATIONS;

    // Wait for Worker 1 to finish
    while (sync[2] == 0) { /* wait for Worker 1 done */ }

    // Write Worker 0 result back to DRAM
    uint64_t stats_noc = get_noc_addr_from_bank_id<true>(stats_dram_bank, stats_dram_addr);
    noc_async_write(stats_l1, stats_noc, 16);
    noc_async_write_barrier();
}
