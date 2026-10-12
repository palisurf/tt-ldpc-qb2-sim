#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

static inline uint32_t get_wall_clock() {
    volatile uint32_t* const wall_clock_reg = reinterpret_cast<volatile uint32_t*>(0xFFB121F0);
    return *wall_clock_reg;
}

void kernel_main() {
    uint32_t stats_dram_addr = get_arg_val<uint32_t>(0);
    uint32_t stats_dram_bank = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_vnode    = tt::CBIndex::c_0;   // 20 KB
    constexpr uint32_t cb_h_offset = tt::CBIndex::c_2;   // 16 KB
    constexpr uint32_t cb_scratch  = tt::CBIndex::c_16;  // 4 KB (Scatter Input)
    constexpr uint32_t cb_stats    = tt::CBIndex::c_17;  // 2 KB Output stats
    constexpr uint32_t cb_telem    = tt::CBIndex::c_18;  // 2 KB Sync flags

    uint32_t vnode_l1    = get_write_ptr(cb_vnode);
    uint32_t h_offset_l1 = get_read_ptr(cb_h_offset);
    uint32_t scratch_l1  = get_read_ptr(cb_scratch);
    uint32_t stats_l1    = get_write_ptr(cb_stats);
    uint32_t telem_l1    = get_write_ptr(cb_telem);

    volatile uint32_t* sync = reinterpret_cast<volatile uint32_t*>(telem_l1);

    char* vnode_bytes_ptr = reinterpret_cast<char*>(vnode_l1);
    const uint32_t* p_hoff_base = reinterpret_cast<const uint32_t*>(h_offset_l1);
    uint32_t* results = reinterpret_cast<uint32_t*>(stats_l1) + 8; // offset 8 words (32 bytes)

    constexpr uint32_t PAIRS_PER_WORKER = 60;
    constexpr uint32_t NUM_SCHEDULES = 5;
    constexpr uint32_t NUM_ITERATIONS = 1000;
    // Worker 1 reads from byte partition [960..1919] of the tile (src32 + 240 words)
    const uint32_t* src32_base = reinterpret_cast<const uint32_t*>(scratch_l1) + 240;

    uint16_t* p_vnode_ptrs1[PAIRS_PER_WORKER * 2];

    for (uint32_t s = 0; s < NUM_SCHEDULES; ++s) {
        // Setup direct pointers for Worker 1 (offset: s * 120 + 60)
        const uint32_t* sched_hoff = p_hoff_base + (s * 120 + 60);
        for (uint32_t p = 0; p < PAIRS_PER_WORKER; ++p) {
            uint32_t off = sched_hoff[p];
            p_vnode_ptrs1[2 * p + 0] = reinterpret_cast<uint16_t*>(vnode_bytes_ptr + (off & 0xFFFF));
            p_vnode_ptrs1[2 * p + 1] = reinterpret_cast<uint16_t*>(vnode_bytes_ptr + (off >> 16));
        }

        // Barrier start with Worker 0
        sync[4 * s + 1] = 1;
        while (sync[4 * s + 0] == 0) { }

        uint32_t t_start = get_wall_clock();
        for (uint32_t it = 0; it < NUM_ITERATIONS; ++it) {
            uint16_t* const* ptrs16 = p_vnode_ptrs1;
            const uint32_t* src = src32_base;

            // 7 chunks of 8 pairs = 56 pairs
            for (uint32_t chunk = 0; chunk < 7; ++chunk) {
                #pragma GCC unroll 8
                for (int k = 0; k < 8; ++k) {
                    uint16_t* pe = ptrs16[2 * k + 0];
                    uint16_t* po = ptrs16[2 * k + 1];
                    uint32_t w0 = src[4 * k + 0], w1 = src[4 * k + 1];
                    uint32_t w2 = src[4 * k + 2], w3 = src[4 * k + 3];
                    pe[0] = static_cast<uint16_t>(w0);
                    pe[1] = static_cast<uint16_t>(w1);
                    pe[2] = static_cast<uint16_t>(w2);
                    pe[3] = static_cast<uint16_t>(w3);
                    po[0] = static_cast<uint16_t>(w0 >> 16);
                    po[1] = static_cast<uint16_t>(w1 >> 16);
                    po[2] = static_cast<uint16_t>(w2 >> 16);
                    po[3] = static_cast<uint16_t>(w3 >> 16);
                }
                ptrs16 += 16;
                src += 32;
            }
            // Tail 4 pairs
            #pragma GCC unroll 4
            for (int k = 0; k < 4; ++k) {
                uint16_t* pe = ptrs16[2 * k + 0];
                uint16_t* po = ptrs16[2 * k + 1];
                uint32_t w0 = src[4 * k + 0], w1 = src[4 * k + 1];
                uint32_t w2 = src[4 * k + 2], w3 = src[4 * k + 3];
                pe[0] = static_cast<uint16_t>(w0);
                pe[1] = static_cast<uint16_t>(w1);
                pe[2] = static_cast<uint16_t>(w2);
                pe[3] = static_cast<uint16_t>(w3);
                po[0] = static_cast<uint16_t>(w0 >> 16);
                po[1] = static_cast<uint16_t>(w1 >> 16);
                po[2] = static_cast<uint16_t>(w2 >> 16);
                po[3] = static_cast<uint16_t>(w3 >> 16);
            }
        }
        uint32_t t_end = get_wall_clock();
        results[s] = (t_end - t_start) / NUM_ITERATIONS;

        // Barrier done with Worker 0
        sync[4 * s + 3] = 1;
        while (sync[4 * s + 2] == 0) { }
    }

    // Write Worker 1 results back to DRAM at offset 32 bytes (8 words)
    uint64_t stats_noc = get_noc_addr_from_bank_id<true>(stats_dram_bank, stats_dram_addr + 32);
    noc_async_write(reinterpret_cast<uint32_t>(results), stats_noc, NUM_SCHEDULES * sizeof(uint32_t));
    noc_async_write_barrier();
}
