#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

// ==============================================================================
// NCRISC Writer Kernel for Multi-Tile Vector Box-Muller AWGN Benchmark
// Streams 10 tiles (20.0 KB) from CB 16 and telemetry from CB 17 to DRAM
// ==============================================================================

void kernel_main() {
    uint32_t dst_dram_addr   = get_arg_val<uint32_t>(0);
    uint32_t stats_dram_addr = get_arg_val<uint32_t>(1);
    uint32_t dram_bank       = get_arg_val<uint32_t>(2);
    uint32_t tile_bytes      = get_arg_val<uint32_t>(3); // 2048 bytes
    uint32_t stats_bytes     = get_arg_val<uint32_t>(4); // 64 bytes
    uint32_t num_tiles       = get_arg_val<uint32_t>(5); // 10 tiles

    constexpr uint32_t cb_id_out   = tt::CBIndex::c_16;
    constexpr uint32_t cb_id_stats = tt::CBIndex::c_17;

    const InterleavedAddrGen<true> dram_addrgen = {
        .bank_base_address = dst_dram_addr,
        .page_size = tile_bytes
    };

    for (uint32_t t = 0; t < num_tiles; ++t) {
        cb_wait_front(cb_id_out, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);

        noc_async_write_tile(t, dram_addrgen, l1_read_addr);
        noc_async_write_barrier();

        cb_pop_front(cb_id_out, 1);
    }

    // Wait for telemetry from TRISC MATH thread
    cb_wait_front(cb_id_stats, 1);
    uint32_t stats_l1_addr = get_read_ptr(cb_id_stats);

    uint64_t stats_noc_addr = get_noc_addr_from_bank_id<true>(dram_bank, stats_dram_addr);
    noc_async_write(stats_l1_addr, stats_noc_addr, stats_bytes);
    noc_async_write_barrier();

    cb_pop_front(cb_id_stats, 1);
}
