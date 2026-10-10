#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

// ==============================================================================
// TRISC SFPU Unit Test: NCRISC Writer Kernel
// Writes computed output tile from CB 16 (2,048 bytes) and telemetry to DRAM
// ==============================================================================

void kernel_main() {
    uint32_t dst_dram_addr   = get_arg_val<uint32_t>(0);
    uint32_t stats_dram_addr = get_arg_val<uint32_t>(1);
    uint32_t dram_bank       = get_arg_val<uint32_t>(2);
    uint32_t tile_bytes      = get_arg_val<uint32_t>(3); // 2048 bytes
    uint32_t stats_bytes     = get_arg_val<uint32_t>(4); // 64 bytes

    constexpr uint32_t cb_id_out   = tt::CBIndex::c_16;
    constexpr uint32_t cb_id_stats = tt::CBIndex::c_17;

    // 1. Wait for computed tile from TRISC PACK thread
    cb_wait_front(cb_id_out, 1);
    uint32_t l1_read_addr = get_read_ptr(cb_id_out);

    uint64_t dst_noc_addr = get_noc_addr_from_bank_id<true>(dram_bank, dst_dram_addr);
    noc_async_write(l1_read_addr, dst_noc_addr, tile_bytes);
    noc_async_write_barrier();

    cb_pop_front(cb_id_out, 1);

    // 2. Wait for telemetry from TRISC MATH thread
    cb_wait_front(cb_id_stats, 1);
    uint32_t stats_l1_addr = get_read_ptr(cb_id_stats);

    uint64_t stats_noc_addr = get_noc_addr_from_bank_id<true>(dram_bank, stats_dram_addr);
    noc_async_write(stats_l1_addr, stats_noc_addr, stats_bytes);
    noc_async_write_barrier();

    cb_pop_front(cb_id_stats, 1);
}
