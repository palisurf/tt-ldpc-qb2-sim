#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

// ==============================================================================
// TRISC SFPU Unit Test: BRISC Reader Kernel
// Reads 1 input test tile (2,048 bytes) from DRAM into CB 0
// ==============================================================================

void kernel_main() {
    uint32_t src_dram_addr = get_arg_val<uint32_t>(0);
    uint32_t dram_bank     = get_arg_val<uint32_t>(1);
    uint32_t tile_bytes    = get_arg_val<uint32_t>(2); // 2048 bytes

    constexpr uint32_t cb_id_in = tt::CBIndex::c_0;

    cb_reserve_back(cb_id_in, 1);
    uint32_t l1_write_addr = get_write_ptr(cb_id_in);

    uint64_t src_noc_addr = get_noc_addr_from_bank_id<true>(dram_bank, src_dram_addr);
    noc_async_read(src_noc_addr, l1_write_addr, tile_bytes);
    noc_async_read_barrier();

    cb_push_back(cb_id_in, 1);
}
