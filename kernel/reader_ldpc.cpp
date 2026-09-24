#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t h_dram_addr = get_arg_val<uint32_t>(0);
    uint32_t dram_bank   = get_arg_val<uint32_t>(1);
    uint32_t h_bytes     = get_arg_val<uint32_t>(2);
    uint32_t num_tiles   = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_h = tt::CBIndex::c_2;
    cb_reserve_back(cb_h, 1);
    uint32_t l1_write_addr = get_write_ptr(cb_h);

    uint64_t src_noc_addr = get_noc_addr_from_bank_id<true>(dram_bank, h_dram_addr);
    noc_async_read(src_noc_addr, l1_write_addr, h_bytes);
    noc_async_read_barrier();

    cb_push_back(cb_h, 1);
}