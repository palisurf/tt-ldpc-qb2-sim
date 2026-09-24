#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t stats_dram_addr = get_arg_val<uint32_t>(0);
    uint32_t dram_bank       = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_stats = tt::CBIndex::c_16;
    cb_wait_front(cb_stats, 1);
    uint32_t l1_read_addr = get_read_ptr(cb_stats);

    uint64_t dst_noc_addr = get_noc_addr_from_bank_id<true>(dram_bank, stats_dram_addr);
    noc_async_write(l1_read_addr, dst_noc_addr, 4 * sizeof(uint32_t));
    noc_async_write_barrier();

    cb_pop_front(cb_stats, 1);
}