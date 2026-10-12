#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dst0_dram_addr = get_arg_val<uint32_t>(0);
    uint32_t dram0_bank     = get_arg_val<uint32_t>(1);
    uint32_t dst1_dram_addr = get_arg_val<uint32_t>(2);
    uint32_t dram1_bank     = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
    constexpr uint32_t cb_out1 = tt::CBIndex::c_17;

    cb_wait_front(cb_out0, 1);
    uint32_t ptr0 = get_read_ptr(cb_out0);
    uint64_t dst0_noc = get_noc_addr_from_bank_id<true>(dram0_bank, dst0_dram_addr);
    noc_async_write(ptr0, dst0_noc, 2048);
    noc_async_write_barrier();
    cb_pop_front(cb_out0, 1);

    cb_wait_front(cb_out1, 1);
    uint32_t ptr1 = get_read_ptr(cb_out1);
    uint64_t dst1_noc = get_noc_addr_from_bank_id<true>(dram1_bank, dst1_dram_addr);
    noc_async_write(ptr1, dst1_noc, 2048);
    noc_async_write_barrier();
    cb_pop_front(cb_out1, 1);
}
