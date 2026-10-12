#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;

    cb_reserve_back(cb_in0, 1);
    uint32_t ptr0 = get_write_ptr(cb_in0);
    auto* p0 = reinterpret_cast<uint16_t*>(ptr0);
    for (uint32_t i = 0; i < 1024; ++i) p0[i] = 0x4120; // 10.0f
    cb_push_back(cb_in0, 1);
}
