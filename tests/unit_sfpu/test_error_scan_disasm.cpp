#include <stdint.h>

void error_scan_current(const uint32_t* p, uint32_t mask01, uint32_t mask23, uint32_t& err01, uint32_t& err23) {
    uint32_t e01 = 0;
    uint32_t e23 = 0;
    if (mask01 != 0 || mask23 != 0) {
        #pragma GCC unroll 0
        for (uint32_t i = 0; i < 2048; i += 16) {
            if (mask01) {
                e01 |= (p[i + 0] | p[i + 2] | p[i + 4] | p[i + 6] |
                        p[i + 8] | p[i + 10] | p[i + 12] | p[i + 14]);
            }
            if (mask23) {
                e23 |= (p[i + 1] | p[i + 3] | p[i + 5] | p[i + 7] |
                        p[i + 9] | p[i + 11] | p[i + 13] | p[i + 15]);
            }
        }
    }
    err01 = e01 & mask01;
    err23 = e23 & mask23;
}
