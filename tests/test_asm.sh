#!/bin/bash
cat << 'EOF' > /tmp/test_pack.c
#include <stdint.h>
uint32_t test_pack(uint32_t a, uint32_t b) {
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
EOF
/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-gcc -mcpu=tt-bh -O2 /tmp/test_pack.c -c -o /tmp/test_pack.o
/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-objdump -d /tmp/test_pack.o
