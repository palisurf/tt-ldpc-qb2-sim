#include <stdint.h>

// RV32 Zba Address Generation
static inline uint32_t rv32_sh1add(uint32_t a, uint32_t b) {
    uint32_t res;
    asm volatile(
        ".option push\n\t"
        ".option arch, +zba\n\t"
        "sh1add %0, %1, %2\n\t"
        ".option pop"
        : "=r"(res) : "r"(a), "r"(b)
    );
    return res;
}

static inline uint32_t rv32_sh2add(uint32_t a, uint32_t b) {
    uint32_t res;
    asm volatile(
        ".option push\n\t"
        ".option arch, +zba\n\t"
        "sh2add %0, %1, %2\n\t"
        ".option pop"
        : "=r"(res) : "r"(a), "r"(b)
    );
    return res;
}

static inline uint32_t rv32_sh3add(uint32_t a, uint32_t b) {
    uint32_t res;
    asm volatile(
        ".option push\n\t"
        ".option arch, +zba\n\t"
        "sh3add %0, %1, %2\n\t"
        ".option pop"
        : "=r"(res) : "r"(a), "r"(b)
    );
    return res;
}

// RV32 Zbb Bit Manipulation
static inline uint32_t rv32_andn(uint32_t a, uint32_t b) {
    uint32_t res;
    asm volatile(
        ".option push\n\t"
        ".option arch, +zbb\n\t"
        "andn %0, %1, %2\n\t"
        ".option pop"
        : "=r"(res) : "r"(a), "r"(b)
    );
    return res;
}

static inline uint32_t rv32_zext_h(uint32_t a) {
    uint32_t res;
    asm volatile(
        ".option push\n\t"
        ".option arch, +zbb\n\t"
        "zext.h %0, %1\n\t"
        ".option pop"
        : "=r"(res) : "r"(a)
    );
    return res;
}

static inline uint32_t rv32_cpop(uint32_t a) {
    uint32_t res;
    asm volatile(
        ".option push\n\t"
        ".option arch, +zbb\n\t"
        "cpop %0, %1\n\t"
        ".option pop"
        : "=r"(res) : "r"(a)
    );
    return res;
}

uint32_t test_calc(uint32_t base, uint32_t idx, uint32_t mask) {
    uint32_t addr = rv32_sh3add(idx, base);
    uint32_t masked = rv32_andn(addr, mask);
    return rv32_zext_h(masked) + rv32_cpop(masked);
}
