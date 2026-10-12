#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

static inline uint32_t get_wall_clock() {
    volatile uint32_t* const wall_clock_reg = reinterpret_cast<volatile uint32_t*>(0xFFB121F0);
    return *wall_clock_reg;
}

static inline uint32_t rv32_pack(uint32_t a, uint32_t b) {
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

void kernel_main() {
    uint32_t stats_dram_addr = get_arg_val<uint32_t>(0);
    uint32_t stats_dram_bank = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_vnode    = tt::CBIndex::c_0;   // 20 KB
    constexpr uint32_t cb_h_offset = tt::CBIndex::c_2;   // 16 KB
    constexpr uint32_t cb_scratch  = tt::CBIndex::c_16;  // 4 KB (Scatter Input)
    constexpr uint32_t cb_stats    = tt::CBIndex::c_17;  // Output buffer for results

    uint32_t vnode_l1    = get_write_ptr(cb_vnode);
    uint32_t h_offset_l1 = get_read_ptr(cb_h_offset);
    uint32_t scratch_l1  = get_read_ptr(cb_scratch);
    uint32_t stats_l1    = get_write_ptr(cb_stats);

    char* vnode_bytes = reinterpret_cast<char*>(vnode_l1);
    const uint32_t* p_hoff_base = reinterpret_cast<const uint32_t*>(h_offset_l1);
    uint32_t* results = reinterpret_cast<uint32_t*>(stats_l1);

    constexpr uint32_t NUM_ITERATIONS  = 1000;
    constexpr uint32_t PAIRS_HALF_TILE = 60; // 60 edge pairs = 1/2 tile

    // Precompute direct 32-bit pointer table
    uint32_t* p_vnode_ptrs32[PAIRS_HALF_TILE * 2];
    uint16_t* p_vnode_ptrs16[PAIRS_HALF_TILE * 2];
    for (uint32_t p = 0; p < PAIRS_HALF_TILE; ++p) {
        uint32_t off = p_hoff_base[p];
        p_vnode_ptrs32[2 * p + 0] = reinterpret_cast<uint32_t*>(vnode_bytes + (off & 0xFFFF));
        p_vnode_ptrs32[2 * p + 1] = reinterpret_cast<uint32_t*>(vnode_bytes + (off >> 16));
        p_vnode_ptrs16[2 * p + 0] = reinterpret_cast<uint16_t*>(vnode_bytes + (off & 0xFFFF));
        p_vnode_ptrs16[2 * p + 1] = reinterpret_cast<uint16_t*>(vnode_bytes + (off >> 16));
    }

    // --------------------------------------------------------------------------
    // Variant 0: Legacy 32-bit Shift/Mask/OR Sequence
    // --------------------------------------------------------------------------
    {
        const uint32_t* src32_base = reinterpret_cast<const uint32_t*>(scratch_l1);
        uint32_t t_start = get_wall_clock();

        for (uint32_t it = 0; it < NUM_ITERATIONS; ++it) {
            const uint32_t* p_hoff = p_hoff_base;
            const uint32_t* src = src32_base;

            for (uint32_t p = 0; p < PAIRS_HALF_TILE; ++p) {
                uint32_t off = *p_hoff++;
                auto* pe = reinterpret_cast<uint32_t*>(vnode_bytes + (off & 0xFFFF));
                auto* po = reinterpret_cast<uint32_t*>(vnode_bytes + (off >> 16));

                uint32_t w0 = src[0], w1 = src[1];
                uint32_t e0_new = (w0 & 0xFFFF) | (w1 << 16);
                uint32_t o0_new = (w0 >> 16)    | (w1 & 0xFFFF0000);
                pe[0] = e0_new;
                po[0] = o0_new;

                uint32_t w2 = src[2], w3 = src[3];
                uint32_t e1_new = (w2 & 0xFFFF) | (w3 << 16);
                uint32_t o1_new = (w2 >> 16)    | (w3 & 0xFFFF0000);
                pe[1] = e1_new;
                po[1] = o1_new;

                src += 4;
            }
        }
        uint32_t t_end = get_wall_clock();
        results[0] = (t_end - t_start) / NUM_ITERATIONS;
    }

    // --------------------------------------------------------------------------
    // Variant 1: Direct 16-bit 'sh' Halfword Stores
    // --------------------------------------------------------------------------
    {
        const uint32_t* src32_base = reinterpret_cast<const uint32_t*>(scratch_l1);
        uint32_t t_start = get_wall_clock();

        for (uint32_t it = 0; it < NUM_ITERATIONS; ++it) {
            const uint32_t* p_hoff = p_hoff_base;
            const uint32_t* src = src32_base;

            for (uint32_t p = 0; p < PAIRS_HALF_TILE; ++p) {
                uint32_t off = *p_hoff++;
                auto* pe16 = reinterpret_cast<uint16_t*>(vnode_bytes + (off & 0xFFFF));
                auto* po16 = reinterpret_cast<uint16_t*>(vnode_bytes + (off >> 16));

                uint32_t w0 = src[0], w1 = src[1];
                uint32_t w2 = src[2], w3 = src[3];

                pe16[0] = static_cast<uint16_t>(w0);
                pe16[1] = static_cast<uint16_t>(w1);
                pe16[2] = static_cast<uint16_t>(w2);
                pe16[3] = static_cast<uint16_t>(w3);

                po16[0] = static_cast<uint16_t>(w0 >> 16);
                po16[1] = static_cast<uint16_t>(w1 >> 16);
                po16[2] = static_cast<uint16_t>(w2 >> 16);
                po16[3] = static_cast<uint16_t>(w3 >> 16);

                src += 4;
            }
        }
        uint32_t t_end = get_wall_clock();
        results[1] = (t_end - t_start) / NUM_ITERATIONS;
    }

    // --------------------------------------------------------------------------
    // Variant 2: Hardware 'pack' Opcode + 32-bit 'sw' Stores
    // --------------------------------------------------------------------------
    {
        const uint32_t* src32_base = reinterpret_cast<const uint32_t*>(scratch_l1);
        uint32_t t_start = get_wall_clock();

        for (uint32_t it = 0; it < NUM_ITERATIONS; ++it) {
            const uint32_t* p_hoff = p_hoff_base;
            const uint32_t* src = src32_base;

            for (uint32_t p = 0; p < PAIRS_HALF_TILE; ++p) {
                uint32_t off = *p_hoff++;
                auto* pe = reinterpret_cast<uint32_t*>(vnode_bytes + (off & 0xFFFF));
                auto* po = reinterpret_cast<uint32_t*>(vnode_bytes + (off >> 16));

                uint32_t w0 = src[0], w1 = src[1];
                uint32_t w2 = src[2], w3 = src[3];

                uint32_t e0 = rv32_pack(w0, w1);
                uint32_t o0 = rv32_pack(w0 >> 16, w1 >> 16);
                uint32_t e1 = rv32_pack(w2, w3);
                uint32_t o1 = rv32_pack(w2 >> 16, w3 >> 16);

                pe[0] = e0;
                po[0] = o0;
                pe[1] = e1;
                po[1] = o1;

                src += 4;
            }
        }
        uint32_t t_end = get_wall_clock();
        results[2] = (t_end - t_start) / NUM_ITERATIONS;
    }

    // --------------------------------------------------------------------------
    // Variant 3: 2-Pair Unrolled Pipeline with 'pack' + 32-bit 'sw' Stores
    // --------------------------------------------------------------------------
    {
        const uint32_t* src32_base = reinterpret_cast<const uint32_t*>(scratch_l1);
        uint32_t t_start = get_wall_clock();

        for (uint32_t it = 0; it < NUM_ITERATIONS; ++it) {
            const uint32_t* p_hoff = p_hoff_base;
            const uint32_t* src = src32_base;

            for (uint32_t p = 0; p < PAIRS_HALF_TILE; p += 2) {
                uint32_t offA = p_hoff[0];
                uint32_t offB = p_hoff[1];
                p_hoff += 2;

                auto* peA = reinterpret_cast<uint32_t*>(vnode_bytes + (offA & 0xFFFF));
                auto* poA = reinterpret_cast<uint32_t*>(vnode_bytes + (offA >> 16));
                auto* peB = reinterpret_cast<uint32_t*>(vnode_bytes + (offB & 0xFFFF));
                auto* poB = reinterpret_cast<uint32_t*>(vnode_bytes + (offB >> 16));

                uint32_t wA0 = src[0], wA1 = src[1], wA2 = src[2], wA3 = src[3];
                uint32_t wB0 = src[4], wB1 = src[5], wB2 = src[6], wB3 = src[7];
                src += 8;

                uint32_t eA0 = rv32_pack(wA0, wA1);
                uint32_t oA0 = rv32_pack(wA0 >> 16, wA1 >> 16);
                uint32_t eA1 = rv32_pack(wA2, wA3);
                uint32_t oA1 = rv32_pack(wA2 >> 16, wA3 >> 16);

                uint32_t eB0 = rv32_pack(wB0, wB1);
                uint32_t oB0 = rv32_pack(wB0 >> 16, wB1 >> 16);
                uint32_t eB1 = rv32_pack(wB2, wB3);
                uint32_t oB1 = rv32_pack(wB2 >> 16, wB3 >> 16);

                peA[0] = eA0; poA[0] = oA0; peA[1] = eA1; poA[1] = oA1;
                peB[0] = eB0; poB[0] = oB0; peB[1] = eB1; poB[1] = oB1;
            }
        }
        uint32_t t_end = get_wall_clock();
        results[3] = (t_end - t_start) / NUM_ITERATIONS;
    }

    // --------------------------------------------------------------------------
    // Variant 4: Direct Pointer Table + 2-Pair Unrolled Direct 16-bit 'sh'
    // --------------------------------------------------------------------------
    {
        const uint32_t* src32_base = reinterpret_cast<const uint32_t*>(scratch_l1);
        uint32_t t_start = get_wall_clock();

        for (uint32_t it = 0; it < NUM_ITERATIONS; ++it) {
            uint16_t* const* ptrs16 = p_vnode_ptrs16;
            const uint32_t* src = src32_base;

            for (uint32_t p = 0; p < PAIRS_HALF_TILE; p += 2) {
                uint16_t* peA = ptrs16[0]; uint16_t* poA = ptrs16[1];
                uint16_t* peB = ptrs16[2]; uint16_t* poB = ptrs16[3];
                ptrs16 += 4;

                uint32_t wA0 = src[0], wA1 = src[1], wA2 = src[2], wA3 = src[3];
                uint32_t wB0 = src[4], wB1 = src[5], wB2 = src[6], wB3 = src[7];
                src += 8;

                peA[0] = static_cast<uint16_t>(wA0);
                peA[1] = static_cast<uint16_t>(wA1);
                peA[2] = static_cast<uint16_t>(wA2);
                peA[3] = static_cast<uint16_t>(wA3);
                poA[0] = static_cast<uint16_t>(wA0 >> 16);
                poA[1] = static_cast<uint16_t>(wA1 >> 16);
                poA[2] = static_cast<uint16_t>(wA2 >> 16);
                poA[3] = static_cast<uint16_t>(wA3 >> 16);

                peB[0] = static_cast<uint16_t>(wB0);
                peB[1] = static_cast<uint16_t>(wB1);
                peB[2] = static_cast<uint16_t>(wB2);
                peB[3] = static_cast<uint16_t>(wB3);
                poB[0] = static_cast<uint16_t>(wB0 >> 16);
                poB[1] = static_cast<uint16_t>(wB1 >> 16);
                poB[2] = static_cast<uint16_t>(wB2 >> 16);
                poB[3] = static_cast<uint16_t>(wB3 >> 16);
            }
        }
        uint32_t t_end = get_wall_clock();
        results[4] = (t_end - t_start) / NUM_ITERATIONS;
    }

    // --------------------------------------------------------------------------
    // Variant 5: Direct Pointer Table + 4-Pair Unrolled Direct 16-bit 'sh'
    // --------------------------------------------------------------------------
    {
        const uint32_t* src32_base = reinterpret_cast<const uint32_t*>(scratch_l1);
        uint32_t t_start = get_wall_clock();

        for (uint32_t it = 0; it < NUM_ITERATIONS; ++it) {
            uint16_t* const* ptrs16 = p_vnode_ptrs16;
            const uint32_t* src = src32_base;

            for (uint32_t p = 0; p < PAIRS_HALF_TILE; p += 4) {
                // Pair A
                uint16_t* peA = ptrs16[0]; uint16_t* poA = ptrs16[1];
                uint32_t wA0 = src[0], wA1 = src[1], wA2 = src[2], wA3 = src[3];
                peA[0] = static_cast<uint16_t>(wA0); peA[1] = static_cast<uint16_t>(wA1);
                peA[2] = static_cast<uint16_t>(wA2); peA[3] = static_cast<uint16_t>(wA3);
                poA[0] = static_cast<uint16_t>(wA0 >> 16); poA[1] = static_cast<uint16_t>(wA1 >> 16);
                poA[2] = static_cast<uint16_t>(wA2 >> 16); poA[3] = static_cast<uint16_t>(wA3 >> 16);

                // Pair B
                uint16_t* peB = ptrs16[2]; uint16_t* poB = ptrs16[3];
                uint32_t wB0 = src[4], wB1 = src[5], wB2 = src[6], wB3 = src[7];
                peB[0] = static_cast<uint16_t>(wB0); peB[1] = static_cast<uint16_t>(wB1);
                peB[2] = static_cast<uint16_t>(wB2); peB[3] = static_cast<uint16_t>(wB3);
                poB[0] = static_cast<uint16_t>(wB0 >> 16); poB[1] = static_cast<uint16_t>(wB1 >> 16);
                poB[2] = static_cast<uint16_t>(wB2 >> 16); poB[3] = static_cast<uint16_t>(wB3 >> 16);

                // Pair C
                uint16_t* peC = ptrs16[4]; uint16_t* poC = ptrs16[5];
                uint32_t wC0 = src[8], wC1 = src[9], wC2 = src[10], wC3 = src[11];
                peC[0] = static_cast<uint16_t>(wC0); peC[1] = static_cast<uint16_t>(wC1);
                peC[2] = static_cast<uint16_t>(wC2); peC[3] = static_cast<uint16_t>(wC3);
                poC[0] = static_cast<uint16_t>(wC0 >> 16); poC[1] = static_cast<uint16_t>(wC1 >> 16);
                poC[2] = static_cast<uint16_t>(wC2 >> 16); poC[3] = static_cast<uint16_t>(wC3 >> 16);

                // Pair D
                uint16_t* peD = ptrs16[6]; uint16_t* poD = ptrs16[7];
                uint32_t wD0 = src[12], wD1 = src[13], wD2 = src[14], wD3 = src[15];
                peD[0] = static_cast<uint16_t>(wD0); peD[1] = static_cast<uint16_t>(wD1);
                peD[2] = static_cast<uint16_t>(wD2); peD[3] = static_cast<uint16_t>(wD3);
                poD[0] = static_cast<uint16_t>(wD0 >> 16); poD[1] = static_cast<uint16_t>(wD1 >> 16);
                poD[2] = static_cast<uint16_t>(wD2 >> 16); poD[3] = static_cast<uint16_t>(wD3 >> 16);

                ptrs16 += 8;
                src += 16;
            }
        }
        uint32_t t_end = get_wall_clock();
        results[5] = (t_end - t_start) / NUM_ITERATIONS;
    }

    // --------------------------------------------------------------------------
    // Variant 6: Direct Pointer Table + 4-Pair Unrolled + 'pack' + 32-bit 'sw'
    // --------------------------------------------------------------------------
    {
        const uint32_t* src32_base = reinterpret_cast<const uint32_t*>(scratch_l1);
        uint32_t t_start = get_wall_clock();

        for (uint32_t it = 0; it < NUM_ITERATIONS; ++it) {
            uint32_t* const* ptrs32 = p_vnode_ptrs32;
            const uint32_t* src = src32_base;

            for (uint32_t p = 0; p < PAIRS_HALF_TILE; p += 4) {
                #pragma GCC unroll 4
                for (int k = 0; k < 4; ++k) {
                    uint32_t* pe = ptrs32[2 * k + 0];
                    uint32_t* po = ptrs32[2 * k + 1];
                    uint32_t w0 = src[4 * k + 0], w1 = src[4 * k + 1];
                    uint32_t w2 = src[4 * k + 2], w3 = src[4 * k + 3];
                    pe[0] = rv32_pack(w0, w1);
                    po[0] = rv32_pack(w0 >> 16, w1 >> 16);
                    pe[1] = rv32_pack(w2, w3);
                    po[1] = rv32_pack(w2 >> 16, w3 >> 16);
                }
                ptrs32 += 8;
                src += 16;
            }
        }
        uint32_t t_end = get_wall_clock();
        results[6] = (t_end - t_start) / NUM_ITERATIONS;
    }

    // --------------------------------------------------------------------------
    // Variant 7: Direct Pointer Table + 8-Pair Unrolled Direct 16-bit 'sh'
    // --------------------------------------------------------------------------
    {
        const uint32_t* src32_base = reinterpret_cast<const uint32_t*>(scratch_l1);
        uint32_t t_start = get_wall_clock();

        for (uint32_t it = 0; it < NUM_ITERATIONS; ++it) {
            uint16_t* const* ptrs16 = p_vnode_ptrs16;
            const uint32_t* src = src32_base;

            // 7 chunks of 8 pairs = 56 pairs
            for (uint32_t chunk = 0; chunk < 7; ++chunk) {
                #pragma GCC unroll 8
                for (int k = 0; k < 8; ++k) {
                    uint16_t* pe = ptrs16[2 * k + 0];
                    uint16_t* po = ptrs16[2 * k + 1];
                    uint32_t w0 = src[4 * k + 0], w1 = src[4 * k + 1];
                    uint32_t w2 = src[4 * k + 2], w3 = src[4 * k + 3];
                    pe[0] = static_cast<uint16_t>(w0);
                    pe[1] = static_cast<uint16_t>(w1);
                    pe[2] = static_cast<uint16_t>(w2);
                    pe[3] = static_cast<uint16_t>(w3);
                    po[0] = static_cast<uint16_t>(w0 >> 16);
                    po[1] = static_cast<uint16_t>(w1 >> 16);
                    po[2] = static_cast<uint16_t>(w2 >> 16);
                    po[3] = static_cast<uint16_t>(w3 >> 16);
                }
                ptrs16 += 16;
                src += 32;
            }
            // Tail 4 pairs
            #pragma GCC unroll 4
            for (int k = 0; k < 4; ++k) {
                uint16_t* pe = ptrs16[2 * k + 0];
                uint16_t* po = ptrs16[2 * k + 1];
                uint32_t w0 = src[4 * k + 0], w1 = src[4 * k + 1];
                uint32_t w2 = src[4 * k + 2], w3 = src[4 * k + 3];
                pe[0] = static_cast<uint16_t>(w0);
                pe[1] = static_cast<uint16_t>(w1);
                pe[2] = static_cast<uint16_t>(w2);
                pe[3] = static_cast<uint16_t>(w3);
                po[0] = static_cast<uint16_t>(w0 >> 16);
                po[1] = static_cast<uint16_t>(w1 >> 16);
                po[2] = static_cast<uint16_t>(w2 >> 16);
                po[3] = static_cast<uint16_t>(w3 >> 16);
            }
        }
        uint32_t t_end = get_wall_clock();
        results[7] = (t_end - t_start) / NUM_ITERATIONS;
    }

    // Write results back to DRAM
    uint64_t stats_noc = get_noc_addr_from_bank_id<true>(stats_dram_bank, stats_dram_addr);
    noc_async_write(stats_l1, stats_noc, 32); // 8 results (32 bytes)
    noc_async_write_barrier();
}
