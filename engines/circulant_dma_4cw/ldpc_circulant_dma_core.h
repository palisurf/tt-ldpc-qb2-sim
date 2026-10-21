// ==============================================================================
// engines/circulant_dma_4cw/ldpc_circulant_dma_core.h
//
// Tenstorrent QuietBox 2: 4-Codeword Circulant DMA LDPC Decoder Core Architecture
// Features:
//   - 4 Codewords Interleaved (1 variable node = 8 Bytes = 64 bits)
//   - 512-bit DMA Contiguous Slices (Zero indirect pointer tables!)
//   - 40-Tile Face-Aligned Schedule (32 Deg-6 Tiles + 8 Deg-3 Tiles)
//   - Zero-Transpose Vertical SIMD Check Node Compute on DEST Registers
//   - Autonomous 128-bit L1 Vector Memory Transfers on RISC-V Workers
// ==============================================================================

#pragma once

#include <stdint.h>
#include <cmath>
#include <cstring>

#include "ar4ja_circulant_descriptors_4cw.h"

namespace QC_LDPC_4CW {
#pragma pack(push, 2)
struct alignas(8) Packed4CW_BF16 {
    uint16_t cw[4]; // [0..3] bfloat16 LLRs for 4 independent codewords

    inline void set_all(uint16_t val) {
        cw[0] = val; cw[1] = val; cw[2] = val; cw[3] = val;
    }

    inline void apply_freeze_mask(const Packed4CW_BF16& updated, uint8_t active_mask) {
        if (active_mask & 0x1) cw[0] = updated.cw[0];
        if (active_mask & 0x2) cw[1] = updated.cw[1];
        if (active_mask & 0x4) cw[2] = updated.cw[2];
        if (active_mask & 0x8) cw[3] = updated.cw[3];
    }
};
#pragma pack(pop)
static_assert(sizeof(Packed4CW_BF16) == 8, "Packed4CW_BF16 must be exactly 8 bytes (64 bits)");
}

using QC_LDPC_4CW::Packed4CW_BF16;

inline float uint_as_float(uint32_t u) {
    union { uint32_t u; float f; } converter;
    converter.u = u;
    return converter.f;
}

inline uint32_t float_as_uint(float f) {
    union { uint32_t u; float f; } converter;
    converter.f = f;
    return converter.u;
}

// ------------------------------------------------------------------------------
// RV32 "Zbb" Native Hardware Bit Manipulation Instructions
// ------------------------------------------------------------------------------
#if defined(__riscv)
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
#else
static inline uint32_t rv32_andn(uint32_t a, uint32_t b) {
    return a & ~b;
}

static inline uint32_t rv32_cpop(uint32_t a) {
    return __builtin_popcount(a);
}
#endif

// ------------------------------------------------------------------------------
// Hardware Compile-Time Test Modes
// ------------------------------------------------------------------------------
#ifndef LDPC_TEST_PASSTHROUGH_MODE
#define LDPC_TEST_PASSTHROUGH_MODE 0
#endif

#ifndef LDPC_TEST_R_DATAFLOW_MODE
#define LDPC_TEST_R_DATAFLOW_MODE 0
#endif

// ------------------------------------------------------------------------------
// Circular Buffer (CB) Allocations in Tensix L1 SRAM (Total 1.5 MB Available)
// Inter-Batch Ping-Pong Architecture: 9 Circular Buffers (220 KB Total, 14.3% L1)
// ------------------------------------------------------------------------------
namespace TileMatrixCB {
    constexpr uint8_t CB_VNODE_A           = 0;   // 20 KB: Active evolving variable posterior LLRs (Batch A: CW 0..3)
    constexpr uint8_t CB_R_MSG_A           = 1;   // 80 KB: 40 tile check messages circular FIFO (Batch A)
    constexpr uint8_t CB_SCRATCH_GATHER_0  = 3;   // 4 KB: Stage 1 Gather Buffer (Batch A, Double-buffered, 2 KB page)
    constexpr uint8_t CB_SCRATCH_GATHER_1  = 4;   // 4 KB: Stage 1 Gather Buffer (Batch B, Double-buffered, 2 KB page)
    constexpr uint8_t CB_VNODE_B           = 5;   // 20 KB: Active evolving variable posterior LLRs (Batch B: CW 4..7)
    constexpr uint8_t CB_SCRATCH_SCATTER_0 = 16;  // 4 KB: Stage 3 Scatter Buffer (Batch A, Double-buffered, 2 KB page)
    constexpr uint8_t CB_R_MSG_B           = 17;  // 80 KB: 40 tile check messages circular FIFO (Batch B)
    constexpr uint8_t CB_SCRATCH_SCATTER_1 = 18;  // 4 KB: Stage 3 Scatter Buffer (Batch B, Double-buffered, 2 KB page)
    constexpr uint8_t CB_TELEMETRY         = 24;  // 4 KB: Hardware semaphores, mailboxes, telemetry
    constexpr uint32_t TELEMETRY_L1_BASE   = 0x39C00;

    // Aliases for Inter-Batch Ping-Pong clarity
    constexpr uint8_t CB_R_MSG             = CB_R_MSG_A;
    constexpr uint8_t CB_SCRATCH_GATHER_A  = CB_SCRATCH_GATHER_0;
    constexpr uint8_t CB_SCRATCH_GATHER_B  = CB_SCRATCH_GATHER_1;
    constexpr uint8_t CB_SCRATCH_SCATTER_A = CB_SCRATCH_SCATTER_0;
    constexpr uint8_t CB_SCRATCH_SCATTER_B = CB_SCRATCH_SCATTER_1;
}

// Maestro Command Enum issued by Centralized TRISC1 Orchestrator (Slot 22)
enum class MaestroCmd : uint32_t {
    CMD_IDLE           = 0, // Initial idle state
    CMD_STEP_EXEC      = 1, // Execute Step k (Scatter k-2, Gather k)
    CMD_DRAIN_RESIDUAL = 2, // Early termination: pop in-flight residual tile
    CMD_BATCH_DONE     = 3, // Batch pair complete: rendezvous for next batch
    CMD_HALT           = 4  // Simulation finished: halt all cores
};

// Hardware Semaphore and Inter-Worker Barrier Slots in CB_TELEMETRY
namespace SyncSlot {
    constexpr uint32_t TRISC_MATH_CYCLES   = 11; // Pure TRISC1 SFPU math cycles
    constexpr uint32_t BRISC_GATHER_ERR_CW01_A = 12;
    constexpr uint32_t BRISC_GATHER_ERR_CW23_A = 13;
    constexpr uint32_t TRISC_CYCLES        = 14;
    constexpr uint32_t BRISC_CYCLES        = 15;
    constexpr uint32_t BATCH_READY         = 16;
    constexpr uint32_t ACTIVE_MASK_A       = 18; // Active codeword mask [3:0] for Batch A
    constexpr uint32_t ACTIVE_MASK_B       = 19; // Active codeword mask [3:0] for Batch B
    constexpr uint32_t EARLY_TERMINATION   = 20; // 1 if both Batch A and Batch B have converged early
    constexpr uint32_t TRISC1_GRANT_STEP   = 21; // Monotonic step grant issued by TRISC1 Orchestrator
    constexpr uint32_t TRISC1_CMD          = 22; // Command enum (MaestroCmd) issued by TRISC1
    constexpr uint32_t DRAIN_ACK_T0        = 23; // TRISC0 acknowledgment of residual FIFO drain
    constexpr uint32_t GATHER_START        = 24; // Signaled when SCRATCH_GATHER_PTR is published
    constexpr uint32_t SCRATCH_GATHER_PTR  = 25; // Active gather L1 write pointer
    constexpr uint32_t SCRATCH_SCATTER_PTR = 26; // Active scatter L1 read pointer
    constexpr uint32_t VNODE_L1_ADDR_A     = 27; // Batch A variable node L1 base address
    constexpr uint32_t VNODE_L1_ADDR_B     = 28; // Batch B variable node L1 base address
    constexpr uint32_t BATCH_BARRIER_BRISC  = 29; // BRISC batch rendezvous
    constexpr uint32_t BATCH_BARRIER_NCRISC = 30; // NCRISC batch rendezvous
    constexpr uint32_t BATCH_BARRIER_TRISC  = 31; // TRISC batch rendezvous

    // Aliases
    constexpr uint32_t ACTIVE_MASK         = ACTIVE_MASK_A;
    constexpr uint32_t VNODE_L1_ADDR       = VNODE_L1_ADDR_A;

    // Diagnostic Progress Checkpoint Slots (Encoded as (step << 8) | phase)
    constexpr uint32_t DEBUG_BRISC_STEP    = 70;
    constexpr uint32_t DEBUG_NCRISC_STEP   = 71;
    constexpr uint32_t DEBUG_TRISC0_STEP   = 72;
    constexpr uint32_t DEBUG_TRISC1_STEP   = 73;
    constexpr uint32_t DEBUG_TRISC2_STEP   = 74;

    // Dual-Worker Face Parallel Barriers
    constexpr uint32_t GATHER_DONE_W0      = 184; // Worker 0 (BRISC: Face 0 & 2) gather done
    constexpr uint32_t GATHER_DONE_W1      = 185; // Worker 1 (NCRISC: Face 1 & 3) gather done
    constexpr uint32_t SCATTER_DONE_W0     = 188; // Worker 0 (BRISC: Face 0 & 2) scatter done
    constexpr uint32_t SCATTER_DONE_W1     = 189; // Worker 1 (NCRISC: Face 1 & 3) scatter done
    constexpr uint32_t GATHER_ALL_DONE     = 192; // All 4 faces gathered into scratch_in
    constexpr uint32_t SCATTER_START       = 193; // Signaled by NCRISC when SCRATCH_SCATTER_PTR is published

    // Satellite Parity Reports to TRISC1
    constexpr uint32_t PARITY_REPORT_W0    = 194;
    constexpr uint32_t PARITY_REPORT_W1    = 195;
    constexpr uint32_t ITER_DONE_A         = 196;
    constexpr uint32_t ITER_DONE_B         = 197;
    constexpr uint32_t DRAIN_ACK_W1        = 198;
}

// Telemetry Buffer Layout in CB_TELEMETRY (1024 uint32 words)
namespace TelemSlot {
    constexpr uint32_t TOTAL_BE            = 0;
    constexpr uint32_t TOTAL_FE            = 1;
    constexpr uint32_t TOTAL_ITERS         = 2;
    constexpr uint32_t TOTAL_CWS           = 3;
    constexpr uint32_t ELAPSED_CYCLES      = 4;
    constexpr uint32_t CYCLES_PER_BATCH    = 5;
    constexpr uint32_t TRISC_CYCLES        = 6;
    constexpr uint32_t BRISC_CYCLES        = 7;
    constexpr uint32_t CYCLES_CHANNEL      = 8;
    constexpr uint32_t CYCLES_ERROR_SCAN   = 9;
    constexpr uint32_t TOTAL_TILES_SIM     = 10;
    constexpr uint32_t TELEMETRY_L1_ADDR   = 11;
    constexpr uint32_t TRISC_MATH_CYCLES   = 12; // Pure TRISC1 SFPU math cycles

    // NCRISC Tile Breakdown
    constexpr uint32_t NC_GATHER           = 32;
    constexpr uint32_t NC_WAIT             = 33;
    constexpr uint32_t NC_SCATTER          = 34;

    // BRISC Tile Breakdown
    constexpr uint32_t BR_GATHER           = 48;
    constexpr uint32_t BR_WAIT             = 49;
    constexpr uint32_t BR_SCATTER          = 50;

    // Diagnostics (written by NCRISC on Batch 0 and exit)
    constexpr uint32_t INIT_NEG_CW0        = 104; // 104..107 (CW 0..3)
    constexpr uint32_t INIT_BF16_W01       = 108;
    constexpr uint32_t INIT_BF16_W23       = 109;
    constexpr uint32_t MU_BITS             = 110;
    constexpr uint32_t SIGMA_BITS          = 111;
    constexpr uint32_t LAST_BATCH_BE_CW0   = 112; // 112..119 (CW 0..7)
    constexpr uint32_t LAST_BATCH_ITER_CW0 = 120; // 120..127 (CW 0..7)

    // Trapped Failing Batch Diagnostics (Slots 210..213)
    constexpr uint32_t FIRST_FAILING_BATCH = 210;
    constexpr uint32_t FIRST_FAILING_CW    = 211;
    constexpr uint32_t FIRST_FAILING_BE    = 212;
    constexpr uint32_t FIRST_FAILING_MASK  = 213;
}

// ------------------------------------------------------------------------------
// Fast 128-bit (32 x 4-Byte Word) Memory Movement Primitives for RISC-V Workers
// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------
// Fast 128-bit (32 x 4-Byte Word) Memory Movement Primitives for RISC-V Workers
// ------------------------------------------------------------------------------
inline void copy_128b(void* dst, const void* src) {
    auto* d = reinterpret_cast<uint32_t*>(dst);
    const auto* s = reinterpret_cast<const uint32_t*>(src);
    #pragma GCC unroll 4
    for (int i = 0; i < 32; i += 8) {
        uint32_t w0 = s[i + 0];
        uint32_t w1 = s[i + 1];
        uint32_t w2 = s[i + 2];
        uint32_t w3 = s[i + 3];
        uint32_t w4 = s[i + 4];
        uint32_t w5 = s[i + 5];
        uint32_t w6 = s[i + 6];
        uint32_t w7 = s[i + 7];
        d[i + 0] = w0;
        d[i + 1] = w1;
        d[i + 2] = w2;
        d[i + 3] = w3;
        d[i + 4] = w4;
        d[i + 5] = w5;
        d[i + 6] = w6;
        d[i + 7] = w7;
    }
}

inline void gather_slice_l1(uint32_t dest_addr, const QC_LDPC_4CW::CirculantSliceDesc& desc, const char* vnode_base) {
#if (defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC))
    const uint32_t src_a = reinterpret_cast<uint32_t>(vnode_base) + desc.offset_a;
    if (__builtin_expect(desc.len_b == 0, 1)) {
        if ((src_a & 0xF) == 0) {
            noc_async_read(get_noc_addr(src_a), dest_addr, 128);
        } else {
            copy_128b(reinterpret_cast<char*>(dest_addr), reinterpret_cast<const char*>(src_a));
        }
    } else {
        std::memcpy(reinterpret_cast<char*>(dest_addr), reinterpret_cast<const char*>(src_a), desc.len_a);
        std::memcpy(reinterpret_cast<char*>(dest_addr) + desc.len_a, vnode_base + desc.offset_b, desc.len_b);
    }
#else
    char* dest = reinterpret_cast<char*>(dest_addr);
    const char* src_a = vnode_base + desc.offset_a;
    if (__builtin_expect(desc.len_b == 0, 1)) {
        copy_128b(dest, src_a);
    } else {
        std::memcpy(dest, src_a, desc.len_a);
        std::memcpy(dest + desc.len_a, vnode_base + desc.offset_b, desc.len_b);
    }
#endif
}

inline void scatter_slice_l1(const QC_LDPC_4CW::CirculantSliceDesc& desc, uint32_t src_addr, char* vnode_base) {
#if (defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC))
    const uint32_t dst_a = reinterpret_cast<uint32_t>(vnode_base) + desc.offset_a;
    if (__builtin_expect(desc.len_b == 0, 1)) {
        if ((dst_a & 0xF) == 0) {
            noc_async_write(src_addr, get_noc_addr(dst_a), 128);
        } else {
            copy_128b(reinterpret_cast<char*>(dst_a), reinterpret_cast<const char*>(src_addr));
        }
    } else {
        std::memcpy(reinterpret_cast<char*>(dst_a), reinterpret_cast<const char*>(src_addr), desc.len_a);
        std::memcpy(reinterpret_cast<char*>(vnode_base) + desc.offset_b, reinterpret_cast<const char*>(src_addr) + desc.len_a, desc.len_b);
    }
#else
    const char* src = reinterpret_cast<const char*>(src_addr);
    char* dst_a = vnode_base + desc.offset_a;
    if (__builtin_expect(desc.len_b == 0, 1)) {
        copy_128b(dst_a, src);
    } else {
        std::memcpy(dst_a, src, desc.len_a);
        std::memcpy(vnode_base + desc.offset_b, src + desc.len_a, desc.len_b);
    }
#endif
}

inline void scatter_slice_masked_l1(const QC_LDPC_4CW::CirculantSliceDesc& desc, uint32_t src_addr, char* vnode_base, uint32_t active_mask) {
    if (active_mask == 0xF) {
        scatter_slice_l1(desc, src_addr, vnode_base);
        return;
    }
    uint32_t mask_cw01 = ((active_mask & 1) ? 0x0000FFFF : 0) | ((active_mask & 2) ? 0xFFFF0000 : 0);
    uint32_t mask_cw23 = ((active_mask & 4) ? 0x0000FFFF : 0) | ((active_mask & 8) ? 0xFFFF0000 : 0);

    auto scatter_masked = [&](uint32_t offset, const char* s, uint32_t len) {
        auto* dest = reinterpret_cast<uint32_t*>(vnode_base + offset);
        const auto* source = reinterpret_cast<const uint32_t*>(s);
        uint32_t num_words = len / 4;
        #pragma GCC unroll 16
        for (uint32_t i = 0; i < num_words; i += 2) {
            dest[i + 0] = (dest[i + 0] & ~mask_cw01) | (source[i + 0] & mask_cw01);
            dest[i + 1] = (dest[i + 1] & ~mask_cw23) | (source[i + 1] & mask_cw23);
        }
    };
    scatter_masked(desc.offset_a, reinterpret_cast<const char*>(src_addr), desc.len_a);
    if (desc.len_b > 0) {
        scatter_masked(desc.offset_b, reinterpret_cast<const char*>(src_addr) + desc.len_a, desc.len_b);
    }
}

// ------------------------------------------------------------------------------
// Scratchpad Face Byte Offset Mapping (Tensix Unpacker Compatibility)
// ------------------------------------------------------------------------------
inline uint32_t get_scratch_byte_offset(uint32_t reg_idx, uint32_t lane) {
    uint32_t face = reg_idx / 8;
    uint32_t r_in_face = reg_idx % 8;
    uint32_t k = r_in_face / 2;
    uint32_t is_odd_reg = r_in_face % 2;
    uint32_t row_in_quartet = lane / 8;
    uint32_t c_pair = lane % 8;

    uint32_t row_in_face = 4 * k + row_in_quartet;
    uint32_t col_in_face = 2 * c_pair + is_odd_reg;

    return face * 512 + row_in_face * 32 + col_in_face * 2;
}
