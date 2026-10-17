#pragma once
// ==============================================================================
// Working 440 code 8CW pipelined simulation, still needs orchestration optimization and FER performance analysis
//
// Tenstorrent QuietBox 2: Tile Matrix 4-Codeword LDPC Decoder Core Architecture
// ==============================================================================

#include <stdint.h>
#include <cmath>
#include <cstring>

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
// ==============================================================================
// Tenstorrent QuietBox 2: Tile Matrix 4-Codeword LDPC Decoder Engine
// Architecture: Decoupled Access-Execute (DAE) with Native 32x32 Tensix Tiles
// Target: Tenstorrent Blackhole (440 Tensix Cores, 2,200 RISC-V Processors)
// ==============================================================================

// ------------------------------------------------------------------------------
// Hardware Compile-Time Test Mode: Identity Pass-Through (R_out = R_in, L_out = L_in)
// Set to 1 to compile kernels in pure memory organization verification mode:
//   - Vnodes initialized to counting numbers (0..2559) matching variable indices.
//   - Check edge messages R are zero.
//   - Compute executes direct identity pass-through (L_out = L_in, R_out = R_in = 0).
//   - Validates Gather -> Memory -> Compute -> Scatter data flow without algorithmic noise.
// Set to 0 for normal production Approximate-Min* LDPC decoding.
// ------------------------------------------------------------------------------
#ifndef LDPC_TEST_PASSTHROUGH_MODE
#define LDPC_TEST_PASSTHROUGH_MODE 0
#endif

// ------------------------------------------------------------------------------
// Hardware Compile-Time Test Mode: R-Edge Dataflow Verification ((V - R_in) + R_out)
// Set to 1 to compile kernels to verify R-message circular FIFO and SIMD ALU:
//   - Vnodes initialized to exact floats 64.0 + (v % 32).
//   - Check edge messages synthesized as R(r) = 1.0 + r.
//   - Iter 0: outputs R(r) to CB_R_MSG, adds R(r) to L.
//   - Iter > 0: unpacks R from CB_R_MSG, computes Q = L - R, L_new = Q + R = L.
//   - Tests circular FIFO page offsets, TRISC destination register mapping, and ALU.
// ------------------------------------------------------------------------------
#ifndef LDPC_TEST_R_DATAFLOW_MODE
#define LDPC_TEST_R_DATAFLOW_MODE 0
#endif

// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------
// Circular Buffer (CB) Allocations in Tensix L1 SRAM (Total 1.5 MB Available)
// Inter-Batch Ping-Pong Architecture: 10 Circular Buffers (236 KB Total, 15.73% L1)
// ------------------------------------------------------------------------------
namespace TileMatrixCB {
    constexpr uint8_t CB_VNODE_A           = 0;   // 20 KB: Active evolving variable posterior LLRs (Batch A: CW 0..3)
    constexpr uint8_t CB_R_MSG_A           = 1;   // 80 KB: 40 tile check messages circular FIFO (Batch A)
    constexpr uint8_t CB_H_OFFSET          = 2;   // 16 KB: Pre-shifted byte offsets (v * 8)
    constexpr uint8_t CB_SCRATCH_GATHER_0  = 3;   // 4 KB: Stage 1 Gather Buffer (Batch A, Double-buffered, 2 KB page)
    constexpr uint8_t CB_SCRATCH_GATHER_1  = 4;   // 4 KB: Stage 1 Gather Buffer (Batch B, Double-buffered, 2 KB page)
    constexpr uint8_t CB_VNODE_B           = 5;   // 20 KB: Active evolving variable posterior LLRs (Batch B: CW 4..7)
    constexpr uint8_t CB_SCRATCH_SCATTER_0 = 16;  // 4 KB: Stage 3 Scatter Buffer (Batch A, Double-buffered, 2 KB page)
    constexpr uint8_t CB_R_MSG_B           = 17;  // 80 KB: 40 tile check messages circular FIFO (Batch B)
    constexpr uint8_t CB_SCRATCH_SCATTER_1 = 18;  // 4 KB: Stage 3 Scatter Buffer (Batch B, Double-buffered, 2 KB page)
    constexpr uint8_t CB_TELEMETRY         = 24;  // 4 KB: Hardware semaphores, mailboxes, telemetry
    constexpr uint32_t TELEMETRY_L1_BASE   = 0x3DC00;

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

// Hardware Semaphore and Inter-Worker Barrier Slots in CB_TELEMETRY (Slots 12..31, 184..199)
namespace SyncSlot {
    constexpr uint32_t BRISC_GATHER_ERR_CW01_A = 12; // Worker 0 (BRISC) gathered negative symbol flag for CW0 & CW1 (Batch A)
    constexpr uint32_t BRISC_GATHER_ERR_CW23_A = 13; // Worker 0 (BRISC) gathered negative symbol flag for CW2 & CW3 (Batch A)
    constexpr uint32_t TRISC_CYCLES        = 14; // TRISC elapsed cycles mailbox
    constexpr uint32_t BRISC_CYCLES        = 15; // BRISC elapsed cycles mailbox
    constexpr uint32_t BATCH_READY         = 16; // Incremented when channel observations are 100% ready in L1
    constexpr uint32_t HOFF_L1_ADDR        = 17; // CB_H_OFFSET L1 base address
    constexpr uint32_t ACTIVE_MASK_A       = 18; // Active codeword mask [3:0] for Batch A
    constexpr uint32_t ACTIVE_MASK_B       = 19; // Active codeword mask [3:0] for Batch B
    constexpr uint32_t EARLY_TERMINATION   = 20; // 1 if both Batch A and Batch B have converged early
    constexpr uint32_t TRISC1_GRANT_STEP   = 21; // Monotonic step grant issued by TRISC1 Orchestrator
    constexpr uint32_t TRISC1_CMD          = 22; // Command enum (MaestroCmd) issued by TRISC1
    constexpr uint32_t DRAIN_ACK_T0        = 23; // TRISC0 acknowledgment of residual FIFO drain
    constexpr uint32_t GATHER_START        = 24; // Signaled by BRISC when SCRATCH_GATHER_PTR is published
    constexpr uint32_t SCRATCH_GATHER_PTR  = 25; // Active gather L1 write pointer
    constexpr uint32_t SCRATCH_SCATTER_PTR = 26; // Active scatter L1 read pointer
    constexpr uint32_t VNODE_L1_ADDR_A     = 27; // Batch A variable node L1 base address
    constexpr uint32_t VNODE_L1_ADDR_B     = 28; // Batch B variable node L1 base address
    constexpr uint32_t BATCH_BARRIER_BRISC  = 29; // BRISC batch rendezvous
    constexpr uint32_t BATCH_BARRIER_NCRISC = 30; // NCRISC batch rendezvous
    constexpr uint32_t BATCH_BARRIER_TRISC  = 31; // TRISC batch rendezvous

    // Backward-compatibility aliases
    constexpr uint32_t ACTIVE_MASK         = ACTIVE_MASK_A;
    constexpr uint32_t VNODE_L1_ADDR       = VNODE_L1_ADDR_A;
    constexpr uint32_t BRISC_GATHER_ERR_CW01 = BRISC_GATHER_ERR_CW01_A;
    constexpr uint32_t BRISC_GATHER_ERR_CW23 = BRISC_GATHER_ERR_CW23_A;

    // 4-Worker Face Parallel Done Barriers (Slots 184..191)
    constexpr uint32_t GATHER_DONE_W0      = 184; // Worker 0 (BRISC: Face 0) gather done
    constexpr uint32_t GATHER_DONE_W1      = 185; // Worker 1 (NCRISC: Face 1) gather done
    constexpr uint32_t GATHER_DONE_T0      = 186; // Worker 2 (TRISC0: Face 2) gather done
    constexpr uint32_t GATHER_DONE_T2      = 187; // Worker 3 (TRISC2: Face 3) gather done
    constexpr uint32_t SCATTER_DONE_W0     = 188; // Worker 0 (BRISC: Face 0) scatter done
    constexpr uint32_t SCATTER_DONE_W1     = 189; // Worker 1 (NCRISC: Face 1) scatter done
    constexpr uint32_t SCATTER_DONE_T0     = 190; // Worker 2 (TRISC0: Face 2) scatter done
    constexpr uint32_t SCATTER_DONE_T2     = 191; // Worker 3 (TRISC2: Face 3) scatter done
    constexpr uint32_t GATHER_ALL_DONE     = 192; // All 4 faces gathered into scratch_in
    constexpr uint32_t SCATTER_START       = 193; // Signaled by NCRISC when SCRATCH_SCATTER_PTR is published

    // Satellite Parity Reports to TRISC1 (Slots 194..195)
    constexpr uint32_t PARITY_REPORT_W0    = 194; // Worker 0 reports info error mask for CW0, CW1
    constexpr uint32_t PARITY_REPORT_W1    = 195; // Worker 1 reports info error mask for CW2, CW3
    constexpr uint32_t BRISC_SCAN_DONE_A   = PARITY_REPORT_W0;
    constexpr uint32_t BRISC_SCAN_DONE_B   = PARITY_REPORT_W1;
    constexpr uint32_t ITER_DONE_A         = 196;
    constexpr uint32_t ITER_DONE_B         = 197;
    constexpr uint32_t ITER_DONE           = ITER_DONE_A;
    constexpr uint32_t BRISC_SCAN_DONE     = BRISC_SCAN_DONE_A;
    constexpr uint32_t DRAIN_ACK_W1        = 198; // NCRISC acknowledgment of residual scatter FIFO drain
}

// Telemetry Buffer Layout in CB_TELEMETRY (1024 uint32 words)
namespace TelemSlot {
    // Primary Simulation Totals (written by NCRISC at exit)
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
    // Slots 12..31 are reserved for live hardware SyncSlot semaphores

    // NCRISC (Worker 1) Tile Breakdown (written by NCRISC at exit)
    constexpr uint32_t NC_GATHER           = 32;
    constexpr uint32_t NC_WAIT             = 33;
    constexpr uint32_t NC_SCATTER          = 34;
    constexpr uint32_t NC_SCAT_START_BAR   = 35;
    constexpr uint32_t NC_SCAT_LOOP        = 36;
    constexpr uint32_t NC_SCAT_RCOPY       = 37;
    constexpr uint32_t NC_SCAT_END_BAR     = 38;
    constexpr uint32_t NC_SCAT_POP         = 39;
    constexpr uint32_t NC_SCAT_TILE0       = 40;
    constexpr uint32_t NC_SCAT_OTHER       = 41;
    constexpr uint32_t NC_GATHER_LOOP      = 42;

    // BRISC (Worker 0) Tile Breakdown (written by BRISC at exit)
    constexpr uint32_t BR_GATHER           = 48;
    constexpr uint32_t BR_WAIT             = 49;
    constexpr uint32_t BR_SCATTER          = 50;
    constexpr uint32_t BR_SCAT_LOOP        = 51;
    constexpr uint32_t BR_SCAT_RCOPY       = 52;
    constexpr uint32_t BR_SCAT_END_BAR     = 53;
    constexpr uint32_t BR_GATHER_LOOP      = 54;

    // Check Node Degrees (pre-loaded / verified)
    constexpr uint32_t DEG_BASE            = 64;  // 64..103

    // Diagnostics (written by NCRISC on Batch 0 and exit)
    constexpr uint32_t INIT_NEG_CW0        = 104; // 104..107 (CW 0..3)
    constexpr uint32_t INIT_BF16_W01       = 108;
    constexpr uint32_t INIT_BF16_W23       = 109;
    constexpr uint32_t MU_BITS             = 110;
    constexpr uint32_t SIGMA_BITS          = 111;
    constexpr uint32_t LAST_BATCH_BE_CW0   = 112; // 112..119 (CW 0..7)
    constexpr uint32_t LAST_BATCH_ITER_CW0 = 120; // 120..127 (CW 0..7)
    constexpr uint32_t FAILING_CW0_COUNT   = 128;
    constexpr uint32_t FAILING_CW1_COUNT   = 129;
    constexpr uint32_t FAILING_CW0_BASE    = 130; // 130..157 (28 max)
    constexpr uint32_t FAILING_CW1_BASE    = 158; // 158..185 (28 max)

    // Trapped Failing Batch Diagnostics (Slots 210..213)
    constexpr uint32_t FIRST_FAILING_BATCH = 210;
    constexpr uint32_t FIRST_FAILING_CW    = 211;
    constexpr uint32_t FIRST_FAILING_BE    = 212;
    constexpr uint32_t FIRST_FAILING_MASK  = 213;
}

// ------------------------------------------------------------------------------
// Hardware PRNG for Channel AWGN Generation (Xoshiro128Plus)
// ------------------------------------------------------------------------------
struct Xoshiro128Plus {
    uint32_t s[4];
    void init(uint32_t seed) {
        uint32_t z = seed;
        for (int i = 0; i < 4; ++i) {
            z += 0x9e3779b9;
            z = (z ^ (z >> 16)) * 0x85ebca6b;
            z = (z ^ (z >> 13)) * 0xc2b2ae35;
            s[i] = z ^ (z >> 16);
            if (s[i] == 0) s[i] = 1;
        }
    }
    inline uint32_t next() {
        const uint32_t result = s[0] + s[3];
        const uint32_t t = s[1] << 9;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t;
        s[3] = (s[3] << 11) | (s[3] >> 21);
        return result;
    }
    inline float next_uniform() {
        uint32_t v = next();
        if (v == 0) v = 1;
        return static_cast<float>(v) * 2.3283064365386963e-10f;
    }
    // Advance PRNG by 2^64 draws via jump()
    inline void jump() {
        static const uint32_t JUMP[] = { 0x8764000b, 0xf542d2d3, 0x6fa035c3, 0x77f2db5b };
        uint32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (int i = 0; i < 4; i++) {
            for (int b = 0; b < 32; b++) {
                if (JUMP[i] & (1U << b)) {
                    s0 ^= s[0]; s1 ^= s[1]; s2 ^= s[2]; s3 ^= s[3];
                }
                next();
            }
        }
        s[0] = s0; s[1] = s1; s[2] = s2; s[3] = s3;
    }
    // Advance PRNG by 2^96 draws via long_jump() (for inter-core partitioning across 440 cores)
    inline void long_jump() {
        static const uint32_t LONG_JUMP[] = { 0xb523952e, 0x0b6f099f, 0xccf5a0ef, 0x1c580662 };
        uint32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (int i = 0; i < 4; i++) {
            for (int b = 0; b < 32; b++) {
                if (LONG_JUMP[i] & (1U << b)) {
                    s0 ^= s[0]; s1 ^= s[1]; s2 ^= s[2]; s3 ^= s[3];
                }
                next();
            }
        }
        s[0] = s0; s[1] = s1; s[2] = s2; s[3] = s3;
    }
};

// Helper: Compute deterministic, orthogonal seed state for (core_id, batch_pair_idx, batch_sub)
inline Xoshiro128Plus get_orthogonal_rng_state(uint32_t global_seed, uint32_t core_id, uint32_t batch_pair_idx, uint32_t batch_sub) {
    Xoshiro128Plus rng;
    rng.init(global_seed);
    // Tier 1: Core offset (core_id * 2^96)
    for (uint32_t c = 0; c < core_id; ++c) {
        rng.long_jump();
    }
    // Tier 2: Inter-batch pair offset (batch_pair_idx * 2 * 2^64)
    for (uint32_t p = 0; p < batch_pair_idx; ++p) {
        rng.jump();
        rng.jump();
    }
    // Tier 3: Intra-pair offset (Batch B is +1 jump)
    if (batch_sub == 1) {
        rng.jump();
    }
    return rng;
}


// ------------------------------------------------------------------------------
// Fast Branchless SFPU-Style Transcendental Functions
// Matches the minimax & Maclaurin polynomials of the Blackhole Tensix SFPU
// ------------------------------------------------------------------------------
inline float sfpu_fast_log(float a) {
    if (a <= 1e-7f) return -16.11809565f;
    uint32_t u = float_as_uint(a);
    int32_t exp_val = static_cast<int32_t>((u >> 23) & 0xFF) - 127;
    uint32_t m_bits = (u & 0x007FFFFF) | 0x3F800000;
    float m = uint_as_float(m_bits);
    float z = m - 1.0f;

    float p = -0.0528f;
    p = p * z + 0.1264f;
    p = p * z - 0.1678f;
    p = p * z + 0.2057f;
    p = p * z - 0.2490f;
    p = p * z + 0.3328f;
    p = p * z - 0.5f;
    p = p * z + 1.0f;
    float ln_m = p * z;

    return ln_m + static_cast<float>(exp_val) * 0.69314718f;
}

inline float sfpu_fast_sqrt(float val) {
    if (val <= 1e-7f) return 0.0f;
    uint32_t u = float_as_uint(val);
    uint32_t magic = 0x5F375A86U;
    uint32_t approx_u = magic - (u >> 1);
    float approx = uint_as_float(approx_u);
    float neg_half_val = val * -0.5f;
    approx = ((approx * approx) * neg_half_val + 1.5f) * approx;
    approx = ((approx * approx) * neg_half_val + 1.5f) * approx;
    return approx * val;
}

inline float sfpu_fast_sin(float val) {
    constexpr float PI = 3.14159265358979323846f;
    constexpr float TWO_PI = 2.0f * PI;
    while (val > PI) val -= TWO_PI;
    while (val < -PI) val += TWO_PI;
    float tmp = val;
    float out = tmp;
    tmp = tmp * val * val; out += -0.16666666666666666f * tmp;
    tmp = tmp * val * val; out += 0.008333333333333333f * tmp;
    tmp = tmp * val * val; out += -0.0001984126984126984f * tmp;
    tmp = tmp * val * val; out += 0.0000027557319223985f * tmp;
    return out;
}

inline float sfpu_fast_cos(float val) {
    constexpr float PI = 3.14159265358979323846f;
    constexpr float TWO_PI = 2.0f * PI;
    while (val > PI) val -= TWO_PI;
    while (val < -PI) val += TWO_PI;
    float out = 1.0f;
    float tmp = val * val; out += -0.5f * tmp;
    tmp = tmp * val * val; out += 0.041666666666666664f * tmp;
    tmp = tmp * val * val; out += -0.001388888888888889f * tmp;
    tmp = tmp * val * val; out += 0.0000248015873015873f * tmp;
    return out;
}

inline float fast_rsqrt(float x) {
    uint32_t u = float_as_uint(x);
    u = 0x5F3759DF - (u >> 1);
    float y = uint_as_float(u);
    float xhalf = 0.5f * x;
    y = y * (1.5f - xhalf * y * y);
    y = y * (1.5f - xhalf * y * y);
    return y;
}

inline float fast_sqrt(float x) {
    return x * fast_rsqrt(x);
}

inline float fast_ln(float s) {
    uint32_t u = float_as_uint(s);
    int32_t exp = static_cast<int32_t>((u >> 23) & 0xFF) - 127;
    u = (u & 0x007FFFFF) | 0x3F800000;
    float m = uint_as_float(u) - 1.0f;
    float p = -0.02397957f;
    p = p * m + 0.10150005f;
    p = p * m - 0.21029369f;
    p = p * m + 0.32529514f;
    p = p * m - 0.49937260f;
    p = p * m + 0.99999183f;
    p = p * m;
    return static_cast<float>(exp) * 0.69314718056f + p;
}

inline void generate_awgn_pair(Xoshiro128Plus& rng, float mu_llr, float sigma_llr, float& llr0, float& llr1) {
    constexpr float INT32_TO_UNIT = 4.656612875245797e-10f;
    float v1, v2, s;
    do {
        int32_t u1 = static_cast<int32_t>(rng.next());
        int32_t u2 = static_cast<int32_t>(rng.next());
        v1 = static_cast<float>(u1) * INT32_TO_UNIT;
        v2 = static_cast<float>(u2) * INT32_TO_UNIT;
        s = v1 * v1 + v2 * v2;
    } while (s >= 1.0f || s <= 1e-15f);

    float rsqrt_s = fast_rsqrt(s);
    float ln_s = fast_ln(s);
    float scale = sigma_llr * fast_sqrt(-2.0f * ln_s) * rsqrt_s;
    llr0 = mu_llr + v1 * scale;
    llr1 = mu_llr + v2 * scale;
}

inline float bf16_to_fp32(uint16_t bf) {
    uint32_t u = static_cast<uint32_t>(bf) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

inline uint16_t fp32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    uint32_t lsb = (u >> 16) & 1;
    uint32_t bias = 0x7FFF + lsb;
    u += bias;
    return static_cast<uint16_t>(u >> 16);
}

inline void init_vnodes_worker0(
    uint32_t vnode_l1,
    float mu_llr,
    float sigma_llr,
    Xoshiro128Plus& rng
) {
    bool bench = (sigma_llr <= 0.0f);
    uint16_t mu_bf = fp32_to_bf16(mu_llr);
    uint32_t mu_word = static_cast<uint32_t>(mu_bf) | (static_cast<uint32_t>(mu_bf) << 16);

    for (uint32_t t = 0; t < 8; ++t) {
        uint32_t tile_base = vnode_l1 + t * 2048;
        for (uint32_t p = 0; p < 16; ++p) {
            auto* p32 = reinterpret_cast<uint32_t*>(tile_base + 128 * p);
            if (bench) {
                #pragma GCC unroll 4
                for (int i = 0; i < 16; ++i) {
                    p32[i] = mu_word;
                }
            } else {
                for (int k = 0; k < 4; ++k) {
                    for (int cw = 0; cw < 4; ++cw) {
                        float l_even, l_odd;
                        generate_awgn_pair(rng, mu_llr, sigma_llr, l_even, l_odd);
                        uint16_t bf_e = fp32_to_bf16(l_even);
                        uint16_t bf_o = fp32_to_bf16(l_odd);
                        p32[4 * k + cw] = static_cast<uint32_t>(bf_e) | (static_cast<uint32_t>(bf_o) << 16);
                    }
                }
            }
        }
    }
    for (uint32_t t = 8; t < 10; ++t) {
        uint32_t tile_base = vnode_l1 + t * 2048;
        for (uint32_t p = 0; p < 16; ++p) {
            auto* p32 = reinterpret_cast<uint32_t*>(tile_base + 128 * p);
            #pragma GCC unroll 4
            for (int i = 0; i < 16; ++i) {
                p32[i] = 0;
            }
        }
    }
}

inline void init_vnodes_worker1(
    uint32_t vnode_l1,
    float mu_llr,
    float sigma_llr,
    Xoshiro128Plus& rng
) {
    bool bench = (sigma_llr <= 0.0f);
    uint16_t mu_bf = fp32_to_bf16(mu_llr);
    uint32_t mu_word = static_cast<uint32_t>(mu_bf) | (static_cast<uint32_t>(mu_bf) << 16);

    for (uint32_t t = 0; t < 8; ++t) {
        uint32_t tile_base = vnode_l1 + t * 2048;
        for (uint32_t p = 0; p < 16; ++p) {
            auto* p32 = reinterpret_cast<uint32_t*>(tile_base + 128 * p + 64);
            if (bench) {
                #pragma GCC unroll 4
                for (int i = 0; i < 16; ++i) {
                    p32[i] = mu_word;
                }
            } else {
                for (int k = 0; k < 4; ++k) {
                    for (int cw = 0; cw < 4; ++cw) {
                        float l_even, l_odd;
                        generate_awgn_pair(rng, mu_llr, sigma_llr, l_even, l_odd);
                        uint16_t bf_e = fp32_to_bf16(l_even);
                        uint16_t bf_o = fp32_to_bf16(l_odd);
                        p32[4 * k + cw] = static_cast<uint32_t>(bf_e) | (static_cast<uint32_t>(bf_o) << 16);
                    }
                }
            }
        }
    }
    for (uint32_t t = 8; t < 10; ++t) {
        uint32_t tile_base = vnode_l1 + t * 2048;
        for (uint32_t p = 0; p < 16; ++p) {
            auto* p32 = reinterpret_cast<uint32_t*>(tile_base + 128 * p + 64);
            #pragma GCC unroll 4
            for (int i = 0; i < 16; ++i) {
                p32[i] = 0;
            }
        }
    }
}


// ------------------------------------------------------------------------------
// Tile Geometry & Systolic Parameters
// ------------------------------------------------------------------------------
constexpr uint32_t TILE_WIDTH               = 32;  // Native Tensix Matrix FPU Tile Width
constexpr uint32_t TILE_HEIGHT              = 32;  // Native Tensix Matrix FPU Tile Height
constexpr uint32_t TILE_ELEMENTS            = 1024; // 32 * 32
constexpr uint32_t NUM_CW_PARALLEL          = 4;   // 4 parallel codewords processed in lockstep

// Check-Node Groupings per Tile Pass
constexpr uint32_t CHECKS_PER_TILE_DEG6     = 32;  // 32 checks * 6 edges = 192 edges (4 faces x 8 checks)
constexpr uint32_t CHECKS_PER_TILE_DEG3     = 64;  // 64 checks * 3 edges = 192 edges (4 faces x 16 checks)
constexpr uint32_t BOUNDARY_CHECKS_DEG6     = 32;  // 32 checks * 6 edges = 192 edges
constexpr uint32_t BOUNDARY_CHECKS_DEG3     = 64;  // 64 checks * 3 edges = 192 edges

// Circulant-Aware Tile Snapping for CCSDS AR4JA Rate-1/2 (p = 512, p' = 128)
// 16 tiles of 32 checks = 512 checks per deg-6 circulant block (2 blocks = 32 tiles)
// 8 tiles of 64 checks = 512 checks per deg-3 circulant block (1 block = 8 tiles)
constexpr uint32_t TILES_PER_DEG6_BLOCK     = 16;  // 16 * 32 = 512 checks (2 blocks = 32 tiles)
constexpr uint32_t TILES_PER_DEG3_BLOCK     = 8;   // 8 * 64 = 512 checks (1 block = 8 tiles)
constexpr uint32_t TOTAL_TILES_PER_ITER     = 40;  // 32 degree-6 tiles + 8 degree-3 tiles

// Active Elements per Tile Type
constexpr uint32_t ACTIVE_EDGES_FULL_TILE   = 192; // 192 edges * 4 codewords = 768 elements
constexpr uint32_t ACTIVE_ROWS_FULL_TILE    = 24;  // 24 rows * 32 cols = 768 elements

// Native Tensix Systolic Tile Element Addressing (32x32 bfloat16 tile format)
inline uint32_t get_tile_index(uint32_t r_dst, uint32_t lane) {
    uint32_t pair = r_dst / 2;
    uint32_t is_odd_reg = r_dst % 2;
    uint32_t row = (lane < 16) ? (2 * pair) : (2 * pair + 1);
    uint32_t col = (lane < 16) ? (2 * lane + is_odd_reg) : (2 * (lane - 16) + is_odd_reg);
    return row * 32 + col;
}

// ------------------------------------------------------------------------------
// 4-Codeword Packed Data Structures
// ------------------------------------------------------------------------------
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

// ------------------------------------------------------------------------------
// TT-Metal Native 32x32 Tile Face Typed Struct Overlay
// ------------------------------------------------------------------------------
#pragma pack(push, 2)
struct alignas(8) FaceRow16 {
    // 16 bfloat16 elements = four 4-codeword groups = 32 bytes
    Packed4CW_BF16 grp[4]; 
};

struct alignas(32) TileFace16x16 {
    FaceRow16 row[16]; // 16 rows x 32 bytes = 512 bytes
};

struct alignas(2048) TensixTile32x32 {
    TileFace16x16 face[4]; // 4 faces x 512 bytes = 2,048 bytes
};
#pragma pack(pop)

#define TT_FACE_OFFSET(row, col) ( \
    (((uint32_t)(row) & 16) << 6) | \
    (((uint32_t)(col) & 16) << 5) | \
    (((uint32_t)(row) & 15) << 5) | \
    (((uint32_t)(col) & 15) << 1)   \
)

// ------------------------------------------------------------------------------
// Compile-Time Edge-Per-Register Scratchpad LUT (Face-Aligned Architecture)
// Maps (check block, edge index) directly to the 4 physical tile faces:
//   Worker 0 -> Face 0 (0..23) and Face 2 (24..47)
//   Worker 1 -> Face 1 (0..23) and Face 3 (24..47)
// ------------------------------------------------------------------------------
struct FaceScratchLUT {
    uint16_t w0_even[48];
    uint16_t w0_odd[48];
    uint16_t w1_even[48];
    uint16_t w1_odd[48];

    constexpr FaceScratchLUT() : w0_even{}, w0_odd{}, w1_even{}, w1_odd{} {
        for (uint32_t p = 0; p < 48; ++p) {
            uint32_t face_w0 = (p < 24) ? 0 : 2;
            uint32_t face_w1 = (p < 24) ? 1 : 3;
            uint32_t p_in_half = (p < 24) ? p : (p - 24);
            uint32_t d  = p_in_half / 4;
            uint32_t kh = p_in_half % 4;

            // Worker 0: strictly Faces 0 & 2
            w0_even[p] = static_cast<uint16_t>(face_w0 * 512 + (2 * d) * 32 + kh * 8);
            w0_odd[p]  = static_cast<uint16_t>(face_w0 * 512 + (2 * d + 1) * 32 + kh * 8);

            // Worker 1: strictly Faces 1 & 3
            w1_even[p] = static_cast<uint16_t>(face_w1 * 512 + (2 * d) * 32 + kh * 8);
            w1_odd[p]  = static_cast<uint16_t>(face_w1 * 512 + (2 * d + 1) * 32 + kh * 8);
        }
    }
};

inline constexpr FaceScratchLUT SCRATCH_FACE_LUT;

inline uint32_t get_tile_vnode_byte_offset(uint32_t v) {
    uint32_t t = v / 256;
    uint32_t v_loc = v % 256;
    uint32_t row = v_loc / 8;
    uint32_t k = v_loc % 8;
    
    uint32_t face = (row < 16) ? ((k < 4) ? 0 : 1) : ((k < 4) ? 2 : 3);
    uint32_t col_grp = (k < 4) ? k : (k - 4);
    
    return t * 2048 + (face * 512) + ((row & 15) << 5) + (col_grp << 3);
}

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

struct TileIndexLUT {
    uint16_t base_idx[32][8];
    constexpr TileIndexLUT() : base_idx{} {
        for (uint32_t r = 0; r < 32; ++r) {
            uint32_t pair = r / 2;
            uint32_t is_odd_reg = r % 2;
            for (uint32_t c_loc = 0; c_loc < 8; ++c_loc) {
                uint32_t lane = c_loc * 4;
                uint32_t row = (lane < 16) ? (2 * pair) : (2 * pair + 1);
                uint32_t col = (lane < 16) ? (2 * lane + is_odd_reg) : (2 * (lane - 16) + is_odd_reg);
                base_idx[r][c_loc] = static_cast<uint16_t>(row * 32 + col);
            }
        }
    }
};

inline constexpr TileIndexLUT TILE_BASE_LUT;

// Flattened Linear Base Index LUT for 1-D Gather and Scatter Loops (Zero Div / Modulo Overhead)
struct StaticTileBaseLUT {
    uint16_t deg6[240];
    uint16_t deg3[240];

    constexpr StaticTileBaseLUT() : deg6{}, deg3{} {
        // Degree-6: 40 checks * 6 edges
        uint32_t e6 = 0;
        for (uint32_t b = 0; b < 5; ++b) {
            for (uint32_t c_loc = 0; c_loc < 8; ++c_loc) {
                for (uint32_t d = 0; d < 6; ++d) {
                    uint32_t r = b * 6 + d;
                    uint32_t pair = r / 2;
                    uint32_t is_odd_reg = r % 2;
                    uint32_t lane = c_loc * 4;
                    uint32_t row = (lane < 16) ? (2 * pair) : (2 * pair + 1);
                    uint32_t col = (lane < 16) ? (2 * lane + is_odd_reg) : (2 * (lane - 16) + is_odd_reg);
                    deg6[e6++] = static_cast<uint16_t>(row * 32 + col);
                }
            }
        }

        // Degree-3: 80 checks * 3 edges
        uint32_t e3 = 0;
        for (uint32_t b = 0; b < 10; ++b) {
            for (uint32_t c_loc = 0; c_loc < 8; ++c_loc) {
                for (uint32_t d = 0; d < 3; ++d) {
                    uint32_t r = b * 3 + d;
                    uint32_t pair = r / 2;
                    uint32_t is_odd_reg = r % 2;
                    uint32_t lane = c_loc * 4;
                    uint32_t row = (lane < 16) ? (2 * pair) : (2 * pair + 1);
                    uint32_t col = (lane < 16) ? (2 * lane + is_odd_reg) : (2 * (lane - 16) + is_odd_reg);
                    deg3[e3++] = static_cast<uint16_t>(row * 32 + col);
                }
            }
        }
    }
};

inline constexpr StaticTileBaseLUT STATIC_TILE_LUT;

// ------------------------------------------------------------------------------
// Runtime Tile Schedule Representation (Generated parametric from .chinn.out)
// ------------------------------------------------------------------------------
struct TileScheduleDescriptor {
    uint16_t check_start_idx;  // Starting check node index in matrix H
    uint8_t  check_degree;      // 6 or 3
    uint8_t  num_checks;        // 40 (full deg-6), 80 (full deg-3), or 32 (boundary)
    uint16_t num_edges;         // num_checks * check_degree (240, 192, or 96)
    uint16_t h_offset_start;    // Byte offset into CB_H_OFFSET
    uint16_t r_msg_page_offset; // Page offset into CB_R_MSG
};

// ------------------------------------------------------------------------------
// Batch Telemetry & Convergence Stats
// ------------------------------------------------------------------------------
struct alignas(16) TileMatrixBatchStats {
    uint32_t total_blocks;      // Total codewords simulated (batches * 4)
    uint32_t bit_errors[4];     // Bit error count per codeword
    uint32_t frame_errors[4];   // Frame error flag per codeword (0 or 1)
    uint32_t iterations_taken[4];// Iterations to convergence per codeword
    uint32_t early_terminated;  // 4-bit bitmask of codewords that converged early
    uint32_t cycles_elapsed;    // Total clock cycles taken by batch
};

// ------------------------------------------------------------------------------


// ------------------------------------------------------------------------------
// Approximate-Min* Formulation (Dr. Christopher R. Jones, MILCOM 2003)
// 2-Region Piecewise Linear Correction (or 3-Region when LDPC_USE_3PIECE_AMIN_STAR is defined):
// 2-Piece:
//   g(Δ) = 0.69315 - 0.40 * Δ        if Δ < 0.5
//          0.49315 - 0.328 * (Δ - 0.5) if 0.5 <= Δ < 2.0
//          0.0                         if Δ >= 2.0
// 3-Piece (Optimal Convex Hull):
//   Piece 1: y = 0.67862 - 0.39622 * Δ
//   Piece 2: y = 0.48710 - 0.18297 * Δ
//   Piece 3: y = 0.17614 - 0.04029 * Δ
// ------------------------------------------------------------------------------
inline float approx_min_star_scalar(float a, float b) {
    float min_val = (a < b) ? a : b;
    float max_val = (a > b) ? a : b;
    float diff = max_val - min_val;
#if defined(LDPC_USE_3PIECE_AMIN_STAR)
    float line1 = 0.67862f - 0.39622f * diff;
    float line2 = 0.48710f - 0.18297f * diff;
    float line3 = 0.17614f - 0.04029f * diff;
    float corr = (line1 > line2) ? line1 : line2;
    if (line3 > corr) corr = line3;
    if (corr < 0.0f) corr = 0.0f;
    float res = min_val - corr;
    return (res > 0.0f) ? res : 0.0f;
#else
    float line1 = 0.69315f - 0.40f * diff;
    float line2 = 0.65715f - 0.328f * diff;
    float corr = (line1 > line2) ? line1 : line2;
    if (corr < 0.0f) corr = 0.0f;
    float res = min_val - corr;
    return (res > 0.0f) ? res : 0.0f;
#endif
}

// ------------------------------------------------------------------------------
// Hardware 1-Face (24 Edge Pairs = 48 Edges) Fast Data Movement Engines
// Optimized for Zero-Stack-Spill RV32 Execution on Blackhole RISC-V Workers
// ------------------------------------------------------------------------------

inline void gather_face_24pairs(uint32_t dst_face_addr, const uint32_t* p_hoff_24, const char* vnode_bytes_ptr) {
    auto* dst_even = reinterpret_cast<uint32_t*>(dst_face_addr);
    const uint32_t* p_h = p_hoff_24;

    #pragma GCC unroll 6
    for (uint32_t d = 0; d < 6; ++d) {
        auto* dst_odd = dst_even + 8;

        // Block 0: Pairs 0 and 1 (aligned 128-bit block stores)
        {
            uint32_t off0 = p_h[0];
            uint32_t off1 = p_h[1];

            uint32_t off0_e = off0 & 0xFFFF;
            uint32_t off0_o = off0 >> 16;
            uint32_t off1_e = off1 & 0xFFFF;
            uint32_t off1_o = off1 >> 16;

            const auto* s0_e = reinterpret_cast<const uint32_t*>(vnode_bytes_ptr + off0_e);
            const auto* s0_o = reinterpret_cast<const uint32_t*>(vnode_bytes_ptr + off0_o);
            const auto* s1_e = reinterpret_cast<const uint32_t*>(vnode_bytes_ptr + off1_e);
            const auto* s1_o = reinterpret_cast<const uint32_t*>(vnode_bytes_ptr + off1_o);

            dst_even[0] = s0_e[0]; dst_even[1] = s0_e[1];
            dst_even[2] = s1_e[0]; dst_even[3] = s1_e[1];
            dst_odd[0]  = s0_o[0]; dst_odd[1]  = s0_o[1];
            dst_odd[2]  = s1_o[0]; dst_odd[3]  = s1_o[1];
        }

        // Block 1: Pairs 2 and 3 (aligned 128-bit block stores)
        {
            uint32_t off2 = p_h[2];
            uint32_t off3 = p_h[3];

            uint32_t off2_e = off2 & 0xFFFF;
            uint32_t off2_o = off2 >> 16;
            uint32_t off3_e = off3 & 0xFFFF;
            uint32_t off3_o = off3 >> 16;

            const auto* s2_e = reinterpret_cast<const uint32_t*>(vnode_bytes_ptr + off2_e);
            const auto* s2_o = reinterpret_cast<const uint32_t*>(vnode_bytes_ptr + off2_o);
            const auto* s3_e = reinterpret_cast<const uint32_t*>(vnode_bytes_ptr + off3_e);
            const auto* s3_o = reinterpret_cast<const uint32_t*>(vnode_bytes_ptr + off3_o);

            dst_even[4] = s2_e[0]; dst_even[5] = s2_e[1];
            dst_even[6] = s3_e[0]; dst_even[7] = s3_e[1];
            dst_odd[4]  = s2_o[0]; dst_odd[5]  = s2_o[1];
            dst_odd[6]  = s3_o[0]; dst_odd[7]  = s3_o[1];
        }

        dst_even += 16;
        p_h += 4;
    }

    // Zero out the 4 padding rows (Rows 12..15 = 128 bytes = 8 x 128-bit aligned vector stores)
    typedef uint32_t v4ui __attribute__((vector_size(16)));
    auto* pad128 = reinterpret_cast<v4ui*>(dst_even);
    v4ui zero = {0, 0, 0, 0};
    #pragma GCC unroll 8
    for (int i = 0; i < 8; ++i) {
        pad128[i] = zero;
    }
}

inline void scatter_face_24pairs_fast(uint32_t src_face_addr, const uint32_t* p_hoff_24, char* vnode_out_ptr) {
    const auto* src_even = reinterpret_cast<const uint32_t*>(src_face_addr);
    const uint32_t* p_h = p_hoff_24;

    #pragma GCC unroll 6
    for (uint32_t d = 0; d < 6; ++d) {
        const auto* src_even64 = reinterpret_cast<const uint64_t*>(src_even);
        const auto* src_odd64  = reinterpret_cast<const uint64_t*>(src_even + 8);

        #pragma GCC unroll 4
        for (uint32_t p = 0; p < 4; ++p) {
            uint32_t off = p_h[p];
            auto* d_e64 = reinterpret_cast<uint64_t*>(vnode_out_ptr + (off & 0xFFFF));
            auto* d_o64 = reinterpret_cast<uint64_t*>(vnode_out_ptr + (off >> 16));

            *d_e64 = src_even64[p];
            *d_o64 = src_odd64[p];
        }

        src_even += 16;
        p_h += 4;
    }
}

inline void scatter_face_24pairs_masked(uint32_t src_face_addr, const uint32_t* p_hoff_24, char* vnode_out_ptr, uint8_t active_mask) {
    uint32_t mask_low  = ((active_mask & 1) ? 0x0000FFFF : 0) | ((active_mask & 2) ? 0xFFFF0000 : 0);
    uint32_t mask_high = ((active_mask & 4) ? 0x0000FFFF : 0) | ((active_mask & 8) ? 0xFFFF0000 : 0);
    bool inv_low   = (mask_low  != 0xFFFFFFFF);
    bool inv_high  = (mask_high != 0xFFFFFFFF);

    const auto* src_face = reinterpret_cast<const uint32_t*>(src_face_addr);
    for (uint32_t d = 0; d < 6; ++d) {
        const auto* src_even = src_face + d * 16;
        const auto* src_odd  = src_even + 8;
        const uint32_t* p_h = p_hoff_24 + d * 4;

        #pragma GCC unroll 4
        for (uint32_t p = 0; p < 4; ++p) {
            uint32_t off = p_h[p];
            auto* d_e = reinterpret_cast<uint32_t*>(vnode_out_ptr + (off & 0xFFFF));
            auto* d_o = reinterpret_cast<uint32_t*>(vnode_out_ptr + (off >> 16));
            uint32_t e0 = src_even[2 * p];
            uint32_t e1 = src_even[2 * p + 1];
            uint32_t o0 = src_odd[2 * p];
            uint32_t o1 = src_odd[2 * p + 1];

            if (inv_low)  { e0 = (e0 & mask_low)  | rv32_andn(d_e[0], mask_low);  o0 = (o0 & mask_low)  | rv32_andn(d_o[0], mask_low); }
            if (inv_high) { e1 = (e1 & mask_high) | rv32_andn(d_e[1], mask_high); o1 = (o1 & mask_high) | rv32_andn(d_o[1], mask_high); }

            d_e[0] = e0; d_e[1] = e1;
            d_o[0] = o0; d_o[1] = o1;
        }
    }
}

// ------------------------------------------------------------------------------
// Architectural Note (Production Decoder vs. Monte Carlo Simulator):
// In an operational communications receiver, once a codeword converges, its decoded information bits
// must either be frozen in-place or transferred to an egress FIFO before subsequent iterations can mutate them.
// In Monte Carlo simulation, our sole objective is computing statistical metrics (FER, BER, and convergence iterations).
// Because the 4 codewords occupy independent vector lanes and circulant slices with zero cross-talk, letting
// an already-converged codeword continue to mutate in L1 SRAM while unconverged companions finish has zero
// impact on companion convergence, and zero impact on simulation metrics (since convergence is an absorbing
// state latched in iter_converged[c], and cw_errs[c] is strictly 0 for converged codewords).
//
// Therefore, we execute scatter_face_24pairs_fast 100% of the time, completely eliminating the Read-Modify-Write
// (RMW) overhead, random L1 loads, and ALU dependency stalls of scatter_face_24pairs_masked.
// ------------------------------------------------------------------------------
inline void scatter_face_24pairs(uint32_t src_face_addr, const uint32_t* p_hoff_24, char* vnode_out_ptr, uint8_t active_mask = 0x0F) {
    (void)active_mask; // Unused in Monte Carlo simulation mode
    scatter_face_24pairs_fast(src_face_addr, p_hoff_24, vnode_out_ptr);
}

// ------------------------------------------------------------------------------
// Hardware 2-Face (48 Edge Pairs = 96 Edges) Fast Data Movement Engines
// Worker 0 gathers/scatters Faces 0 & 2 (96 edges x 4 CWs = 384 elements)
// Worker 1 gathers/scatters Faces 1 & 3 (96 edges x 4 CWs = 384 elements)
// ------------------------------------------------------------------------------
inline void gather_face_48pairs_w0(uint32_t dst_scratch_addr, const uint32_t* p_hoff_48, const char* vnode_bytes_ptr) {
    gather_face_24pairs(dst_scratch_addr + 0,    p_hoff_48 + 0,  vnode_bytes_ptr); // Face 0
    gather_face_24pairs(dst_scratch_addr + 1024, p_hoff_48 + 24, vnode_bytes_ptr); // Face 2
}

inline void gather_face_48pairs_w1(uint32_t dst_scratch_addr, const uint32_t* p_hoff_48, const char* vnode_bytes_ptr) {
    gather_face_24pairs(dst_scratch_addr + 512,  p_hoff_48 + 0,  vnode_bytes_ptr); // Face 1
    gather_face_24pairs(dst_scratch_addr + 1536, p_hoff_48 + 24, vnode_bytes_ptr); // Face 3
}

inline void scatter_face_48pairs_w0(uint32_t src_scratch_addr, const uint32_t* p_hoff_48, char* vnode_out_ptr) {
    scatter_face_24pairs_fast(src_scratch_addr + 0,    p_hoff_48 + 0,  vnode_out_ptr); // Face 0
    scatter_face_24pairs_fast(src_scratch_addr + 1024, p_hoff_48 + 24, vnode_out_ptr); // Face 2
}

inline void scatter_face_48pairs_w1(uint32_t src_scratch_addr, const uint32_t* p_hoff_48, char* vnode_out_ptr) {
    scatter_face_24pairs_fast(src_scratch_addr + 512,  p_hoff_48 + 0,  vnode_out_ptr); // Face 1
    scatter_face_24pairs_fast(src_scratch_addr + 1536, p_hoff_48 + 24, vnode_out_ptr); // Face 3
}


