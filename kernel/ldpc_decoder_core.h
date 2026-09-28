#pragma once

#include <stdint.h>
#include <cstring>

// Memory layout constants for 5 concurrent RISC-V workers per Tensix core
constexpr uint32_t NUM_WORKERS              = 5;
constexpr uint32_t LLR_WORKER_STRIDE_BYTES  = 6144;   // 6 KB per worker (2560 * 2 = 5120 bytes) -> 5 * 6KB = 30 KB <= 32 KB CB0
constexpr uint32_t R_WORKER_STRIDE_BYTES    = 25600;  // 25 KB per worker (1536 * 8 * 2 = 24576 bytes) -> 5 * 25KB = 125 KB <= 128 KB CB1
constexpr uint32_t MAX_DEG_STRIDE           = 8;      // Support graphs up to check degree 8
constexpr uint32_t WORKER_STATS_OFFSET      = 64;     // Offset in CB16 for 5 workers * 4 words = 80 bytes

// Fast SplitMix32 for state initialization from 32-bit seeds
inline uint32_t splitmix32(uint32_t& x) {
    uint32_t z = (x += 0x9E3779B9);
    z ^= z >> 16;
    z *= 0x85EBCA6B;
    z ^= z >> 13;
    z *= 0xC2B2AE35;
    z ^= z >> 16;
    return z;
}

// Xoshiro128+ Uniform PRNG (period: 2^128 - 1 ≈ 3.4e38)
// Pure 32-bit native RV32 arithmetic (shifts, adds, rotates) with zero 64-bit software emulation
struct Xoshiro128Plus {
    uint32_t s[4];

    inline void init_state(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3) {
        s[0] = s0;
        s[1] = s1;
        s[2] = s2;
        s[3] = s3;
        if ((s[0] | s[1] | s[2] | s[3]) == 0) {
            s[0] = 1;
        }
    }

    inline uint32_t next() {
        const uint32_t result = s[0] + s[3];
        const uint32_t t = s[1] << 9;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = (s[3] << 11) | (s[3] >> 21);
        return result;
    }

    // Equivalent to 2^64 calls to next()
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
};

inline uint32_t float_as_uint(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

inline float uint_as_float(uint32_t u) {
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

inline float bf16_to_fp32(uint16_t bf) {
    uint32_t u = static_cast<uint32_t>(bf) << 16;
    return uint_as_float(u);
}

inline uint16_t fp32_to_bf16(float f) {
    uint32_t u = float_as_uint(f);
    uint32_t lsb = (u >> 16) & 1u;
    uint32_t bias = 0x7FFFu + lsb;
    u += bias;
    return static_cast<uint16_t>(u >> 16);
}

// Fast natural logarithm ln(s) using IEEE-754 bit extraction and a 6th-degree minimax polynomial.
// Max absolute error < 6.1e-6 across the full range (0.0001, 1.0).
inline float fast_ln(float s) {
    uint32_t u = float_as_uint(s);
    int32_t exp = static_cast<int32_t>((u >> 23) & 0xFF) - 127;
    u = (u & 0x007FFFFF) | 0x3F800000; // Normalized float in [1.0f, 2.0f)
    float m = uint_as_float(u) - 1.0f; // Mantissa fraction in [0.0f, 1.0f)

    float p = -0.02397957f;
    p = p * m + 0.10150005f;
    p = p * m - 0.21029369f;
    p = p * m + 0.32529514f;
    p = p * m - 0.49937260f;
    p = p * m + 0.99999183f;
    p = p * m;

    return static_cast<float>(exp) * 0.69314718056f + p;
}

// Fast reciprocal square root 1/sqrt(x) using bit-level initial approximation + 2 Newton-Raphson iterations.
inline float fast_rsqrt(float x) {
    uint32_t u = float_as_uint(x);
    u = 0x5F3759DF - (u >> 1);
    float y = uint_as_float(u);
    float xhalf = 0.5f * x;
    y = y * (1.5f - xhalf * y * y);
    y = y * (1.5f - xhalf * y * y);
    return y;
}

// Fast square root sqrt(x) = x * rsqrt(x)
inline float fast_sqrt(float x) {
    return x * fast_rsqrt(x);
}

// Marsaglia Polar AWGN Generator: completely eliminates trigonometric functions (sin/cos).
inline void generate_awgn_llr_pair(
    Xoshiro128Plus& rng, 
    float mu_llr, 
    float sigma_llr, 
    float& llr0, 
    float& llr1
) {
    constexpr float INT32_TO_UNIT = 4.656612875245797e-10f; // 1.0f / 2147483648.0f
    float v1, v2, s;
    do {
        int32_t u1 = static_cast<int32_t>(rng.next());
        int32_t u2 = static_cast<int32_t>(rng.next());
        v1 = static_cast<float>(u1) * INT32_TO_UNIT;
        v2 = static_cast<float>(u2) * INT32_TO_UNIT;
        s = v1 * v1 + v2 * v2;
    } while (s >= 1.0f || s <= 1e-15f);

    float rsqrt_s = fast_rsqrt(s);
    float inv_s = rsqrt_s * rsqrt_s;
    float neg2_ln = -2.0f * fast_ln(s);
    float factor = fast_sqrt(neg2_ln * inv_s);

    llr0 = mu_llr + sigma_llr * (v1 * factor);
    llr1 = mu_llr + sigma_llr * (v2 * factor);
}

// Unified LDPC decoding worker executed on any of the 5 RISC-V processors:
// Worker 0: BRISC (DataMovement 0)
// Worker 1: NCRISC (DataMovement 1)
// Worker 2: TRISC0 (Compute UNPACK)
// Worker 3: TRISC1 (Compute MATH)
// Worker 4: TRISC2 (Compute PACK)
inline void run_ldpc_worker(
    uint32_t worker_id,
    uint32_t num_codewords,
    uint32_t N_var_nodes,
    uint32_t M_check_nodes,
    uint32_t max_check_deg,
    uint32_t P_punctured,
    uint32_t max_iter,
    float mu_llr,
    float sigma_llr,
    float l_max,
    float r_max,
    uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3,
    const uint16_t* h_col_idx,
    uint16_t* channel_llrs,
    uint16_t* r_msg_base,
    uint32_t* stats_out
) {
    if (num_codewords == 0) {
        stats_out[0] = 0;
        stats_out[1] = 0;
        stats_out[2] = 0;
        stats_out[3] = 0;
        return;
    }

    // Initialize PRNG state and advance by worker_id * 2^64 draws via jump()
    Xoshiro128Plus rng;
    rng.init_state(s0, s1, s2, s3);
    for (uint32_t j = 0; j < worker_id; j++) {
        rng.jump();
    }

    constexpr uint32_t MAX_LOCAL_DEG = 12;
    constexpr float alpha            = 0.75f;
    auto* r_msg = reinterpret_cast<uint16_t(*)[MAX_DEG_STRIDE]>(r_msg_base);

    uint32_t unpunctured_nodes  = N_var_nodes - P_punctured;
    uint32_t total_bit_errors   = 0;
    uint32_t total_frame_errors = 0;
    uint32_t cw0_iters          = 0;
    uint32_t sample_llr         = 0;

    for (uint32_t cw = 0; cw < num_codewords; cw++) {

        // STEP 1: AWGN Noise Generation (Trigonometry-free Polar Box-Muller)
        if (sigma_llr > 0.0f) {
            for (uint32_t i = 0; i < unpunctured_nodes; i += 2) {
                float l0 = 0.0f, l1 = 0.0f;
                generate_awgn_llr_pair(rng, mu_llr, sigma_llr, l0, l1);
                if (l_max > 0.0f) {
                    if (l0 > l_max) l0 = l_max;
                    else if (l0 < -l_max) l0 = -l_max;
                    if (l1 > l_max) l1 = l_max;
                    else if (l1 < -l_max) l1 = -l_max;
                }
                channel_llrs[i]     = fp32_to_bf16(l0);
                channel_llrs[i + 1] = fp32_to_bf16(l1);
            }
        } else {
            float mu_val = mu_llr;
            if (l_max > 0.0f && mu_val > l_max) mu_val = l_max;
            uint16_t mu_bf = fp32_to_bf16(mu_val);
            for (uint32_t i = 0; i < unpunctured_nodes; i++) {
                channel_llrs[i] = mu_bf;
            }
        }

        // STEP 2: Puncturing (Neutral LLR = 0)
        for (uint32_t i = unpunctured_nodes; i < N_var_nodes; i++) {
            channel_llrs[i] = 0;
        }

        // Reset check-node memory
        for (uint32_t m = 0; m < M_check_nodes; m++) {
            for (uint32_t d = 0; d < max_check_deg; d++) {
                r_msg[m][d] = 0;
            }
        }

        // STEP 3: Row-Layered Decoder (Approximate-Min* by default, or Normalized Min-Sum)
        uint32_t row_stride = max_check_deg + 1;
        for (uint32_t iter = 0; iter < max_iter; iter++) {
            for (uint32_t m = 0; m < M_check_nodes; m++) {
                uint32_t row_base = m * row_stride;
                uint32_t deg = h_col_idx[row_base];
                const uint16_t* vn_list = &h_col_idx[row_base + 1];
                uint16_t vn_cache[MAX_LOCAL_DEG];

#ifdef USE_NORMALIZED_MIN_SUM
                // --- Algorithm: Normalized Min-Sum (alpha = 0.75) ---
                float min1 = 999.0f, min2 = 999.0f;
                uint32_t min_idx = 0, global_sign = 0;
                float q_val[MAX_LOCAL_DEG];

                for (uint32_t d = 0; d < deg; d++) {
                    uint16_t vn = vn_list[d];
                    vn_cache[d] = vn;

                    q_val[d] = bf16_to_fp32(channel_llrs[vn]) - bf16_to_fp32(r_msg[m][d]);

                    uint32_t q_u = float_as_uint(q_val[d]);
                    uint32_t sign = q_u >> 31;
                    global_sign ^= sign;
                    float abs_q = uint_as_float(q_u & 0x7FFFFFFF);

                    if (abs_q < min1) {
                        min2 = min1; min1 = abs_q; min_idx = d;
                    } else if (abs_q < min2) {
                        min2 = abs_q;
                    }
                }

                for (uint32_t d = 0; d < deg; d++) {
                    uint16_t vn = vn_cache[d];
                    float current_min = (d == min_idx) ? min2 : min1;
                    uint32_t node_sign = float_as_uint(q_val[d]) >> 31;
                    uint32_t msg_sign = global_sign ^ node_sign;

                    float r_mag = alpha * current_min;
                    if (r_max > 0.0f && r_mag > r_max) {
                        r_mag = r_max;
                    }
                    uint32_t r_u = float_as_uint(r_mag) | (msg_sign << 31);
                    float r_new = uint_as_float(r_u);

                    float new_llr = q_val[d] + r_new;
                    channel_llrs[vn] = fp32_to_bf16(new_llr);
                    r_msg[m][d] = fp32_to_bf16(r_new);
                }
#else
                // --- Algorithm: Christopher Jones Approximate-Min* (MILCOM 2003) ---
                // Pre-loaded row degree: zero sentinels, zero 0xFFFF branches
                uint32_t min_idx = 0, global_sign = 0;
                float q_val[MAX_LOCAL_DEG];
                float min_mag = 9999.0f;

                for (uint32_t d = 0; d < deg; d++) {
                    uint16_t vn = vn_list[d];
                    vn_cache[d] = vn; // Register cache: zero SRAM reads in writeback

                    q_val[d] = bf16_to_fp32(channel_llrs[vn]) - bf16_to_fp32(r_msg[m][d]);

                    uint32_t q_u = float_as_uint(q_val[d]);
                    uint32_t sign = q_u >> 31;
                    global_sign ^= sign;
                    float abs_q = uint_as_float(q_u & 0x7FFFFFFF);

                    if (abs_q < min_mag) {
                        min_mag = abs_q;
                        min_idx = d;
                    }
                }

                float r_mag_min, r_mag_all;
                if (deg == 3) {
                    // Fully unrolled branch-free fast path for degree-3 check nodes
                    float other0 = (min_idx == 0) ? uint_as_float(float_as_uint(q_val[1]) & 0x7FFFFFFF) : uint_as_float(float_as_uint(q_val[0]) & 0x7FFFFFFF);
                    float other1 = (min_idx == 2) ? uint_as_float(float_as_uint(q_val[1]) & 0x7FFFFFFF) : uint_as_float(float_as_uint(q_val[2]) & 0x7FFFFFFF);
                    float diff = (other0 > other1) ? (other0 - other1) : (other1 - other0);
                    float m_min = (other0 < other1) ? other0 : other1;
                    float corr = (diff >= 2.0f) ? 0.0f : ((diff < 0.5f) ? (0.69315f - 0.40f * diff) : (0.49315f - 0.328f * (diff - 0.5f)));
                    float m_others = (m_min > corr) ? (m_min - corr) : 0.0f;

                    float diff_all = (m_others > min_mag) ? (m_others - min_mag) : (min_mag - m_others);
                    float m_all_min = (m_others < min_mag) ? m_others : min_mag;
                    float corr_all = (diff_all >= 2.0f) ? 0.0f : ((diff_all < 0.5f) ? (0.69315f - 0.40f * diff_all) : (0.49315f - 0.328f * (diff_all - 0.5f)));
                    float m_all = (m_all_min > corr_all) ? (m_all_min - corr_all) : 0.0f;

                    r_mag_min = m_others;
                    r_mag_all = m_all;
                } else {
                    // General path for arbitrary degree graphs
                    uint32_t first_idx = (min_idx == 0) ? 1 : 0;
                    float m_others = (deg > 1) ? uint_as_float(float_as_uint(q_val[first_idx]) & 0x7FFFFFFF) : min_mag;

                    for (uint32_t d = first_idx + 1; d < deg; d++) {
                        if (d == min_idx) continue;
                        float abs_qd = uint_as_float(float_as_uint(q_val[d]) & 0x7FFFFFFF);
                        float diff = (m_others > abs_qd) ? (m_others - abs_qd) : (abs_qd - m_others);
                        float m_curr = (m_others < abs_qd) ? m_others : abs_qd;
                        float corr = (diff >= 2.0f) ? 0.0f : ((diff < 0.5f) ? (0.69315f - 0.40f * diff) : (0.49315f - 0.328f * (diff - 0.5f)));
                        float next_m = m_curr - corr;
                        m_others = (next_m > 0.0f) ? next_m : 0.0f;
                    }

                    float diff_all = (m_others > min_mag) ? (m_others - min_mag) : (min_mag - m_others);
                    float m_all_curr = (m_others < min_mag) ? m_others : min_mag;
                    float corr_all = (diff_all >= 2.0f) ? 0.0f : ((diff_all < 0.5f) ? (0.69315f - 0.40f * diff_all) : (0.49315f - 0.328f * (diff_all - 0.5f)));
                    float next_m_all = m_all_curr - corr_all;
                    float m_all = (next_m_all > 0.0f) ? next_m_all : 0.0f;

                    r_mag_min = m_others;
                    r_mag_all = m_all;
                }

                // Step 4: Writeback with zero L1 memory reads for vn
                for (uint32_t d = 0; d < deg; d++) {
                    uint16_t vn = vn_cache[d];
                    float r_mag = (d == min_idx) ? r_mag_min : r_mag_all;
                    if (r_max > 0.0f && r_mag > r_max) {
                        r_mag = r_max;
                    }
                    uint32_t node_sign = float_as_uint(q_val[d]) >> 31;
                    uint32_t msg_sign = global_sign ^ node_sign;

                    uint32_t r_u = float_as_uint(r_mag) | (msg_sign << 31);
                    float r_new = uint_as_float(r_u);

                    float new_llr = q_val[d] + r_new;
                    channel_llrs[vn] = fp32_to_bf16(new_llr);
                    r_msg[m][d] = fp32_to_bf16(r_new);
                }
#endif
            }

            // STEP 4: Inline Syndrome Check (H * c^T == 0)
            // Adaptive check: bypass iterations 0 and 1
            if (iter >= 2) {
                uint32_t syndrome_errors = 0;
                for (uint32_t m = 0; m < M_check_nodes; m++) {
                    uint32_t r_base = m * row_stride;
                    uint32_t d_count = h_col_idx[r_base];
                    const uint16_t* v_nodes = &h_col_idx[r_base + 1];
                    uint32_t row_parity = 0;
                    for (uint32_t d = 0; d < d_count; d++) {
                        row_parity ^= (channel_llrs[v_nodes[d]] >> 15);
                    }
                    syndrome_errors |= row_parity;
                    if (syndrome_errors != 0) break;
                }

                if (syndrome_errors == 0) {
                    if (cw == 0) cw0_iters = iter + 1;
                    break; // Early termination on valid codeword
                }
            }
        }
        if (cw == 0 && cw0_iters == 0) cw0_iters = max_iter;

        // STEP 5: Tally Bit and Frame Errors
        uint32_t cw_bit_errors = 0;
        for (uint32_t i = 0; i < unpunctured_nodes; i++) {
            cw_bit_errors += (channel_llrs[i] >> 15);
        }

        if (cw_bit_errors > 0) {
            total_bit_errors += cw_bit_errors;
            total_frame_errors++;
        }
    }

    sample_llr = static_cast<uint32_t>(channel_llrs[0]) << 16;

    stats_out[0] = total_bit_errors;
    stats_out[1] = total_frame_errors;
    stats_out[2] = cw0_iters;
    stats_out[3] = sample_llr;
}
