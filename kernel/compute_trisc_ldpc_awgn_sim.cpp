#include "api/compute/eltwise_binary.h"
#include "api/compute/cb_api.h"
#include <cstring>

// Fast Xorshift32 Uniform PRNG
inline uint32_t xorshift32(uint32_t& state) {
    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return state = x;
}

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

// Fast natural logarithm ln(s) using IEEE-754 bit extraction and a 6th-degree minimax polynomial.
// Max absolute error < 6.1e-6 across the full range (0.0001, 1.0).
// Avoids software emulation of <cmath> std::log on RV32 TRISC.
inline float fast_ln(float s) {
    uint32_t u = float_as_uint(s);
    int32_t exp = static_cast<int32_t>((u >> 23) & 0xFF) - 127;
    u = (u & 0x007FFFFF) | 0x3F800000; // Normalized float in [1.0f, 2.0f)
    float m = uint_as_float(u) - 1.0f; // Mantissa fraction in [0.0f, 1.0f)

    // Minimax polynomial for ln(1 + m) on [0, 1]
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
// All operations are purely local to this Tensix core's pipeline with zero cross-core communication.
inline void generate_awgn_llr_pair(
    uint32_t& prng_state, 
    float mu_llr, 
    float sigma_llr, 
    float& llr0, 
    float& llr1
) {
    // Exact reciprocal multiplier for uint32 -> uniform float in [-1.0f, +1.0f) without float division
    constexpr float INT32_TO_UNIT = 4.656612875245797e-10f; // 1.0f / 2147483648.0f
    float v1, v2, s;
    do {
        int32_t u1 = static_cast<int32_t>(xorshift32(prng_state));
        int32_t u2 = static_cast<int32_t>(xorshift32(prng_state));
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

void kernel_main() {
    // Runtime Arguments from Host
    uint32_t num_codewords = get_arg_val<uint32_t>(0);
    uint32_t N_var_nodes   = get_arg_val<uint32_t>(1);
    uint32_t M_check_nodes = get_arg_val<uint32_t>(2);
    uint32_t P_punctured   = get_arg_val<uint32_t>(3);
    uint32_t max_check_deg = get_arg_val<uint32_t>(4);
    uint32_t mu_bits       = get_arg_val<uint32_t>(5);
    uint32_t sigma_bits    = get_arg_val<uint32_t>(6);
    uint32_t prng_state    = get_arg_val<uint32_t>(7);
    uint32_t num_h_tiles   = get_arg_val<uint32_t>(8);
    uint32_t max_iter      = get_arg_val<uint32_t>(9);
    if (max_iter == 0) max_iter = 16;

    float mu_llr = 0.0f;
    float sigma_llr = 0.0f;
    std::memcpy(&mu_llr, &mu_bits, sizeof(float));
    std::memcpy(&sigma_llr, &sigma_bits, sizeof(float));

    constexpr uint32_t MAX_DEG  = 12;
    constexpr float alpha       = 0.75f;

    // Host provides unique deterministic seed per core and per batch.
    // rdcycle (CSR 0xc00) is bypassed here to support ttsim instruction-set simulation.
    // uint32_t cycle;
    // asm volatile("rdcycle %0" : "=r"(cycle));
    // prng_state ^= cycle;

    // Wait for matrix topology in Circular Buffer c_2
    constexpr uint32_t cb_h = tt::CBIndex::c_2;
    cb_wait_front(cb_h, 1);
    // Non-volatile pointer allows the compiler to cache indices and hoist memory operations
    const auto* h_col_idx = reinterpret_cast<const uint16_t*>(get_tile_address(cb_h, 0));

    // Map arrays to L1 Circular Buffers (zero TRISC stack overhead)
    float* channel_llrs = reinterpret_cast<float*>(get_tile_address(tt::CBIndex::c_0, 0));
    auto* r_msg = reinterpret_cast<float(*)[MAX_DEG]>(get_tile_address(tt::CBIndex::c_1, 0));

    uint32_t unpunctured_nodes = N_var_nodes - P_punctured;
    uint32_t total_bit_errors   = 0;
    uint32_t total_frame_errors = 0;
    uint32_t dbg_min1 = 0;
    uint32_t dbg_min2 = 0;
    uint32_t dbg_min_idx = 0;
    uint32_t dbg_global_sign = 0;
    uint32_t dbg_q3 = 0;
    uint32_t dbg_new_llr2347 = 0;
    uint32_t cw0_iters = 0;
    uint32_t cw1_iters = 0;

    constexpr uint32_t cb_stats = tt::CBIndex::c_16;
    auto* stats_out = reinterpret_cast<uint32_t*>(get_tile_address(cb_stats, 0));

    UNPACK({
        for (uint32_t cw = 0; cw < num_codewords; cw++) {

            // STEP 1: AWGN Noise Generation (Trigonometry-free Polar Box-Muller)
            if (sigma_llr > 0.0f) {
                for (uint32_t i = 0; i < unpunctured_nodes; i += 2) {
                    generate_awgn_llr_pair(
                        prng_state, 
                        mu_llr, 
                        sigma_llr, 
                        channel_llrs[i], 
                        channel_llrs[i + 1]
                    );
                }
            } else {
                for (uint32_t i = 0; i < unpunctured_nodes; i++) {
                    channel_llrs[i] = mu_llr;
                }
            }

            // STEP 2: Puncturing (Neutral LLR = 0)
            for (uint32_t i = unpunctured_nodes; i < N_var_nodes; i++) {
                channel_llrs[i] = 0.0f;
            }

            // Reset check-node memory: only clear active check degree entries to avoid 384 KB memset
            for (uint32_t m = 0; m < M_check_nodes; m++) {
                for (uint32_t d = 0; d < max_check_deg; d++) {
                    r_msg[m][d] = 0.0f;
                }
            }

            // STEP 3: Layered Normalized Min-Sum Decoder
            for (uint32_t iter = 0; iter < max_iter; iter++) {
                for (uint32_t m = 0; m < M_check_nodes; m++) {
                    float min1 = 999.0f, min2 = 999.0f;
                    uint32_t min_idx = 0, global_sign = 0;
                    float q_val[MAX_DEG];
                    uint32_t actual_deg = 0;

                    for (uint32_t d = 0; d < max_check_deg; d++) {
                        uint16_t vn = h_col_idx[m * max_check_deg + d];
                        if (vn == 0xFFFF) break; // Sentinel check for irregular graphs
                        actual_deg++;

                        q_val[d] = channel_llrs[vn] - r_msg[m][d];

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

                    for (uint32_t d = 0; d < actual_deg; d++) {
                        uint16_t vn = h_col_idx[m * max_check_deg + d];
                        float current_min = (d == min_idx) ? min2 : min1;
                        uint32_t node_sign = float_as_uint(q_val[d]) >> 31;
                        uint32_t msg_sign = global_sign ^ node_sign;

                        float r_mag = alpha * current_min;
                        uint32_t r_u = float_as_uint(r_mag) | (msg_sign << 31);
                        float r_new = uint_as_float(r_u);

                        channel_llrs[vn] = q_val[d] + r_new;
                        r_msg[m][d] = r_new;
                    }
                }

                // STEP 4: Inline Syndrome Check (H * c^T == 0)
                uint32_t syndrome_errors = 0;
                for (uint32_t m = 0; m < M_check_nodes; m++) {
                    uint32_t row_parity = 0;
                    for (uint32_t d = 0; d < max_check_deg; d++) {
                        uint16_t vn = h_col_idx[m * max_check_deg + d];
                        if (vn == 0xFFFF) break;
                        row_parity ^= (float_as_uint(channel_llrs[vn]) >> 31);
                    }
                    syndrome_errors |= row_parity;
                    if (syndrome_errors != 0) break;
                }

                if (cw == 0 && iter == 0) {
                    dbg_q3 = float_as_uint(channel_llrs[803]);
                    dbg_new_llr2347 = float_as_uint(channel_llrs[2347]);
                }

                if (syndrome_errors == 0) {
                    if (cw == 0) cw0_iters = iter + 1;
                    else if (cw == 1) cw1_iters = iter + 1;
                    break; // Early termination on valid codeword
                }
            }
            if (cw == 0 && cw0_iters == 0) cw0_iters = max_iter;
            if (cw == 1 && cw1_iters == 0) cw1_iters = max_iter;

            // STEP 5: Tally Bit and Frame Errors
            uint32_t cw_bit_errors = 0;
            for (uint32_t i = 0; i < unpunctured_nodes; i++) {
                cw_bit_errors += (float_as_uint(channel_llrs[i]) >> 31);
            }

            if (cw_bit_errors > 0) {
                total_bit_errors += cw_bit_errors;
                total_frame_errors++;
            }
        }

        // Release matrix buffer on UNPACK
        cb_pop_front(cb_h, 1);

        stats_out[0] = total_bit_errors;
        stats_out[1] = total_frame_errors;
        stats_out[2] = cw0_iters;
        stats_out[3] = float_as_uint(channel_llrs[0]);

        mailbox_write(ckernel::ThreadId::MathThreadId, 1);
        mailbox_write(ckernel::ThreadId::PackThreadId, 1);
    })

    MATH({
        mailbox_read(ckernel::ThreadId::UnpackThreadId);
    })

    PACK({
        mailbox_read(ckernel::ThreadId::UnpackThreadId);
        cb_reserve_back(cb_stats, 1);
        cb_push_back(cb_stats, 1);
    })
}