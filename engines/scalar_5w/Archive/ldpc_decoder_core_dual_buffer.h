#pragma once

#include <stdint.h>
#include <cstring>
#include <cmath>
#include "fp8_types.h"

// ==============================================================================
// Dual-Buffer Bfloat8 (FP8 E5M2) LDPC Decoder Core
// ==============================================================================
// Architectural Features:
// 1. Dual-Buffer Variable Node Memory:
//    - channel_llrs_pair: Immutable channel observations L_v^ch in Bfloat8
//    - post_llrs_pair:    Evolving posterior accumulations L_v^post in Bfloat8
// 2. True Channel Observation Clipping:
//    - l_ch_max strictly bounds raw channel noise at generation without
//      truncating evolved belief propagation posteriors during iterations.
// 3. 2-Codeword Packed Interleaving in 16-Bit Words:
//    - [7:0]  = Codeword 0 (fp8_e5m2)
//    - [15:8] = Codeword 1 (fp8_e5m2)
//    - Total memory: 2560 * 2 bytes = 5.0 KB for L_ch + 5.0 KB for L_post = 10.0 KB
//    - Matches exact memory footprint of single-buffer bfloat16 (10.24 KB)!
// ==============================================================================

constexpr uint32_t DUAL_BUF_NUM_WORKERS     = 5;
constexpr uint32_t DUAL_BUF_MAX_DEG_STRIDE  = 8;

// True Recursive Approximate-Min* (MILCOM 2003) for degree-3 check nodes
inline void dual_buf_compute_amin_star_deg3(
    const float* q_val, uint32_t min_idx, float min_mag,
    float& r_mag_min, float& r_mag_all
) {
    float other0 = (min_idx == 0) ? std::abs(q_val[1]) : std::abs(q_val[0]);
    float other1 = (min_idx == 2) ? std::abs(q_val[1]) : std::abs(q_val[2]);
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
}

// True Recursive Approximate-Min* (MILCOM 2003) for arbitrary-degree check nodes
inline void dual_buf_compute_amin_star_general(
    const float* q_val, uint32_t deg, uint32_t min_idx, float min_mag,
    float& r_mag_min, float& r_mag_all
) {
    uint32_t first_idx = (min_idx == 0) ? 1 : 0;
    float m_others = (deg > 1) ? std::abs(q_val[first_idx]) : min_mag;

    for (uint32_t d = first_idx + 1; d < deg; d++) {
        if (d == min_idx) continue;
        float abs_qd = std::abs(q_val[d]);
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

// Float <-> Bfloat16 helper for check messages
inline uint16_t dual_buf_fp32_to_bf16(float f) {
    uint32_t x = float_to_raw_u32(f);
    return static_cast<uint16_t>(x >> 16);
}

inline float dual_buf_bf16_to_fp32(uint16_t u) {
    return raw_u32_to_float(static_cast<uint32_t>(u) << 16);
}

// PRNG support for AWGN generation
struct DualBufXoshiro128Plus {
    uint32_t s[4];

    inline void init_state(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3) {
        s[0] = s0; s[1] = s1; s[2] = s2; s[3] = s3;
        if ((s[0] | s[1] | s[2] | s[3]) == 0) s[0] = 1;
    }

    inline uint32_t next() {
        const uint32_t result = s[0] + s[3];
        const uint32_t t = s[1] << 9;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t; s[3] = (s[3] << 11) | (s[3] >> 21);
        return result;
    }

    void jump() {
        static const uint32_t JUMP[] = { 0x8764000B, 0xF542D2D3, 0x6FA035C3, 0x77F2DB5B };
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

inline void dual_buf_generate_awgn_llr_pair(
    DualBufXoshiro128Plus& rng,
    float mu_llr,
    float sigma_llr,
    float& llr0,
    float& llr1
) {
    constexpr float INT32_TO_UNIT = 1.0f / 2147483648.0f;
    float v1, v2, s;
    do {
        int32_t u1 = static_cast<int32_t>(rng.next());
        int32_t u2 = static_cast<int32_t>(rng.next());
        v1 = static_cast<float>(u1) * INT32_TO_UNIT;
        v2 = static_cast<float>(u2) * INT32_TO_UNIT;
        s = v1 * v1 + v2 * v2;
    } while (s >= 1.0f || s == 0.0f);

    float norm = std::sqrt(-2.0f * std::log(s) / s);
    llr0 = mu_llr + sigma_llr * (v1 * norm);
    llr1 = mu_llr + sigma_llr * (v2 * norm);
}

// ------------------------------------------------------------------------------
// Core LDPC Belief Propagation Decoder Engine with Dual-Buffer Bfloat8 LLRs
// ------------------------------------------------------------------------------
inline void decode_ldpc_dual_buffer_fp8(
    uint32_t num_codewords,
    uint32_t worker_id,
    uint32_t N_var_nodes,
    uint32_t M_check_nodes,
    uint32_t max_check_deg,
    uint32_t P_punctured,
    uint32_t max_iter,
    float mu_llr,
    float sigma_llr,
    float l_ch_max,      // TRUE channel clipping limit (bounds raw AWGN)
    float r_max,         // Check message clipping limit
    float l_post_max,    // Posterior accumulation limit
    uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3,
    const uint16_t* h_col_idx,
    uint16_t* channel_llrs_pair, // Buffer 1: Immutable L_v^ch (Bfloat8 pair)
    uint16_t* post_llrs_pair,    // Buffer 2: Evolving L_v^post (Bfloat8 pair)
    uint32_t* r_msg_base,        // Check messages (bfloat16 pairs)
    uint32_t* stats_out          // [bit_errors, frame_errors, cw0_iters, sample_llr]
) {
    if (num_codewords == 0) {
        stats_out[0] = stats_out[1] = stats_out[2] = stats_out[3] = 0;
        return;
    }

    DualBufXoshiro128Plus rng;
    rng.init_state(s0, s1, s2, s3);
    for (uint32_t j = 0; j < worker_id; j++) {
        rng.jump();
    }

    constexpr uint32_t MAX_LOCAL_DEG = 12;
    auto* r_msg = reinterpret_cast<uint32_t(*)[DUAL_BUF_MAX_DEG_STRIDE]>(r_msg_base);

    uint32_t unpunctured_nodes  = N_var_nodes - P_punctured;
    uint32_t total_bit_errors   = 0;
    uint32_t total_frame_errors = 0;
    uint32_t cw0_iters          = 0;
    uint32_t sample_llr         = 0;

    for (uint32_t cw = 0; cw < num_codewords; cw += 2) {
        bool has_cw1 = (cw + 1 < num_codewords);

        // STEP 1: AWGN Channel Generation with TRUE Channel Clipping
        if (sigma_llr > 0.0f) {
            for (uint32_t i = 0; i < unpunctured_nodes; i += 2) {
                float l0_cw0 = 0.0f, l1_cw0 = 0.0f;
                dual_buf_generate_awgn_llr_pair(rng, mu_llr, sigma_llr, l0_cw0, l1_cw0);
                if (l_ch_max > 0.0f) {
                    if (l0_cw0 > l_ch_max) l0_cw0 = l_ch_max; else if (l0_cw0 < -l_ch_max) l0_cw0 = -l_ch_max;
                    if (l1_cw0 > l_ch_max) l1_cw0 = l_ch_max; else if (l1_cw0 < -l_ch_max) l1_cw0 = -l_ch_max;
                }

                float l0_cw1 = 0.0f, l1_cw1 = 0.0f;
                dual_buf_generate_awgn_llr_pair(rng, mu_llr, sigma_llr, l0_cw1, l1_cw1);
                if (l_ch_max > 0.0f) {
                    if (l0_cw1 > l_ch_max) l0_cw1 = l_ch_max; else if (l0_cw1 < -l_ch_max) l0_cw1 = -l_ch_max;
                    if (l1_cw1 > l_ch_max) l1_cw1 = l_ch_max; else if (l1_cw1 < -l_ch_max) l1_cw1 = -l_ch_max;
                }

                uint8_t fp8_l0_cw0 = fp32_to_fp8_e5m2(l0_cw0);
                uint8_t fp8_l1_cw0 = fp32_to_fp8_e5m2(l1_cw0);
                uint8_t fp8_l0_cw1 = fp32_to_fp8_e5m2(l0_cw1);
                uint8_t fp8_l1_cw1 = fp32_to_fp8_e5m2(l1_cw1);

                uint16_t pair0 = static_cast<uint16_t>(fp8_l0_cw0) | (static_cast<uint16_t>(fp8_l0_cw1) << 8);
                uint16_t pair1 = static_cast<uint16_t>(fp8_l1_cw0) | (static_cast<uint16_t>(fp8_l1_cw1) << 8);

                // Store into immutable channel observation buffer
                channel_llrs_pair[i]     = pair0;
                channel_llrs_pair[i + 1] = pair1;

                // Initialize evolving posterior buffer
                post_llrs_pair[i]        = pair0;
                post_llrs_pair[i + 1]    = pair1;
            }
        } else {
            float mu_val = mu_llr;
            if (l_ch_max > 0.0f && mu_val > l_ch_max) mu_val = l_ch_max;
            uint8_t mu_fp8 = fp32_to_fp8_e5m2(mu_val);
            uint16_t mu_pair = static_cast<uint16_t>(mu_fp8) | (static_cast<uint16_t>(mu_fp8) << 8);
            for (uint32_t i = 0; i < unpunctured_nodes; i++) {
                channel_llrs_pair[i] = mu_pair;
                post_llrs_pair[i]    = mu_pair;
            }
        }

        // STEP 2: Puncturing (Neutral LLR = 0)
        for (uint32_t i = unpunctured_nodes; i < N_var_nodes; i++) {
            channel_llrs_pair[i] = 0;
            post_llrs_pair[i]    = 0;
        }

        // Reset check-node memory
        for (uint32_t m = 0; m < M_check_nodes; m++) {
            for (uint32_t d = 0; d < DUAL_BUF_MAX_DEG_STRIDE; d++) {
                r_msg[m][d] = 0;
            }
        }

        bool done0 = false;
        bool done1 = !has_cw1;
        uint32_t row_stride = max_check_deg + 1;

        // STEP 3: Row-Layered Belief Propagation Iterations
        for (uint32_t iter = 0; iter < max_iter; iter++) {
            for (uint32_t m = 0; m < M_check_nodes; m++) {
                uint32_t r_base = m * row_stride;
                uint32_t deg = h_col_idx[r_base];
                const uint16_t* vn_list = &h_col_idx[r_base + 1];

                uint16_t vn_cache[MAX_LOCAL_DEG];
                uint32_t min_idx0 = 0, global_sign0 = 0;
                float q_val0[MAX_LOCAL_DEG];
                float min_mag0 = 9999.0f;

                uint32_t min_idx1 = 0, global_sign1 = 0;
                float q_val1[MAX_LOCAL_DEG];
                float min_mag1 = 9999.0f;

                for (uint32_t d = 0; d < deg; d++) {
                    uint16_t vn = vn_list[d];
                    vn_cache[d] = vn;

                    uint16_t post_pair = post_llrs_pair[vn];
                    uint32_t r_pair = r_msg[m][d];

                    if (!done0) {
                        float post0 = fp8_e5m2_to_fp32(static_cast<uint8_t>(post_pair & 0xFF));
                        float r0    = dual_buf_bf16_to_fp32(static_cast<uint16_t>(r_pair & 0xFFFF));
                        float q0    = post0 - r0;
                        q_val0[d]   = q0;
                        uint32_t q0_u = float_to_raw_u32(q0);
                        global_sign0 ^= (q0_u >> 31);
                        float abs_q0 = std::abs(q0);
                        if (abs_q0 < min_mag0) {
                            min_mag0 = abs_q0;
                            min_idx0 = d;
                        }
                    }

                    if (!done1) {
                        float post1 = fp8_e5m2_to_fp32(static_cast<uint8_t>(post_pair >> 8));
                        float r1    = dual_buf_bf16_to_fp32(static_cast<uint16_t>(r_pair >> 16));
                        float q1    = post1 - r1;
                        q_val1[d]   = q1;
                        uint32_t q1_u = float_to_raw_u32(q1);
                        global_sign1 ^= (q1_u >> 31);
                        float abs_q1 = std::abs(q1);
                        if (abs_q1 < min_mag1) {
                            min_mag1 = abs_q1;
                            min_idx1 = d;
                        }
                    }
                }

                float r_mag_min0 = 0.0f, r_mag_all0 = 0.0f;
                float r_mag_min1 = 0.0f, r_mag_all1 = 0.0f;

                if (!done0) {
                    if (deg == 3) {
                        dual_buf_compute_amin_star_deg3(q_val0, min_idx0, min_mag0, r_mag_min0, r_mag_all0);
                    } else {
                        dual_buf_compute_amin_star_general(q_val0, deg, min_idx0, min_mag0, r_mag_min0, r_mag_all0);
                    }
                    if (r_max > 0.0f) {
                        if (r_mag_min0 > r_max) r_mag_min0 = r_max;
                        if (r_mag_all0 > r_max) r_mag_all0 = r_max;
                    }
                }

                if (!done1) {
                    if (deg == 3) {
                        dual_buf_compute_amin_star_deg3(q_val1, min_idx1, min_mag1, r_mag_min1, r_mag_all1);
                    } else {
                        dual_buf_compute_amin_star_general(q_val1, deg, min_idx1, min_mag1, r_mag_min1, r_mag_all1);
                    }
                    if (r_max > 0.0f) {
                        if (r_mag_min1 > r_max) r_mag_min1 = r_max;
                        if (r_mag_all1 > r_max) r_mag_all1 = r_max;
                    }
                }

                for (uint32_t d = 0; d < deg; d++) {
                    uint16_t vn = vn_cache[d];
                    uint16_t old_post_pair = post_llrs_pair[vn];
                    uint32_t old_r_pair    = r_msg[m][d];

                    uint8_t fp8_post0 = static_cast<uint8_t>(old_post_pair & 0xFF);
                    uint16_t bf_r0   = static_cast<uint16_t>(old_r_pair & 0xFFFF);
                    if (!done0) {
                        float r_mag0 = (d == min_idx0) ? r_mag_min0 : r_mag_all0;
                        uint32_t node_sign0 = (float_to_raw_u32(q_val0[d]) >> 31);
                        uint32_t msg_sign0  = global_sign0 ^ node_sign0;
                        float r_new0 = (msg_sign0) ? -r_mag0 : r_mag0;
                        float new_post0 = q_val0[d] + r_new0;

                        if (l_post_max > 0.0f) {
                            if (new_post0 > l_post_max) new_post0 = l_post_max;
                            else if (new_post0 < -l_post_max) new_post0 = -l_post_max;
                        }
                        fp8_post0 = fp32_to_fp8_e5m2(new_post0);
                        bf_r0     = dual_buf_fp32_to_bf16(r_new0);
                    }

                    uint8_t fp8_post1 = static_cast<uint8_t>(old_post_pair >> 8);
                    uint16_t bf_r1   = static_cast<uint16_t>(old_r_pair >> 16);
                    if (!done1) {
                        float r_mag1 = (d == min_idx1) ? r_mag_min1 : r_mag_all1;
                        uint32_t node_sign1 = (float_to_raw_u32(q_val1[d]) >> 31);
                        uint32_t msg_sign1  = global_sign1 ^ node_sign1;
                        float r_new1 = (msg_sign1) ? -r_mag1 : r_mag1;
                        float new_post1 = q_val1[d] + r_new1;

                        if (l_post_max > 0.0f) {
                            if (new_post1 > l_post_max) new_post1 = l_post_max;
                            else if (new_post1 < -l_post_max) new_post1 = -l_post_max;
                        }
                        fp8_post1 = fp32_to_fp8_e5m2(new_post1);
                        bf_r1     = dual_buf_fp32_to_bf16(r_new1);
                    }

                    // Update evolving posterior buffer (channel_llrs_pair is completely untouched!)
                    post_llrs_pair[vn] = static_cast<uint16_t>(fp8_post0) | (static_cast<uint16_t>(fp8_post1) << 8);
                    r_msg[m][d]        = static_cast<uint32_t>(bf_r0) | (static_cast<uint32_t>(bf_r1) << 16);
                }
            }

            // STEP 4: Inline Syndrome Check (H * c^T == 0)
            if (iter >= 2) {
                uint32_t syndrome_errors_packed = 0;
                for (uint32_t m = 0; m < M_check_nodes; m++) {
                    uint32_t r_base = m * row_stride;
                    uint32_t d_count = h_col_idx[r_base];
                    const uint16_t* v_nodes = &h_col_idx[r_base + 1];
                    uint16_t row_parity_packed = 0;
                    for (uint32_t d = 0; d < d_count; d++) {
                        row_parity_packed ^= post_llrs_pair[v_nodes[d]];
                    }
                    syndrome_errors_packed |= (row_parity_packed & 0x8080);
                    if ((done0 || (syndrome_errors_packed & 0x0080)) &&
                        (done1 || (syndrome_errors_packed & 0x8000))) {
                        break;
                    }
                }

                if (!done0 && (syndrome_errors_packed & 0x0080) == 0) {
                    done0 = true;
                    if (cw == 0 && cw0_iters == 0) cw0_iters = iter + 1;
                }
                if (!done1 && (syndrome_errors_packed & 0x8000) == 0) {
                    done1 = true;
                }
                if (done0 && done1) {
                    break;
                }
            }
        }

        if (cw == 0 && cw0_iters == 0) cw0_iters = max_iter;

        // STEP 5: Tally Bit and Frame Errors from Hard Decisions
        uint32_t cw_bit_errors0 = 0;
        for (uint32_t i = 0; i < unpunctured_nodes; i++) {
            cw_bit_errors0 += (post_llrs_pair[i] >> 7) & 1;
        }
        if (cw_bit_errors0 > 0) {
            total_bit_errors += cw_bit_errors0;
            total_frame_errors++;
        }

        if (has_cw1) {
            uint32_t cw_bit_errors1 = 0;
            for (uint32_t i = 0; i < unpunctured_nodes; i++) {
                cw_bit_errors1 += (post_llrs_pair[i] >> 15) & 1;
            }
            if (cw_bit_errors1 > 0) {
                total_bit_errors += cw_bit_errors1;
                total_frame_errors++;
            }
        }
    }

    sample_llr = (post_llrs_pair[0] & 0xFF) << 24;

    stats_out[0] = total_bit_errors;
    stats_out[1] = total_frame_errors;
    stats_out[2] = cw0_iters;
    stats_out[3] = sample_llr;
}

// ------------------------------------------------------------------------------
// Hybrid Dual-Buffer: Immutable Channel LLR in Bfloat8 + Evolving Posterior in Bfloat16
// Combines true channel clipping and compact channel storage with zero posterior quantization loss
// ------------------------------------------------------------------------------
inline void decode_ldpc_dual_buffer_hybrid(
    uint32_t num_codewords,
    uint32_t worker_id,
    uint32_t N_var_nodes,
    uint32_t M_check_nodes,
    uint32_t max_check_deg,
    uint32_t P_punctured,
    uint32_t max_iter,
    float mu_llr,
    float sigma_llr,
    float l_ch_max,      // TRUE channel clipping limit (bounds raw AWGN)
    float r_max,         // Check message clipping limit
    float l_post_max,    // Posterior accumulation limit
    uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3,
    const uint16_t* h_col_idx,
    uint16_t* channel_llrs_pair, // Buffer 1: Immutable L_v^ch in Bfloat8
    uint32_t* post_llrs_pair,    // Buffer 2: Evolving L_v^post in Bfloat16
    uint32_t* r_msg_base,        // Check messages (bfloat16 pairs)
    uint32_t* stats_out          // [bit_errors, frame_errors, cw0_iters, sample_llr]
) {
    if (num_codewords == 0) {
        stats_out[0] = stats_out[1] = stats_out[2] = stats_out[3] = 0;
        return;
    }

    DualBufXoshiro128Plus rng;
    rng.init_state(s0, s1, s2, s3);
    for (uint32_t j = 0; j < worker_id; j++) {
        rng.jump();
    }

    constexpr uint32_t MAX_LOCAL_DEG = 12;
    auto* r_msg = reinterpret_cast<uint32_t(*)[DUAL_BUF_MAX_DEG_STRIDE]>(r_msg_base);

    uint32_t unpunctured_nodes  = N_var_nodes - P_punctured;
    uint32_t total_bit_errors   = 0;
    uint32_t total_frame_errors = 0;
    uint32_t cw0_iters          = 0;
    uint32_t sample_llr         = 0;

    for (uint32_t cw = 0; cw < num_codewords; cw += 2) {
        bool has_cw1 = (cw + 1 < num_codewords);

        // STEP 1: AWGN Channel Generation with TRUE Channel Clipping & Bfloat8 Quantization
        if (sigma_llr > 0.0f) {
            for (uint32_t i = 0; i < unpunctured_nodes; i += 2) {
                float l0_cw0 = 0.0f, l1_cw0 = 0.0f;
                dual_buf_generate_awgn_llr_pair(rng, mu_llr, sigma_llr, l0_cw0, l1_cw0);
                if (l_ch_max > 0.0f) {
                    if (l0_cw0 > l_ch_max) l0_cw0 = l_ch_max; else if (l0_cw0 < -l_ch_max) l0_cw0 = -l_ch_max;
                    if (l1_cw0 > l_ch_max) l1_cw0 = l_ch_max; else if (l1_cw0 < -l_ch_max) l1_cw0 = -l_ch_max;
                }

                float l0_cw1 = 0.0f, l1_cw1 = 0.0f;
                dual_buf_generate_awgn_llr_pair(rng, mu_llr, sigma_llr, l0_cw1, l1_cw1);
                if (l_ch_max > 0.0f) {
                    if (l0_cw1 > l_ch_max) l0_cw1 = l_ch_max; else if (l0_cw1 < -l_ch_max) l0_cw1 = -l_ch_max;
                    if (l1_cw1 > l_ch_max) l1_cw1 = l_ch_max; else if (l1_cw1 < -l_ch_max) l1_cw1 = -l_ch_max;
                }

                // Quantize to Bfloat8 (FP8 E5M2)
                uint8_t fp8_l0_cw0 = fp32_to_fp8_e5m2(l0_cw0);
                uint8_t fp8_l1_cw0 = fp32_to_fp8_e5m2(l1_cw0);
                uint8_t fp8_l0_cw1 = fp32_to_fp8_e5m2(l0_cw1);
                uint8_t fp8_l1_cw1 = fp32_to_fp8_e5m2(l1_cw1);

                channel_llrs_pair[i]     = static_cast<uint16_t>(fp8_l0_cw0) | (static_cast<uint16_t>(fp8_l0_cw1) << 8);
                channel_llrs_pair[i + 1] = static_cast<uint16_t>(fp8_l1_cw0) | (static_cast<uint16_t>(fp8_l1_cw1) << 8);

                // Initial posterior in Bfloat16 expanded from Bfloat8 channel observations
                uint16_t bf16_l0_cw0 = dual_buf_fp32_to_bf16(fp8_e5m2_to_fp32(fp8_l0_cw0));
                uint16_t bf16_l0_cw1 = dual_buf_fp32_to_bf16(fp8_e5m2_to_fp32(fp8_l0_cw1));
                uint16_t bf16_l1_cw0 = dual_buf_fp32_to_bf16(fp8_e5m2_to_fp32(fp8_l1_cw0));
                uint16_t bf16_l1_cw1 = dual_buf_fp32_to_bf16(fp8_e5m2_to_fp32(fp8_l1_cw1));

                post_llrs_pair[i]     = static_cast<uint32_t>(bf16_l0_cw0) | (static_cast<uint32_t>(bf16_l0_cw1) << 16);
                post_llrs_pair[i + 1] = static_cast<uint32_t>(bf16_l1_cw0) | (static_cast<uint32_t>(bf16_l1_cw1) << 16);
            }
        } else {
            float mu_val = mu_llr;
            if (l_ch_max > 0.0f && mu_val > l_ch_max) mu_val = l_ch_max;
            uint8_t mu_fp8 = fp32_to_fp8_e5m2(mu_val);
            uint16_t mu_pair_8 = static_cast<uint16_t>(mu_fp8) | (static_cast<uint16_t>(mu_fp8) << 8);

            uint16_t mu_bf = dual_buf_fp32_to_bf16(fp8_e5m2_to_fp32(mu_fp8));
            uint32_t mu_pair_16 = static_cast<uint32_t>(mu_bf) | (static_cast<uint32_t>(mu_bf) << 16);

            for (uint32_t i = 0; i < unpunctured_nodes; i++) {
                channel_llrs_pair[i] = mu_pair_8;
                post_llrs_pair[i]    = mu_pair_16;
            }
        }

        // STEP 2: Puncturing
        for (uint32_t i = unpunctured_nodes; i < N_var_nodes; i++) {
            channel_llrs_pair[i] = 0;
            post_llrs_pair[i]    = 0;
        }

        // Reset check-node memory
        for (uint32_t m = 0; m < M_check_nodes; m++) {
            for (uint32_t d = 0; d < DUAL_BUF_MAX_DEG_STRIDE; d++) {
                r_msg[m][d] = 0;
            }
        }

        bool done0 = false;
        bool done1 = !has_cw1;
        uint32_t row_stride = max_check_deg + 1;

        // STEP 3: Row-Layered Belief Propagation (Evolving in Bfloat16)
        for (uint32_t iter = 0; iter < max_iter; iter++) {
            if (done0 && done1) break;

            for (uint32_t m = 0; m < M_check_nodes; m++) {
                uint32_t r_base = m * row_stride;
                uint32_t deg = h_col_idx[r_base];
                const uint16_t* vn_list = &h_col_idx[r_base + 1];

                uint16_t vn_cache[MAX_LOCAL_DEG];
                uint32_t min_idx0 = 0, global_sign0 = 0;
                float q_val0[MAX_LOCAL_DEG];
                float min_mag0 = 9999.0f;

                uint32_t min_idx1 = 0, global_sign1 = 0;
                float q_val1[MAX_LOCAL_DEG];
                float min_mag1 = 9999.0f;

                for (uint32_t d = 0; d < deg; d++) {
                    uint16_t vn = vn_list[d];
                    vn_cache[d] = vn;

                    uint32_t post_pair = post_llrs_pair[vn];
                    uint32_t r_pair = r_msg[m][d];

                    if (!done0) {
                        float post0 = dual_buf_bf16_to_fp32(static_cast<uint16_t>(post_pair & 0xFFFF));
                        float r0    = dual_buf_bf16_to_fp32(static_cast<uint16_t>(r_pair & 0xFFFF));
                        float q0    = post0 - r0;
                        q_val0[d]   = q0;
                        uint32_t q0_u = float_to_raw_u32(q0);
                        global_sign0 ^= (q0_u >> 31);
                        float abs_q0 = std::abs(q0);
                        if (abs_q0 < min_mag0) {
                            min_mag0 = abs_q0;
                            min_idx0 = d;
                        }
                    }

                    if (!done1) {
                        float post1 = dual_buf_bf16_to_fp32(static_cast<uint16_t>(post_pair >> 16));
                        float r1    = dual_buf_bf16_to_fp32(static_cast<uint16_t>(r_pair >> 16));
                        float q1    = post1 - r1;
                        q_val1[d]   = q1;
                        uint32_t q1_u = float_to_raw_u32(q1);
                        global_sign1 ^= (q1_u >> 31);
                        float abs_q1 = std::abs(q1);
                        if (abs_q1 < min_mag1) {
                            min_mag1 = abs_q1;
                            min_idx1 = d;
                        }
                    }
                }

                float r_mag_min0 = 0.0f, r_mag_all0 = 0.0f;
                float r_mag_min1 = 0.0f, r_mag_all1 = 0.0f;

                if (!done0) {
                    if (deg == 3) {
                        dual_buf_compute_amin_star_deg3(q_val0, min_idx0, min_mag0, r_mag_min0, r_mag_all0);
                    } else {
                        dual_buf_compute_amin_star_general(q_val0, deg, min_idx0, min_mag0, r_mag_min0, r_mag_all0);
                    }
                    if (r_max > 0.0f) {
                        if (r_mag_min0 > r_max) r_mag_min0 = r_max;
                        if (r_mag_all0 > r_max) r_mag_all0 = r_max;
                    }
                }

                if (!done1) {
                    if (deg == 3) {
                        dual_buf_compute_amin_star_deg3(q_val1, min_idx1, min_mag1, r_mag_min1, r_mag_all1);
                    } else {
                        dual_buf_compute_amin_star_general(q_val1, deg, min_idx1, min_mag1, r_mag_min1, r_mag_all1);
                    }
                    if (r_max > 0.0f) {
                        if (r_mag_min1 > r_max) r_mag_min1 = r_max;
                        if (r_mag_all1 > r_max) r_mag_all1 = r_max;
                    }
                }

                for (uint32_t d = 0; d < deg; d++) {
                    uint16_t vn = vn_cache[d];
                    uint32_t old_post_pair = post_llrs_pair[vn];
                    uint32_t old_r_pair    = r_msg[m][d];

                    uint16_t bf_post0 = static_cast<uint16_t>(old_post_pair & 0xFFFF);
                    uint16_t bf_r0    = static_cast<uint16_t>(old_r_pair & 0xFFFF);
                    if (!done0) {
                        float r_mag0 = (d == min_idx0) ? r_mag_min0 : r_mag_all0;
                        uint32_t node_sign0 = (float_to_raw_u32(q_val0[d]) >> 31);
                        uint32_t msg_sign0  = global_sign0 ^ node_sign0;
                        float r_new0 = (msg_sign0) ? -r_mag0 : r_mag0;
                        float new_post0 = q_val0[d] + r_new0;

                        if (l_post_max > 0.0f) {
                            if (new_post0 > l_post_max) new_post0 = l_post_max;
                            else if (new_post0 < -l_post_max) new_post0 = -l_post_max;
                        }
                        bf_post0 = dual_buf_fp32_to_bf16(new_post0);
                        bf_r0    = dual_buf_fp32_to_bf16(r_new0);
                    }

                    uint16_t bf_post1 = static_cast<uint16_t>(old_post_pair >> 16);
                    uint16_t bf_r1    = static_cast<uint16_t>(old_r_pair >> 16);
                    if (!done1) {
                        float r_mag1 = (d == min_idx1) ? r_mag_min1 : r_mag_all1;
                        uint32_t node_sign1 = (float_to_raw_u32(q_val1[d]) >> 31);
                        uint32_t msg_sign1  = global_sign1 ^ node_sign1;
                        float r_new1 = (msg_sign1) ? -r_mag1 : r_mag1;
                        float new_post1 = q_val1[d] + r_new1;

                        if (l_post_max > 0.0f) {
                            if (new_post1 > l_post_max) new_post1 = l_post_max;
                            else if (new_post1 < -l_post_max) new_post1 = -l_post_max;
                        }
                        bf_post1 = dual_buf_fp32_to_bf16(new_post1);
                        bf_r1    = dual_buf_fp32_to_bf16(r_new1);
                    }

                    post_llrs_pair[vn] = static_cast<uint32_t>(bf_post0) | (static_cast<uint32_t>(bf_post1) << 16);
                    r_msg[m][d]        = static_cast<uint32_t>(bf_r0) | (static_cast<uint32_t>(bf_r1) << 16);
                }
            }

            // STEP 4: Inline Syndrome Check (H * c^T == 0)
            if (iter >= 2) {
                uint32_t syndrome_errors_packed = 0;
                for (uint32_t m = 0; m < M_check_nodes; m++) {
                    uint32_t r_base = m * row_stride;
                    uint32_t d_count = h_col_idx[r_base];
                    const uint16_t* v_nodes = &h_col_idx[r_base + 1];
                    uint32_t row_parity_packed = 0;
                    for (uint32_t d = 0; d < d_count; d++) {
                        row_parity_packed ^= post_llrs_pair[v_nodes[d]];
                    }
                    syndrome_errors_packed |= (row_parity_packed & 0x80008000);
                    if ((done0 || (syndrome_errors_packed & 0x8000)) &&
                        (done1 || (syndrome_errors_packed & 0x80000000))) {
                        break;
                    }
                }

                if (!done0 && (syndrome_errors_packed & 0x8000) == 0) {
                    done0 = true;
                    if (cw == 0 && cw0_iters == 0) cw0_iters = iter + 1;
                }
                if (!done1 && (syndrome_errors_packed & 0x80000000) == 0) {
                    done1 = true;
                }
                if (done0 && done1) {
                    break;
                }
            }
        }

        if (cw == 0 && cw0_iters == 0) cw0_iters = max_iter;

        // STEP 5: Tally Bit and Frame Errors
        uint32_t cw_bit_errors0 = 0;
        for (uint32_t i = 0; i < unpunctured_nodes; i++) {
            cw_bit_errors0 += (post_llrs_pair[i] >> 15) & 1;
        }
        if (cw_bit_errors0 > 0) {
            total_bit_errors += cw_bit_errors0;
            total_frame_errors++;
        }

        if (has_cw1) {
            uint32_t cw_bit_errors1 = 0;
            for (uint32_t i = 0; i < unpunctured_nodes; i++) {
                cw_bit_errors1 += (post_llrs_pair[i] >> 31);
            }
            if (cw_bit_errors1 > 0) {
                total_bit_errors += cw_bit_errors1;
                total_frame_errors++;
            }
        }
    }

    sample_llr = (post_llrs_pair[0] & 0xFFFF) << 16;

    stats_out[0] = total_bit_errors;
    stats_out[1] = total_frame_errors;
    stats_out[2] = cw0_iters;
    stats_out[3] = sample_llr;
}

// ------------------------------------------------------------------------------
// Proposed Engine: Channel LLR in Bfloat8, Variable Node APP in Bfloat16, Check Node Edges in Bfloat8
// - Storage: 2 bytes/node (channel), 4 bytes/node (posteriors), 2 bytes/edge (check updates)
// - Computations: Internal arithmetic in FP32 / Bfloat16 with zero accumulation drift
// - Total Memory: 39.0 KiB / worker (195.0 KiB / core across 5 workers = 12.7% L1 SRAM)
// ------------------------------------------------------------------------------
inline void decode_ldpc_dual_buffer_proposal(
    uint32_t num_codewords,
    uint32_t worker_id,
    uint32_t N_var_nodes,
    uint32_t M_check_nodes,
    uint32_t max_check_deg,
    uint32_t P_punctured,
    uint32_t max_iter,
    float mu_llr,
    float sigma_llr,
    float l_ch_max,      // TRUE channel clipping limit (bounds raw AWGN)
    float r_max,         // Check message clipping limit
    float l_post_max,    // Posterior accumulation limit
    uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3,
    const uint16_t* h_col_idx,
    uint16_t* channel_llrs_pair, // Buffer 1: Immutable L_v^ch in Bfloat8 (2 CW packed)
    uint32_t* post_llrs_pair,    // Buffer 2: Evolving L_v^post in Bfloat16 (2 CW packed)
    uint16_t* r_msg_base,        // Check edge updates in Bfloat8 (2 CW packed)
    uint32_t* stats_out          // [bit_errors, frame_errors, cw0_iters, sample_llr]
) {
    if (num_codewords == 0) {
        stats_out[0] = stats_out[1] = stats_out[2] = stats_out[3] = 0;
        return;
    }

    DualBufXoshiro128Plus rng;
    rng.init_state(s0, s1, s2, s3);
    for (uint32_t j = 0; j < worker_id; j++) {
        rng.jump();
    }

    constexpr uint32_t MAX_LOCAL_DEG = 12;
    auto* r_msg = reinterpret_cast<uint16_t(*)[DUAL_BUF_MAX_DEG_STRIDE]>(r_msg_base);

    uint32_t unpunctured_nodes  = N_var_nodes - P_punctured;
    uint32_t total_bit_errors   = 0;
    uint32_t total_frame_errors = 0;
    uint32_t cw0_iters          = 0;
    uint32_t sample_llr         = 0;

    for (uint32_t cw = 0; cw < num_codewords; cw += 2) {
        bool has_cw1 = (cw + 1 < num_codewords);

        // STEP 1: AWGN Channel Generation with TRUE Channel Clipping & Bfloat8 Quantization
        if (sigma_llr > 0.0f) {
            for (uint32_t i = 0; i < unpunctured_nodes; i += 2) {
                float l0_cw0 = 0.0f, l1_cw0 = 0.0f;
                dual_buf_generate_awgn_llr_pair(rng, mu_llr, sigma_llr, l0_cw0, l1_cw0);
                if (l_ch_max > 0.0f) {
                    if (l0_cw0 > l_ch_max) l0_cw0 = l_ch_max; else if (l0_cw0 < -l_ch_max) l0_cw0 = -l_ch_max;
                    if (l1_cw0 > l_ch_max) l1_cw0 = l_ch_max; else if (l1_cw0 < -l_ch_max) l1_cw0 = -l_ch_max;
                }

                float l0_cw1 = 0.0f, l1_cw1 = 0.0f;
                dual_buf_generate_awgn_llr_pair(rng, mu_llr, sigma_llr, l0_cw1, l1_cw1);
                if (l_ch_max > 0.0f) {
                    if (l0_cw1 > l_ch_max) l0_cw1 = l_ch_max; else if (l0_cw1 < -l_ch_max) l0_cw1 = -l_ch_max;
                    if (l1_cw1 > l_ch_max) l1_cw1 = l_ch_max; else if (l1_cw1 < -l_ch_max) l1_cw1 = -l_ch_max;
                }

                // Quantize channel observation to Bfloat8 (FP8 E5M2)
                uint8_t fp8_l0_cw0 = fp32_to_fp8_e5m2(l0_cw0);
                uint8_t fp8_l1_cw0 = fp32_to_fp8_e5m2(l1_cw0);
                uint8_t fp8_l0_cw1 = fp32_to_fp8_e5m2(l0_cw1);
                uint8_t fp8_l1_cw1 = fp32_to_fp8_e5m2(l1_cw1);

                channel_llrs_pair[i]     = static_cast<uint16_t>(fp8_l0_cw0) | (static_cast<uint16_t>(fp8_l0_cw1) << 8);
                channel_llrs_pair[i + 1] = static_cast<uint16_t>(fp8_l1_cw0) | (static_cast<uint16_t>(fp8_l1_cw1) << 8);

                // Initial posterior in Bfloat16 unpacked from Bfloat8 channel observation
                uint16_t bf16_l0_cw0 = dual_buf_fp32_to_bf16(fp8_e5m2_to_fp32(fp8_l0_cw0));
                uint16_t bf16_l0_cw1 = dual_buf_fp32_to_bf16(fp8_e5m2_to_fp32(fp8_l0_cw1));
                uint16_t bf16_l1_cw0 = dual_buf_fp32_to_bf16(fp8_e5m2_to_fp32(fp8_l1_cw0));
                uint16_t bf16_l1_cw1 = dual_buf_fp32_to_bf16(fp8_e5m2_to_fp32(fp8_l1_cw1));

                post_llrs_pair[i]     = static_cast<uint32_t>(bf16_l0_cw0) | (static_cast<uint32_t>(bf16_l0_cw1) << 16);
                post_llrs_pair[i + 1] = static_cast<uint32_t>(bf16_l1_cw0) | (static_cast<uint32_t>(bf16_l1_cw1) << 16);
            }
        } else {
            float mu_val = mu_llr;
            if (l_ch_max > 0.0f && mu_val > l_ch_max) mu_val = l_ch_max;
            uint8_t mu_fp8 = fp32_to_fp8_e5m2(mu_val);
            uint16_t mu_pair_8 = static_cast<uint16_t>(mu_fp8) | (static_cast<uint16_t>(mu_fp8) << 8);

            uint16_t mu_bf = dual_buf_fp32_to_bf16(fp8_e5m2_to_fp32(mu_fp8));
            uint32_t mu_pair_16 = static_cast<uint32_t>(mu_bf) | (static_cast<uint32_t>(mu_bf) << 16);

            for (uint32_t i = 0; i < unpunctured_nodes; i++) {
                channel_llrs_pair[i] = mu_pair_8;
                post_llrs_pair[i]    = mu_pair_16;
            }
        }

        // STEP 2: Puncturing (Neutral LLR = 0)
        for (uint32_t i = unpunctured_nodes; i < N_var_nodes; i++) {
            channel_llrs_pair[i] = 0;
            post_llrs_pair[i]    = 0;
        }

        // Reset check-node memory (Bfloat8: 16-bit zeros)
        for (uint32_t m = 0; m < M_check_nodes; m++) {
            for (uint32_t d = 0; d < DUAL_BUF_MAX_DEG_STRIDE; d++) {
                r_msg[m][d] = 0;
            }
        }

        bool done0 = false;
        bool done1 = !has_cw1;
        uint32_t row_stride = max_check_deg + 1;

        // STEP 3: Row-Layered Belief Propagation Iterations
        for (uint32_t iter = 0; iter < max_iter; iter++) {
            if (done0 && done1) break;

            for (uint32_t m = 0; m < M_check_nodes; m++) {
                uint32_t r_base = m * row_stride;
                uint32_t deg = h_col_idx[r_base];
                const uint16_t* vn_list = &h_col_idx[r_base + 1];

                uint16_t vn_cache[MAX_LOCAL_DEG];
                uint32_t min_idx0 = 0, global_sign0 = 0;
                float q_val0[MAX_LOCAL_DEG];
                float min_mag0 = 9999.0f;

                uint32_t min_idx1 = 0, global_sign1 = 0;
                float q_val1[MAX_LOCAL_DEG];
                float min_mag1 = 9999.0f;

                for (uint32_t d = 0; d < deg; d++) {
                    uint16_t vn = vn_list[d];
                    vn_cache[d] = vn;

                    uint32_t post_pair = post_llrs_pair[vn]; // Bfloat16 pair
                    uint16_t r_pair = r_msg[m][d];          // Bfloat8 pair

                    if (!done0) {
                        float post0 = dual_buf_bf16_to_fp32(static_cast<uint16_t>(post_pair & 0xFFFF));
                        // Unpack check edge from Bfloat8
                        float r0    = fp8_e5m2_to_fp32(static_cast<uint8_t>(r_pair & 0xFF));
                        float q0    = post0 - r0;
                        q_val0[d]   = q0;
                        uint32_t q0_u = float_to_raw_u32(q0);
                        global_sign0 ^= (q0_u >> 31);
                        float abs_q0 = std::abs(q0);
                        if (abs_q0 < min_mag0) {
                            min_mag0 = abs_q0;
                            min_idx0 = d;
                        }
                    }

                    if (!done1) {
                        float post1 = dual_buf_bf16_to_fp32(static_cast<uint16_t>(post_pair >> 16));
                        // Unpack check edge from Bfloat8
                        float r1    = fp8_e5m2_to_fp32(static_cast<uint8_t>(r_pair >> 8));
                        float q1    = post1 - r1;
                        q_val1[d]   = q1;
                        uint32_t q1_u = float_to_raw_u32(q1);
                        global_sign1 ^= (q1_u >> 31);
                        float abs_q1 = std::abs(q1);
                        if (abs_q1 < min_mag1) {
                            min_mag1 = abs_q1;
                            min_idx1 = d;
                        }
                    }
                }

                // Check node Approximate-Min* accumulation
                float r_mag_min0 = 0.0f, r_mag_all0 = 0.0f;
                float r_mag_min1 = 0.0f, r_mag_all1 = 0.0f;

                if (!done0) {
                    if (deg == 3) {
                        dual_buf_compute_amin_star_deg3(q_val0, min_idx0, min_mag0, r_mag_min0, r_mag_all0);
                    } else {
                        dual_buf_compute_amin_star_general(q_val0, deg, min_idx0, min_mag0, r_mag_min0, r_mag_all0);
                    }
                    if (r_max > 0.0f) {
                        if (r_mag_min0 > r_max) r_mag_min0 = r_max;
                        if (r_mag_all0 > r_max) r_mag_all0 = r_max;
                    }
                }

                if (!done1) {
                    if (deg == 3) {
                        dual_buf_compute_amin_star_deg3(q_val1, min_idx1, min_mag1, r_mag_min1, r_mag_all1);
                    } else {
                        dual_buf_compute_amin_star_general(q_val1, deg, min_idx1, min_mag1, r_mag_min1, r_mag_all1);
                    }
                    if (r_max > 0.0f) {
                        if (r_mag_min1 > r_max) r_mag_min1 = r_max;
                        if (r_mag_all1 > r_max) r_mag_all1 = r_max;
                    }
                }

                for (uint32_t d = 0; d < deg; d++) {
                    uint16_t vn = vn_cache[d];
                    uint32_t old_post_pair = post_llrs_pair[vn];
                    uint16_t old_r_pair    = r_msg[m][d];

                    uint16_t bf_post0 = static_cast<uint16_t>(old_post_pair & 0xFFFF);
                    uint8_t  bf8_r0   = static_cast<uint8_t>(old_r_pair & 0xFF);
                    if (!done0) {
                        float r_mag0 = (d == min_idx0) ? r_mag_min0 : r_mag_all0;
                        uint32_t node_sign0 = (float_to_raw_u32(q_val0[d]) >> 31);
                        uint32_t msg_sign0  = global_sign0 ^ node_sign0;
                        float r_new0 = (msg_sign0) ? -r_mag0 : r_mag0;
                        float new_post0 = q_val0[d] + r_new0;

                        if (l_post_max > 0.0f) {
                            if (new_post0 > l_post_max) new_post0 = l_post_max;
                            else if (new_post0 < -l_post_max) new_post0 = -l_post_max;
                        }
                        // Evolving posterior stored in Bfloat16
                        bf_post0 = dual_buf_fp32_to_bf16(new_post0);
                        // Check edge update cast to Bfloat8 for compact storage
                        bf8_r0   = fp32_to_fp8_e5m2(r_new0);
                    }

                    uint16_t bf_post1 = static_cast<uint16_t>(old_post_pair >> 16);
                    uint8_t  bf8_r1   = static_cast<uint8_t>(old_r_pair >> 8);
                    if (!done1) {
                        float r_mag1 = (d == min_idx1) ? r_mag_min1 : r_mag_all1;
                        uint32_t node_sign1 = (float_to_raw_u32(q_val1[d]) >> 31);
                        uint32_t msg_sign1  = global_sign1 ^ node_sign1;
                        float r_new1 = (msg_sign1) ? -r_mag1 : r_mag1;
                        float new_post1 = q_val1[d] + r_new1;

                        if (l_post_max > 0.0f) {
                            if (new_post1 > l_post_max) new_post1 = l_post_max;
                            else if (new_post1 < -l_post_max) new_post1 = -l_post_max;
                        }
                        // Evolving posterior stored in Bfloat16
                        bf_post1 = dual_buf_fp32_to_bf16(new_post1);
                        // Check edge update cast to Bfloat8 for compact storage
                        bf8_r1   = fp32_to_fp8_e5m2(r_new1);
                    }

                    post_llrs_pair[vn] = static_cast<uint32_t>(bf_post0) | (static_cast<uint32_t>(bf_post1) << 16);
                    r_msg[m][d]        = static_cast<uint16_t>(bf8_r0)   | (static_cast<uint16_t>(bf8_r1) << 8);
                }
            }

            // STEP 4: Inline Syndrome Check (H * c^T == 0)
            if (iter >= 2) {
                uint32_t syndrome_errors_packed = 0;
                for (uint32_t m = 0; m < M_check_nodes; m++) {
                    uint32_t r_base = m * row_stride;
                    uint32_t d_count = h_col_idx[r_base];
                    const uint16_t* v_nodes = &h_col_idx[r_base + 1];
                    uint32_t row_parity_packed = 0;
                    for (uint32_t d = 0; d < d_count; d++) {
                        row_parity_packed ^= post_llrs_pair[v_nodes[d]];
                    }
                    syndrome_errors_packed |= (row_parity_packed & 0x80008000);
                    if ((done0 || (syndrome_errors_packed & 0x8000)) &&
                        (done1 || (syndrome_errors_packed & 0x80000000))) {
                        break;
                    }
                }

                if (!done0 && (syndrome_errors_packed & 0x8000) == 0) {
                    done0 = true;
                    if (cw == 0 && cw0_iters == 0) cw0_iters = iter + 1;
                }
                if (!done1 && (syndrome_errors_packed & 0x80000000) == 0) {
                    done1 = true;
                }
                if (done0 && done1) {
                    break;
                }
            }
        }

        if (cw == 0 && cw0_iters == 0) cw0_iters = max_iter;

        // STEP 5: Tally Bit and Frame Errors
        uint32_t cw_bit_errors0 = 0;
        for (uint32_t i = 0; i < unpunctured_nodes; i++) {
            cw_bit_errors0 += (post_llrs_pair[i] >> 15) & 1;
        }
        if (cw_bit_errors0 > 0) {
            total_bit_errors += cw_bit_errors0;
            total_frame_errors++;
        }

        if (has_cw1) {
            uint32_t cw_bit_errors1 = 0;
            for (uint32_t i = 0; i < unpunctured_nodes; i++) {
                cw_bit_errors1 += (post_llrs_pair[i] >> 31);
            }
            if (cw_bit_errors1 > 0) {
                total_bit_errors += cw_bit_errors1;
                total_frame_errors++;
            }
        }
    }

    sample_llr = (post_llrs_pair[0] & 0xFFFF) << 16;

    stats_out[0] = total_bit_errors;
    stats_out[1] = total_frame_errors;
    stats_out[2] = cw0_iters;
    stats_out[3] = sample_llr;
}
