#pragma once

#include <stdint.h>
#include <cstring>
#include <cmath>
#include "fixed_point_types.h"
#include "ldpc_decoder_core_dual_buffer.h"

// ==============================================================================
// Fixed-Point LDPC Decoder Core: Signed Q4.3 / Signed Q3.4
// ==============================================================================
// 1. Channel LLRs (L_v^ch):
//    - Signed Q4.3: Range [-16.0, +15.875], step 0.125 (scale 8.0)
//    - 2 codewords interleaved into 16-bit word (int8_t pair)
//    - Allows TRUE isolated channel clipping (L_ch_max)
//
// 2. Variable Node Posteriors (L_v^post):
//    - In Pure Fixed-Point: Signed Q4.3 (8-bit, scale 8.0)
//    - In Hybrid Fixed-Point: Bfloat16 (16-bit)
//
// 3. Check Node Edge Updates (r_m,d):
//    - Signed Q3.4: Range [-8.0, +7.9375], step 0.0625 (scale 16.0)
//    - 2 codewords interleaved into 16-bit word (int8_t pair)
//    - Recomputed every iteration with zero long-term error drift
// ==============================================================================

// ------------------------------------------------------------------------------
// Pure Fixed-Point Engine: Channel Q4.3, Posteriors Q4.3, Check Edges Q3.4
// Total Memory: 34.0 KiB / worker (170.0 KiB / core across 5 workers = 11.1% L1)
// ------------------------------------------------------------------------------
inline void decode_ldpc_fixed_point_all_q43_q34(
    uint32_t num_codewords,
    uint32_t worker_id,
    uint32_t N_var_nodes,
    uint32_t M_check_nodes,
    uint32_t max_check_deg,
    uint32_t P_punctured,
    uint32_t max_iter,
    float mu_llr,
    float sigma_llr,
    float l_ch_max,      // TRUE channel clipping limit
    float r_max,         // Check message clipping limit
    float l_post_max,    // Posterior accumulation limit
    uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3,
    const uint16_t* h_col_idx,
    uint16_t* channel_llrs_pair, // Buffer 1: Immutable L_v^ch in Signed Q4.3 (2 CW packed)
    uint16_t* post_llrs_pair,    // Buffer 2: Evolving L_v^post in Signed Q4.3 (2 CW packed)
    uint16_t* r_msg_base,        // Check edge updates in Signed Q3.4 (2 CW packed)
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

        // STEP 1: AWGN Channel Generation with TRUE Channel Clipping & Q4.3 Quantization
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

                // Quantize to Signed Q4.3
                int8_t q43_l0_cw0 = float_to_q43(l0_cw0);
                int8_t q43_l1_cw0 = float_to_q43(l1_cw0);
                int8_t q43_l0_cw1 = float_to_q43(l0_cw1);
                int8_t q43_l1_cw1 = float_to_q43(l1_cw1);

                uint16_t pair0 = static_cast<uint16_t>(static_cast<uint8_t>(q43_l0_cw0)) | 
                                (static_cast<uint16_t>(static_cast<uint8_t>(q43_l0_cw1)) << 8);
                uint16_t pair1 = static_cast<uint16_t>(static_cast<uint8_t>(q43_l1_cw0)) | 
                                (static_cast<uint16_t>(static_cast<uint8_t>(q43_l1_cw1)) << 8);

                channel_llrs_pair[i]     = pair0;
                channel_llrs_pair[i + 1] = pair1;

                post_llrs_pair[i]        = pair0;
                post_llrs_pair[i + 1]    = pair1;
            }
        } else {
            float mu_val = mu_llr;
            if (l_ch_max > 0.0f && mu_val > l_ch_max) mu_val = l_ch_max;
            int8_t mu_q43 = float_to_q43(mu_val);
            uint16_t mu_pair = static_cast<uint16_t>(static_cast<uint8_t>(mu_q43)) | 
                              (static_cast<uint16_t>(static_cast<uint8_t>(mu_q43)) << 8);
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

        // Reset check-node memory (Q3.4: 16-bit zeros)
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

                    uint16_t post_pair = post_llrs_pair[vn]; // Q4.3 pair
                    uint16_t r_pair = r_msg[m][d];          // Q3.4 pair

                    if (!done0) {
                        float post0 = q43_to_float(static_cast<int8_t>(post_pair & 0xFF));
                        float r0    = q34_to_float(static_cast<int8_t>(r_pair & 0xFF));
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
                        float post1 = q43_to_float(static_cast<int8_t>(post_pair >> 8));
                        float r1    = q34_to_float(static_cast<int8_t>(r_pair >> 8));
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
                    uint16_t old_post_pair = post_llrs_pair[vn];
                    uint16_t old_r_pair    = r_msg[m][d];

                    int8_t q43_post0 = static_cast<int8_t>(old_post_pair & 0xFF);
                    int8_t q34_r0    = static_cast<int8_t>(old_r_pair & 0xFF);
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
                        q43_post0 = float_to_q43(new_post0);
                        q34_r0    = float_to_q34(r_new0);
                    }

                    int8_t q43_post1 = static_cast<int8_t>(old_post_pair >> 8);
                    int8_t q34_r1    = static_cast<int8_t>(old_r_pair >> 8);
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
                        q43_post1 = float_to_q43(new_post1);
                        q34_r1    = float_to_q34(r_new1);
                    }

                    post_llrs_pair[vn] = static_cast<uint16_t>(static_cast<uint8_t>(q43_post0)) | 
                                        (static_cast<uint16_t>(static_cast<uint8_t>(q43_post1)) << 8);
                    r_msg[m][d]        = static_cast<uint16_t>(static_cast<uint8_t>(q34_r0)) | 
                                        (static_cast<uint16_t>(static_cast<uint8_t>(q34_r1)) << 8);
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

        // STEP 5: Tally Bit and Frame Errors (Sign bit in Q4.3 is bit 7)
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
// Hybrid Fixed-Point Engine: Channel Q4.3, Posteriors Bfloat16, Check Edges Q3.4
// Total Memory: 39.0 KiB / worker (195.0 KiB / core across 5 workers = 12.7% L1)
// ------------------------------------------------------------------------------
inline void decode_ldpc_fixed_point_hybrid(
    uint32_t num_codewords,
    uint32_t worker_id,
    uint32_t N_var_nodes,
    uint32_t M_check_nodes,
    uint32_t max_check_deg,
    uint32_t P_punctured,
    uint32_t max_iter,
    float mu_llr,
    float sigma_llr,
    float l_ch_max,      // TRUE channel clipping limit
    float r_max,         // Check message clipping limit
    float l_post_max,    // Posterior accumulation limit
    uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3,
    const uint16_t* h_col_idx,
    uint16_t* channel_llrs_pair, // Buffer 1: Immutable L_v^ch in Signed Q4.3 (2 CW packed)
    uint32_t* post_llrs_pair,    // Buffer 2: Evolving L_v^post in Bfloat16 (2 CW packed)
    uint16_t* r_msg_base,        // Check edge updates in Signed Q3.4 (2 CW packed)
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

        // STEP 1: AWGN Channel Generation with TRUE Channel Clipping & Q4.3 Quantization
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

                int8_t q43_l0_cw0 = float_to_q43(l0_cw0);
                int8_t q43_l1_cw0 = float_to_q43(l1_cw0);
                int8_t q43_l0_cw1 = float_to_q43(l0_cw1);
                int8_t q43_l1_cw1 = float_to_q43(l1_cw1);

                channel_llrs_pair[i]     = static_cast<uint16_t>(static_cast<uint8_t>(q43_l0_cw0)) | 
                                          (static_cast<uint16_t>(static_cast<uint8_t>(q43_l0_cw1)) << 8);
                channel_llrs_pair[i + 1] = static_cast<uint16_t>(static_cast<uint8_t>(q43_l1_cw0)) | 
                                          (static_cast<uint16_t>(static_cast<uint8_t>(q43_l1_cw1)) << 8);

                // Initial posterior in Bfloat16 expanded from Q4.3 channel observation
                uint16_t bf16_l0_cw0 = dual_buf_fp32_to_bf16(q43_to_float(q43_l0_cw0));
                uint16_t bf16_l0_cw1 = dual_buf_fp32_to_bf16(q43_to_float(q43_l0_cw1));
                uint16_t bf16_l1_cw0 = dual_buf_fp32_to_bf16(q43_to_float(q43_l1_cw0));
                uint16_t bf16_l1_cw1 = dual_buf_fp32_to_bf16(q43_to_float(q43_l1_cw1));

                post_llrs_pair[i]     = static_cast<uint32_t>(bf16_l0_cw0) | (static_cast<uint32_t>(bf16_l0_cw1) << 16);
                post_llrs_pair[i + 1] = static_cast<uint32_t>(bf16_l1_cw0) | (static_cast<uint32_t>(bf16_l1_cw1) << 16);
            }
        } else {
            float mu_val = mu_llr;
            if (l_ch_max > 0.0f && mu_val > l_ch_max) mu_val = l_ch_max;
            int8_t mu_q43 = float_to_q43(mu_val);
            uint16_t mu_pair_8 = static_cast<uint16_t>(static_cast<uint8_t>(mu_q43)) | 
                                (static_cast<uint16_t>(static_cast<uint8_t>(mu_q43)) << 8);

            uint16_t mu_bf = dual_buf_fp32_to_bf16(q43_to_float(mu_q43));
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

        // Reset check-node memory (Q3.4: 16-bit zeros)
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
                    uint16_t r_pair = r_msg[m][d];          // Q3.4 pair

                    if (!done0) {
                        float post0 = dual_buf_bf16_to_fp32(static_cast<uint16_t>(post_pair & 0xFFFF));
                        float r0    = q34_to_float(static_cast<int8_t>(r_pair & 0xFF));
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
                        float r1    = q34_to_float(static_cast<int8_t>(r_pair >> 8));
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
                    int8_t   q34_r0   = static_cast<int8_t>(old_r_pair & 0xFF);
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
                        q34_r0   = float_to_q34(r_new0);
                    }

                    uint16_t bf_post1 = static_cast<uint16_t>(old_post_pair >> 16);
                    int8_t   q34_r1   = static_cast<int8_t>(old_r_pair >> 8);
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
                        q34_r1   = float_to_q34(r_new1);
                    }

                    post_llrs_pair[vn] = static_cast<uint32_t>(bf_post0) | (static_cast<uint32_t>(bf_post1) << 16);
                    r_msg[m][d]        = static_cast<uint16_t>(static_cast<uint8_t>(q34_r0)) | 
                                        (static_cast<uint16_t>(static_cast<uint8_t>(q34_r1)) << 8);
                }
            }

            // STEP 4: Inline Syndrome Check
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
