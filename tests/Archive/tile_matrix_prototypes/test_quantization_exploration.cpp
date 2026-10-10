#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>

#include "../shared/fp8_types.h"
#include "../shared/fixed_point_types.h"
#include "../engines/scalar_5w/ldpc_decoder_core.h"
#include "../engines/scalar_5w/Archive/ldpc_decoder_core_dual_buffer.h"

// ==============================================================================
// Chinn Matrix Parser with Variable Node Degree Extraction
// ==============================================================================
struct ChinnMatrix {
    uint32_t N = 0;
    uint32_t M = 0;
    uint32_t max_var_deg = 0;
    uint32_t max_check_deg = 0;
    std::vector<uint16_t> flattened_indices;
    std::vector<uint8_t> var_deg;
};

ChinnMatrix parse_matrix_with_degrees(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) throw std::runtime_error("Could not open matrix: " + filepath);

    struct Edge { uint32_t r, c; float val; };
    std::vector<Edge> raw_lines;
    raw_lines.reserve(100000);

    std::string line;
    while (std::getline(file, line)) {
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos || line[start] == '#') continue;
        std::stringstream ss(line);
        uint32_t r = 0, c = 0;
        float val = 0.0f;
        if (ss >> r >> c >> val) {
            raw_lines.push_back({r, c, val});
        }
    }
    if (raw_lines.empty()) throw std::runtime_error("Empty matrix file");

    Edge dim_line = raw_lines.back();
    ChinnMatrix chinn;
    chinn.M = dim_line.r;
    chinn.N = dim_line.c;
    chinn.var_deg.assign(chinn.N, 0);

    std::vector<std::vector<uint16_t>> check_nodes(chinn.M);

    for (size_t i = 0; i + 1 < raw_lines.size(); i++) {
        const auto& edge = raw_lines[i];
        if (std::abs(edge.val) > 1e-6f) {
            uint32_t m_idx = edge.r - 1;
            uint32_t vn_idx = edge.c - 1;
            check_nodes[m_idx].push_back(static_cast<uint16_t>(vn_idx));
            chinn.var_deg[vn_idx]++;
        }
    }

    chinn.max_check_deg = 0;
    for (uint32_t m = 0; m < chinn.M; m++) {
        if (check_nodes[m].size() > chinn.max_check_deg) {
            chinn.max_check_deg = static_cast<uint32_t>(check_nodes[m].size());
        }
    }

    uint32_t stride = chinn.max_check_deg + 1;
    chinn.flattened_indices.assign(chinn.M * stride, 0xFFFF);
    for (uint32_t m = 0; m < chinn.M; m++) {
        uint32_t deg = static_cast<uint32_t>(check_nodes[m].size());
        chinn.flattened_indices[m * stride] = static_cast<uint16_t>(deg);
        for (size_t d = 0; d < deg && d < chinn.max_check_deg; d++) {
            chinn.flattened_indices[m * stride + 1 + d] = check_nodes[m][d];
        }
    }
    return chinn;
}

// ==============================================================================
// Quantization Enums & Codecs
// ==============================================================================
enum class ChannelFormat {
    BFLOAT16 = 0,
    Q4_3     = 1, // Signed Q4.3: Scale 8.0, step 0.125, range [-16.0, +15.875]
    Q3_4     = 2, // Signed Q3.4: Scale 16.0, step 0.0625, range [-8.0, +7.9375]
    Q5_2     = 3  // Signed Q5.2: Scale 4.0, step 0.25, range [-32.0, +31.75]
};

enum class EdgeFormat {
    BFLOAT16 = 0,
    Q3_4     = 1, // Signed Q3.4: Scale 16.0, step 0.0625, range [-8.0, +7.9375]
    SCALE20  = 2, // Scale 20.0: step 0.05, range [-6.4, +6.35]
    SCALE24  = 3, // Scale 24.0: step 0.04167, range [-5.33, +5.29]
    Q2_5     = 4, // Signed Q2.5: Scale 32.0, step 0.03125, range [-4.0, +3.96875]
    FP8_E4M3 = 5, // FP8 E4M3: 1 sign, 4 exp, 3 mantissa (high precision)
    FP8_E5M2 = 6  // FP8 E5M2: 1 sign, 5 exp, 2 mantissa (Bfloat8)
};

inline int8_t float_to_scale_clamp(float f, float scale, int8_t min_v = -128, int8_t max_v = 127) {
    float scaled = f * scale;
    int32_t rounded = (scaled >= 0.0f) ? static_cast<int32_t>(scaled + 0.5f) : static_cast<int32_t>(scaled - 0.5f);
    if (rounded > max_v) rounded = max_v;
    else if (rounded < min_v) rounded = min_v;
    return static_cast<int8_t>(rounded);
}

// Quantize float to 8-bit representation according to EdgeFormat
inline uint8_t quantize_edge_8bit(float r, EdgeFormat fmt) {
    switch (fmt) {
        case EdgeFormat::Q3_4:
            return static_cast<uint8_t>(float_to_q34(r));
        case EdgeFormat::SCALE20:
            return static_cast<uint8_t>(float_to_scale_clamp(r, 20.0f));
        case EdgeFormat::SCALE24:
            return static_cast<uint8_t>(float_to_scale_clamp(r, 24.0f));
        case EdgeFormat::Q2_5:
            return static_cast<uint8_t>(float_to_scale_clamp(r, 32.0f));
        case EdgeFormat::FP8_E4M3:
            return fp32_to_fp8_e4m3(r);
        case EdgeFormat::FP8_E5M2:
            return fp32_to_fp8_e5m2(r);
        default:
            return static_cast<uint8_t>(float_to_q34(r));
    }
}

// Dequantize 8-bit edge representation back to float
inline float dequantize_edge_8bit(uint8_t u, EdgeFormat fmt) {
    switch (fmt) {
        case EdgeFormat::Q3_4:
            return q34_to_float(static_cast<int8_t>(u));
        case EdgeFormat::SCALE20:
            return static_cast<float>(static_cast<int8_t>(u)) * (1.0f / 20.0f);
        case EdgeFormat::SCALE24:
            return static_cast<float>(static_cast<int8_t>(u)) * (1.0f / 24.0f);
        case EdgeFormat::Q2_5:
            return static_cast<float>(static_cast<int8_t>(u)) * (1.0f / 32.0f);
        case EdgeFormat::FP8_E4M3:
            return fp8_e4m3_to_fp32(u);
        case EdgeFormat::FP8_E5M2:
            return fp8_e5m2_to_fp32(u);
        default:
            return q34_to_float(static_cast<int8_t>(u));
    }
}

// Quantize float to 8-bit channel LLR
inline uint8_t quantize_channel_8bit(float ch, ChannelFormat fmt) {
    switch (fmt) {
        case ChannelFormat::Q4_3:
            return static_cast<uint8_t>(float_to_q43(ch));
        case ChannelFormat::Q3_4:
            return static_cast<uint8_t>(float_to_q34(ch));
        case ChannelFormat::Q5_2:
            return static_cast<uint8_t>(float_to_scale_clamp(ch, 4.0f));
        default:
            return static_cast<uint8_t>(float_to_q43(ch));
    }
}

inline float dequantize_channel_8bit(uint8_t u, ChannelFormat fmt) {
    switch (fmt) {
        case ChannelFormat::Q4_3:
            return q43_to_float(static_cast<int8_t>(u));
        case ChannelFormat::Q3_4:
            return q34_to_float(static_cast<int8_t>(u));
        case ChannelFormat::Q5_2:
            return static_cast<float>(static_cast<int8_t>(u)) * 0.25f;
        default:
            return q43_to_float(static_cast<int8_t>(u));
    }
}

// Clip modes for Degree-1 Variable Nodes (all strictly symmetric f(-L) = -f(L)):
// 0: None (unclipped)
// 1: |L| <= K * sigma
// 2: |L| <= mu
// 3: |L| <= mu + K * sigma
inline float apply_deg1_clip(float l, uint8_t deg, int clip_mode, float k_factor, float mu_llr, float sigma_llr) {
    if (deg != 1 || clip_mode == 0) return l;

    float limit = 0.0f;
    if (clip_mode == 1) { // |L| <= K * sigma
        limit = k_factor * sigma_llr;
    } else if (clip_mode == 2) { // |L| <= mu
        limit = mu_llr;
    } else if (clip_mode == 3) { // |L| <= mu + K * sigma
        limit = mu_llr + k_factor * sigma_llr;
    }

    if (limit > 0.0f) {
        if (l > limit) return limit;
        if (l < -limit) return -limit;
    }
    return l;
}

// ==============================================================================
// Unified Quantized Hybrid Belief Propagation Decoder
// ==============================================================================
inline void decode_ldpc_quantized_sweep(
    uint32_t num_codewords,
    uint32_t worker_id,
    uint32_t N_var_nodes,
    uint32_t M_check_nodes,
    uint32_t max_check_deg,
    uint32_t P_punctured,
    uint32_t max_iter,
    float mu_llr,
    float sigma_llr,
    int deg1_clip_mode,      // 0: none, 1: sym |L|<=K*sig, 2: asym L>=-K*sig, 3: mu+-K*sig
    float deg1_clip_k,       // K factor
    float general_ch_clip,   // global channel clip (0.0f = unclipped)
    ChannelFormat ch_fmt,
    EdgeFormat edge_fmt,
    uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3,
    const uint16_t* h_col_idx,
    const uint8_t* var_deg,
    uint32_t* stats_out
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
    uint32_t unpunctured_nodes  = N_var_nodes - P_punctured;
    uint32_t total_bit_errors   = 0;
    uint32_t total_frame_errors = 0;
    uint32_t cw0_iters          = 0;
    uint32_t sample_llr         = 0;

    // Buffer 1: Channel LLRs (packed 2 codewords: 16-bit word if 8-bit, 32-bit if BF16)
    std::vector<uint16_t> ch_llrs_8bit(N_var_nodes, 0);
    std::vector<uint32_t> ch_llrs_bf16(N_var_nodes, 0);

    // Buffer 2: Posterior LLRs in Bfloat16 (packed 2 codewords in uint32_t)
    std::vector<uint32_t> post_llrs(N_var_nodes, 0);

    // Buffer 3: Check edge updates (packed 2 codewords: 16-bit word if 8-bit, 32-bit if BF16)
    std::vector<uint16_t> r_msg_8bit(M_check_nodes * DUAL_BUF_MAX_DEG_STRIDE, 0);
    std::vector<uint32_t> r_msg_bf16(M_check_nodes * DUAL_BUF_MAX_DEG_STRIDE, 0);

    auto* r_msg_8 = reinterpret_cast<uint16_t(*)[DUAL_BUF_MAX_DEG_STRIDE]>(r_msg_8bit.data());
    auto* r_msg_16 = reinterpret_cast<uint32_t(*)[DUAL_BUF_MAX_DEG_STRIDE]>(r_msg_bf16.data());

    for (uint32_t cw = 0; cw < num_codewords; cw += 2) {
        bool has_cw1 = (cw + 1 < num_codewords);

        // STEP 1: AWGN Noise Generation & Degree-1 Selective Clipping
        if (sigma_llr > 0.0f) {
            for (uint32_t i = 0; i < unpunctured_nodes; i += 2) {
                float l0_cw0 = 0.0f, l1_cw0 = 0.0f;
                dual_buf_generate_awgn_llr_pair(rng, mu_llr, sigma_llr, l0_cw0, l1_cw0);

                l0_cw0 = apply_deg1_clip(l0_cw0, var_deg[i], deg1_clip_mode, deg1_clip_k, mu_llr, sigma_llr);
                if (general_ch_clip > 0.0f && var_deg[i] != 1) {
                    if (l0_cw0 > general_ch_clip) l0_cw0 = general_ch_clip;
                    else if (l0_cw0 < -general_ch_clip) l0_cw0 = -general_ch_clip;
                }

                l1_cw0 = apply_deg1_clip(l1_cw0, var_deg[i + 1], deg1_clip_mode, deg1_clip_k, mu_llr, sigma_llr);
                if (general_ch_clip > 0.0f && var_deg[i + 1] != 1) {
                    if (l1_cw0 > general_ch_clip) l1_cw0 = general_ch_clip;
                    else if (l1_cw0 < -general_ch_clip) l1_cw0 = -general_ch_clip;
                }

                // Codeword 1
                float l0_cw1 = 0.0f, l1_cw1 = 0.0f;
                dual_buf_generate_awgn_llr_pair(rng, mu_llr, sigma_llr, l0_cw1, l1_cw1);

                l0_cw1 = apply_deg1_clip(l0_cw1, var_deg[i], deg1_clip_mode, deg1_clip_k, mu_llr, sigma_llr);
                if (general_ch_clip > 0.0f && var_deg[i] != 1) {
                    if (l0_cw1 > general_ch_clip) l0_cw1 = general_ch_clip;
                    else if (l0_cw1 < -general_ch_clip) l0_cw1 = -general_ch_clip;
                }

                l1_cw1 = apply_deg1_clip(l1_cw1, var_deg[i + 1], deg1_clip_mode, deg1_clip_k, mu_llr, sigma_llr);
                if (general_ch_clip > 0.0f && var_deg[i + 1] != 1) {
                    if (l1_cw1 > general_ch_clip) l1_cw1 = general_ch_clip;
                    else if (l1_cw1 < -general_ch_clip) l1_cw1 = -general_ch_clip;
                }

                // Store in selected ChannelFormat and initialize Posteriors (Bfloat16)
                if (ch_fmt == ChannelFormat::BFLOAT16) {
                    uint16_t bf_l0_cw0 = dual_buf_fp32_to_bf16(l0_cw0);
                    uint16_t bf_l0_cw1 = dual_buf_fp32_to_bf16(l0_cw1);
                    uint16_t bf_l1_cw0 = dual_buf_fp32_to_bf16(l1_cw0);
                    uint16_t bf_l1_cw1 = dual_buf_fp32_to_bf16(l1_cw1);

                    ch_llrs_bf16[i]     = static_cast<uint32_t>(bf_l0_cw0) | (static_cast<uint32_t>(bf_l0_cw1) << 16);
                    ch_llrs_bf16[i + 1] = static_cast<uint32_t>(bf_l1_cw0) | (static_cast<uint32_t>(bf_l1_cw1) << 16);
                    post_llrs[i]        = ch_llrs_bf16[i];
                    post_llrs[i + 1]    = ch_llrs_bf16[i + 1];
                } else {
                    uint8_t q_l0_cw0 = quantize_channel_8bit(l0_cw0, ch_fmt);
                    uint8_t q_l0_cw1 = quantize_channel_8bit(l0_cw1, ch_fmt);
                    uint8_t q_l1_cw0 = quantize_channel_8bit(l1_cw0, ch_fmt);
                    uint8_t q_l1_cw1 = quantize_channel_8bit(l1_cw1, ch_fmt);

                    ch_llrs_8bit[i]     = static_cast<uint16_t>(q_l0_cw0) | (static_cast<uint16_t>(q_l0_cw1) << 8);
                    ch_llrs_8bit[i + 1] = static_cast<uint16_t>(q_l1_cw0) | (static_cast<uint16_t>(q_l1_cw1) << 8);

                    uint16_t bf_l0_cw0 = dual_buf_fp32_to_bf16(dequantize_channel_8bit(q_l0_cw0, ch_fmt));
                    uint16_t bf_l0_cw1 = dual_buf_fp32_to_bf16(dequantize_channel_8bit(q_l0_cw1, ch_fmt));
                    uint16_t bf_l1_cw0 = dual_buf_fp32_to_bf16(dequantize_channel_8bit(q_l1_cw0, ch_fmt));
                    uint16_t bf_l1_cw1 = dual_buf_fp32_to_bf16(dequantize_channel_8bit(q_l1_cw1, ch_fmt));

                    post_llrs[i]     = static_cast<uint32_t>(bf_l0_cw0) | (static_cast<uint32_t>(bf_l0_cw1) << 16);
                    post_llrs[i + 1] = static_cast<uint32_t>(bf_l1_cw0) | (static_cast<uint32_t>(bf_l1_cw1) << 16);
                }
            }
        }

        // STEP 2: Puncturing
        for (uint32_t i = unpunctured_nodes; i < N_var_nodes; i++) {
            ch_llrs_8bit[i] = 0;
            ch_llrs_bf16[i] = 0;
            post_llrs[i]    = 0;
        }

        // Reset check-node memory
        if (edge_fmt == EdgeFormat::BFLOAT16) {
            std::fill(r_msg_bf16.begin(), r_msg_bf16.end(), 0);
        } else {
            std::fill(r_msg_8bit.begin(), r_msg_8bit.end(), 0);
        }

        bool done0 = false;
        bool done1 = !has_cw1;
        uint32_t row_stride = max_check_deg + 1;

        // STEP 3: Belief Propagation Iterations
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

                    uint32_t post_pair = post_llrs[vn]; // Bfloat16 pair

                    float r0 = 0.0f, r1 = 0.0f;
                    if (edge_fmt == EdgeFormat::BFLOAT16) {
                        uint32_t r_pair = r_msg_16[m][d];
                        r0 = dual_buf_bf16_to_fp32(static_cast<uint16_t>(r_pair & 0xFFFF));
                        r1 = dual_buf_bf16_to_fp32(static_cast<uint16_t>(r_pair >> 16));
                    } else {
                        uint16_t r_pair = r_msg_8[m][d];
                        r0 = dequantize_edge_8bit(static_cast<uint8_t>(r_pair & 0xFF), edge_fmt);
                        r1 = dequantize_edge_8bit(static_cast<uint8_t>(r_pair >> 8), edge_fmt);
                    }

                    if (!done0) {
                        float post0 = dual_buf_bf16_to_fp32(static_cast<uint16_t>(post_pair & 0xFFFF));
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
                }

                if (!done1) {
                    if (deg == 3) {
                        dual_buf_compute_amin_star_deg3(q_val1, min_idx1, min_mag1, r_mag_min1, r_mag_all1);
                    } else {
                        dual_buf_compute_amin_star_general(q_val1, deg, min_idx1, min_mag1, r_mag_min1, r_mag_all1);
                    }
                }

                for (uint32_t d = 0; d < deg; d++) {
                    uint16_t vn = vn_cache[d];
                    uint32_t old_post_pair = post_llrs[vn];

                    uint16_t bf_post0 = static_cast<uint16_t>(old_post_pair & 0xFFFF);
                    float r_new0 = 0.0f;
                    if (!done0) {
                        float r_mag0 = (d == min_idx0) ? r_mag_min0 : r_mag_all0;
                        uint32_t node_sign0 = (float_to_raw_u32(q_val0[d]) >> 31);
                        uint32_t msg_sign0  = global_sign0 ^ node_sign0;
                        r_new0 = (msg_sign0) ? -r_mag0 : r_mag0;
                        float new_post0 = q_val0[d] + r_new0;
                        bf_post0 = dual_buf_fp32_to_bf16(new_post0);
                    }

                    uint16_t bf_post1 = static_cast<uint16_t>(old_post_pair >> 16);
                    float r_new1 = 0.0f;
                    if (!done1) {
                        float r_mag1 = (d == min_idx1) ? r_mag_min1 : r_mag_all1;
                        uint32_t node_sign1 = (float_to_raw_u32(q_val1[d]) >> 31);
                        uint32_t msg_sign1  = global_sign1 ^ node_sign1;
                        r_new1 = (msg_sign1) ? -r_mag1 : r_mag1;
                        float new_post1 = q_val1[d] + r_new1;
                        bf_post1 = dual_buf_fp32_to_bf16(new_post1);
                    }

                    post_llrs[vn] = static_cast<uint32_t>(bf_post0) | (static_cast<uint32_t>(bf_post1) << 16);

                    if (edge_fmt == EdgeFormat::BFLOAT16) {
                        uint16_t bf_r0 = dual_buf_fp32_to_bf16(r_new0);
                        uint16_t bf_r1 = dual_buf_fp32_to_bf16(r_new1);
                        r_msg_16[m][d] = static_cast<uint32_t>(bf_r0) | (static_cast<uint32_t>(bf_r1) << 16);
                    } else {
                        uint8_t q_r0 = quantize_edge_8bit(r_new0, edge_fmt);
                        uint8_t q_r1 = quantize_edge_8bit(r_new1, edge_fmt);
                        r_msg_8[m][d] = static_cast<uint16_t>(q_r0) | (static_cast<uint16_t>(q_r1) << 8);
                    }
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
                        row_parity_packed ^= post_llrs[v_nodes[d]];
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
                if (done0 && done1) break;
            }
        }

        if (cw == 0 && cw0_iters == 0) cw0_iters = max_iter;

        // STEP 5: Tally Bit and Frame Errors
        uint32_t cw_bit_errors0 = 0;
        for (uint32_t i = 0; i < unpunctured_nodes; i++) {
            cw_bit_errors0 += (post_llrs[i] >> 15) & 1;
        }
        if (cw_bit_errors0 > 0) {
            total_bit_errors += cw_bit_errors0;
            total_frame_errors++;
        }

        if (has_cw1) {
            uint32_t cw_bit_errors1 = 0;
            for (uint32_t i = 0; i < unpunctured_nodes; i++) {
                cw_bit_errors1 += (post_llrs[i] >> 31);
            }
            if (cw_bit_errors1 > 0) {
                total_bit_errors += cw_bit_errors1;
                total_frame_errors++;
            }
        }
    }

    sample_llr = (post_llrs[0] & 0xFFFF) << 16;
    stats_out[0] = total_bit_errors;
    stats_out[1] = total_frame_errors;
    stats_out[2] = cw0_iters;
    stats_out[3] = sample_llr;
}

// ==============================================================================
// Main Exploration Benchmark Driver
// ==============================================================================
int main(int argc, char** argv) {
    std::string matrix_path = "matrices/AR4JA_r45_4c_128c_r12.chinn.out";
    if (argc > 1) matrix_path = argv[1];

    std::cout << "====================================================================================" << std::endl;
    std::cout << "  Fixed-Point Hybrid Quantization & Degree-1 Channel Clipping Exploration Benchmark  " << std::endl;
    std::cout << "====================================================================================" << std::endl;

    ChinnMatrix chinn = parse_matrix_with_degrees(matrix_path);
    uint32_t deg1_count = 0;
    for (uint32_t i = 0; i < chinn.N - 512; i++) {
        if (chinn.var_deg[i] == 1) deg1_count++;
    }
    std::cout << "Matrix: N=" << chinn.N << ", M=" << chinn.M 
              << ", max_check_deg=" << chinn.max_check_deg 
              << " | Unpunctured Deg-1 Nodes: " << deg1_count << std::endl;

    uint32_t P_punctured = 512;
    uint32_t max_iter = 16;
    uint32_t test_blocks = 3000; // 1500 pairs of codewords

    std::vector<double> snr_list = {1.6, 1.8, 2.0, 2.2};

    uint32_t s0 = 0x12345678, s1 = 0x9ABCDEF0, s2 = 0x0FEDCBA9, s3 = 0x87654321;

    struct Config {
        std::string name;
        ChannelFormat ch_fmt;
        EdgeFormat edge_fmt;
        int deg1_mode;   // 0: none, 1: sym |L|<=K*sig, 2: asym L>=-K*sig, 3: mu+-K*sig
        float deg1_k;    // K factor
        float gen_clip;  // global channel clip (0 = none)
    };

    std::vector<Config> configs = {
        {"1. Baseline (Ch:BF16, Post:BF16, Edge:BF16)",           ChannelFormat::BFLOAT16, EdgeFormat::BFLOAT16, 0, 0.0f, 0.0f},
        {"2. Baseline + Deg1 Clip (|L| <= mu + 1.5*sig)",         ChannelFormat::BFLOAT16, EdgeFormat::BFLOAT16, 3, 1.5f, 0.0f},
        {"3. Hybrid (Ch:Q4.3, Post:BF16, Edge:BF16)",             ChannelFormat::Q4_3,     EdgeFormat::BFLOAT16, 0, 0.0f, 0.0f},
        {"4. Hybrid (Ch:Q4.3, Post:BF16, Edge:BF16) + Deg1 Clip", ChannelFormat::Q4_3,     EdgeFormat::BFLOAT16, 3, 1.5f, 0.0f},
        {"5. Hybrid (Ch:Q4.3, Post:BF16, Edge:Q3.4)",             ChannelFormat::Q4_3,     EdgeFormat::Q3_4,     0, 0.0f, 0.0f},
        {"6. Hybrid (Ch:Q4.3, Post:BF16, Edge:Q3.4) + Deg1 Clip", ChannelFormat::Q4_3,     EdgeFormat::Q3_4,     3, 1.5f, 0.0f}
    };

    for (double eb_n0_db : snr_list) {
        double rate = static_cast<double>(chinn.N - chinn.M) / static_cast<double>(chinn.N - P_punctured);
        double snr_lin = std::pow(10.0, eb_n0_db / 10.0);
        float mu_llr = static_cast<float>(4.0 * rate * snr_lin);
        float sigma_llr = static_cast<float>(std::sqrt(8.0 * rate * snr_lin));

        std::cout << "\n====================================================================================" << std::endl;
        std::cout << "Eb/N0 = " << std::fixed << std::setprecision(2) << eb_n0_db 
                  << " dB (mu=" << mu_llr << ", sigma=" << sigma_llr << ", blocks=" << test_blocks << ")" << std::endl;
        std::cout << "====================================================================================" << std::endl;

        std::cout << std::left << std::setw(52) << "Configuration"
                  << std::setw(12) << "Bit Errs"
                  << std::setw(12) << "Frame Errs"
                  << std::setw(14) << "FER"
                  << std::setw(12) << "Time (ms)" << std::endl;
        std::cout << std::string(102, '-') << std::endl;

        for (const auto& cfg : configs) {
            uint32_t stats[4] = {0};
            auto t0 = std::chrono::high_resolution_clock::now();
            decode_ldpc_quantized_sweep(
                test_blocks, 0, chinn.N, chinn.M, chinn.max_check_deg, P_punctured, max_iter,
                mu_llr, sigma_llr, cfg.deg1_mode, cfg.deg1_k, cfg.gen_clip,
                cfg.ch_fmt, cfg.edge_fmt,
                s0, s1, s2, s3,
                chinn.flattened_indices.data(),
                chinn.var_deg.data(),
                stats
            );
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            std::cout << std::left << std::setw(52) << cfg.name
                      << std::setw(12) << stats[0]
                      << std::setw(12) << stats[1]
                      << std::setw(14) << std::scientific << std::setprecision(3) << (double)stats[1] / test_blocks
                      << std::setw(12) << std::fixed << std::setprecision(1) << ms << std::endl;
        }
    }

    std::cout << "\n====================================================================================" << std::endl;
    std::cout << "                 Quantization Exploration Complete                                  " << std::endl;
    std::cout << "====================================================================================" << std::endl;
    return 0;
}
