#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cassert>

constexpr uint32_t MAX_LOCAL_DEG = 12;

struct Xoshiro128Plus {
    uint32_t s[4];

    inline void init_state(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3) {
        s[0] = s0; s[1] = s1; s[2] = s2; s[3] = s3;
        if ((s[0] | s[1] | s[2] | s[3]) == 0) s[0] = 1;
    }

    inline uint32_t next() {
        const uint32_t result = s[0] + s[3];
        const uint32_t t = s[1] << 9;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t;
        s[3] = (s[3] << 11) | (s[3] >> 21);
        return result;
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

inline void generate_awgn_llr_pair(
    Xoshiro128Plus& rng, float mu_llr, float sigma_llr, float& llr0, float& llr1
) {
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
    float inv_s = rsqrt_s * rsqrt_s;
    float neg2_ln = -2.0f * fast_ln(s);
    float factor = fast_sqrt(neg2_ln * inv_s);

    llr0 = mu_llr + sigma_llr * (v1 * factor);
    llr1 = mu_llr + sigma_llr * (v2 * factor);
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

inline void compute_amin_star_deg3(
    const float* q_val, uint32_t min_idx, float min_mag,
    float& r_mag_min, float& r_mag_all
) {
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
}

inline void compute_amin_star_general(
    const float* q_val, uint32_t deg, uint32_t min_idx, float min_mag,
    float& r_mag_min, float& r_mag_all
) {
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

#include <fstream>
#include <sstream>

int main() {
    std::cout << "Testing 2-Codeword In-Place Freezing Logic on CCSDS AR4JA Matrix..." << std::endl;

    std::string matrix_path = "matrices/AR4JA_r45_4c_128c_r12.chinn.out";
    std::ifstream file(matrix_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << matrix_path << std::endl;
        return 1;
    }

    struct Edge { uint32_t r, c; float val; };
    std::vector<Edge> raw_lines;
    std::string line;
    while (std::getline(file, line)) {
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos || line[start] == '#') continue;
        std::istringstream iss(line);
        uint32_t r, c; float val;
        if (iss >> r >> c >> val) raw_lines.push_back({r, c, val});
    }

    uint32_t M = raw_lines.back().r;
    uint32_t N = raw_lines.back().c;
    uint32_t P = 512;
    uint32_t max_check_deg = 6;
    uint32_t row_stride = max_check_deg + 1;

    std::vector<std::vector<uint16_t>> adj(M);
    for (size_t i = 0; i + 1 < raw_lines.size(); i++) {
        const auto& edge = raw_lines[i];
        if (std::abs(edge.val) > 1e-6f) {
            adj[edge.r - 1].push_back(static_cast<uint16_t>(edge.c - 1));
        }
    }

    std::vector<uint16_t> h_matrix(M * row_stride, 0);
    for (uint32_t m = 0; m < M; m++) {
        uint32_t deg = adj[m].size();
        h_matrix[m * row_stride] = deg;
        for (uint32_t d = 0; d < deg; d++) {
            h_matrix[m * row_stride + 1 + d] = adj[m][d];
        }
    }

    std::cout << "Loaded AR4JA matrix: N=" << N << ", M=" << M << std::endl;

    // Test across 100 frame pairs with simulated AWGN at 2.0 dB
    // Eb/N0 = 2.0 dB -> Rate = 1/2 -> Es/N0 = -1.01 dB -> sigma = 1 / sqrt(2 * Es/N0)
    float rate = 0.5f;
    float eb_n0_lin = std::pow(10.0f, 2.0f / 10.0f);
    float sigma_ch = 1.0f / std::sqrt(2.0f * rate * eb_n0_lin);
    float mu_llr = 2.0f / (sigma_ch * sigma_ch);
    float sigma_llr = 2.0f / sigma_ch;

    std::cout << "Simulating AWGN at 2.0 dB (mu_llr=" << mu_llr << ", sigma_llr=" << sigma_llr << ")..." << std::endl;

    Xoshiro128Plus rng;
    rng.init_state(12345, 67890, 111213, 141516);

    std::vector<uint32_t> channel_llrs_pair(N);
    std::vector<std::vector<uint32_t>> r_msg(M, std::vector<uint32_t>(max_check_deg, 0));

    uint32_t unpunctured_nodes = N - P;
    uint32_t total_tested_pairs = 2500; // 5000 codewords
    uint32_t total_cw0_fe = 0, total_cw1_fe = 0;
    uint32_t total_cw0_be = 0, total_cw1_be = 0;
    uint32_t max_iter = 200;

    for (uint32_t pair = 0; pair < total_tested_pairs; pair++) {
        // Noise generation
        for (uint32_t i = 0; i < unpunctured_nodes; i += 2) {
            float l0_0 = 0.0f, l1_0 = 0.0f;
            float l0_1 = 0.0f, l1_1 = 0.0f;
            generate_awgn_llr_pair(rng, mu_llr, sigma_llr, l0_0, l1_0);
            generate_awgn_llr_pair(rng, mu_llr, sigma_llr, l0_1, l1_1);

            uint16_t bf0_0 = fp32_to_bf16(l0_0);
            uint16_t bf1_0 = fp32_to_bf16(l1_0);
            uint16_t bf0_1 = fp32_to_bf16(l0_1);
            uint16_t bf1_1 = fp32_to_bf16(l1_1);

            channel_llrs_pair[i]     = static_cast<uint32_t>(bf0_0) | (static_cast<uint32_t>(bf0_1) << 16);
            channel_llrs_pair[i + 1] = static_cast<uint32_t>(bf1_0) | (static_cast<uint32_t>(bf1_1) << 16);
        }
        for (uint32_t i = unpunctured_nodes; i < N; i++) {
            channel_llrs_pair[i] = 0;
        }

        // Reset check messages
        for (uint32_t m = 0; m < M; m++) {
            for (uint32_t d = 0; d < max_check_deg; d++) {
                r_msg[m][d] = 0;
            }
        }

        bool done0 = false, done1 = false;
        uint32_t cw0_iters = 0, cw1_iters = 0;

        for (uint32_t iter = 0; iter < max_iter; iter++) {
            if (done0 && done1) break;

            for (uint32_t m = 0; m < M; m++) {
                uint32_t row_base = m * row_stride;
                uint32_t deg = h_matrix[row_base];
                const uint16_t* vn_list = &h_matrix[row_base + 1];
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
                    uint32_t llr_pair = channel_llrs_pair[vn];
                    uint32_t r_pair = r_msg[m][d];

                    if (!done0) {
                        float llr0 = uint_as_float(llr_pair << 16);
                        float r0   = uint_as_float(r_pair << 16);
                        float q0   = llr0 - r0;
                        q_val0[d]  = q0;
                        uint32_t q0_u = float_as_uint(q0);
                        global_sign0 ^= (q0_u >> 31);
                        float abs_q0 = uint_as_float(q0_u & 0x7FFFFFFF);
                        if (abs_q0 < min_mag0) { min_mag0 = abs_q0; min_idx0 = d; }
                    }

                    if (!done1) {
                        float llr1 = uint_as_float(llr_pair & 0xFFFF0000);
                        float r1   = uint_as_float(r_pair & 0xFFFF0000);
                        float q1   = llr1 - r1;
                        q_val1[d]  = q1;
                        uint32_t q1_u = float_as_uint(q1);
                        global_sign1 ^= (q1_u >> 31);
                        float abs_q1 = uint_as_float(q1_u & 0x7FFFFFFF);
                        if (abs_q1 < min_mag1) { min_mag1 = abs_q1; min_idx1 = d; }
                    }
                }

                float r_mag_min0 = 0, r_mag_all0 = 0;
                float r_mag_min1 = 0, r_mag_all1 = 0;

                if (!done0) {
                    if (deg == 3) compute_amin_star_deg3(q_val0, min_idx0, min_mag0, r_mag_min0, r_mag_all0);
                    else         compute_amin_star_general(q_val0, deg, min_idx0, min_mag0, r_mag_min0, r_mag_all0);
                }
                if (!done1) {
                    if (deg == 3) compute_amin_star_deg3(q_val1, min_idx1, min_mag1, r_mag_min1, r_mag_all1);
                    else         compute_amin_star_general(q_val1, deg, min_idx1, min_mag1, r_mag_min1, r_mag_all1);
                }

                for (uint32_t d = 0; d < deg; d++) {
                    uint16_t vn = vn_cache[d];
                    uint32_t old_llr = channel_llrs_pair[vn];
                    uint32_t old_r   = r_msg[m][d];

                    uint32_t bf_llr0 = old_llr & 0xFFFF;
                    uint32_t bf_r0   = old_r   & 0xFFFF;
                    if (!done0) {
                        float r_mag0 = (d == min_idx0) ? r_mag_min0 : r_mag_all0;
                        uint32_t node_sign0 = float_as_uint(q_val0[d]) >> 31;
                        uint32_t msg_sign0 = global_sign0 ^ node_sign0;
                        float r_new0 = uint_as_float(float_as_uint(r_mag0) | (msg_sign0 << 31));
                        float new_llr0 = q_val0[d] + r_new0;
                        bf_llr0 = fp32_to_bf16(new_llr0);
                        bf_r0   = fp32_to_bf16(r_new0);
                    }

                    uint32_t bf_llr1 = old_llr >> 16;
                    uint32_t bf_r1   = old_r   >> 16;
                    if (!done1) {
                        float r_mag1 = (d == min_idx1) ? r_mag_min1 : r_mag_all1;
                        uint32_t node_sign1 = float_as_uint(q_val1[d]) >> 31;
                        uint32_t msg_sign1 = global_sign1 ^ node_sign1;
                        float r_new1 = uint_as_float(float_as_uint(r_mag1) | (msg_sign1 << 31));
                        float new_llr1 = q_val1[d] + r_new1;
                        bf_llr1 = fp32_to_bf16(new_llr1);
                        bf_r1   = fp32_to_bf16(r_new1);
                    }

                    channel_llrs_pair[vn] = bf_llr0 | (bf_llr1 << 16);
                    r_msg[m][d]           = bf_r0   | (bf_r1 << 16);
                }
            }

            // Syndrome check (bypass iter 0 and 1)
            if (iter >= 2) {
                uint32_t syndrome_errors_packed = 0;
                for (uint32_t m = 0; m < M; m++) {
                    uint32_t r_base = m * row_stride;
                    uint32_t d_count = h_matrix[r_base];
                    const uint16_t* v_nodes = &h_matrix[r_base + 1];
                    uint32_t row_parity_packed = 0;
                    for (uint32_t d = 0; d < d_count; d++) {
                        row_parity_packed ^= channel_llrs_pair[v_nodes[d]];
                    }
                    syndrome_errors_packed |= (row_parity_packed & 0x80008000);
                    if ((done0 || (syndrome_errors_packed & 0x8000)) &&
                        (done1 || (syndrome_errors_packed & 0x80000000))) {
                        break;
                    }
                }

                if (!done0 && (syndrome_errors_packed & 0x8000) == 0) {
                    done0 = true;
                    cw0_iters = iter + 1;
                }
                if (!done1 && (syndrome_errors_packed & 0x80000000) == 0) {
                    done1 = true;
                    cw1_iters = iter + 1;
                }
                if (done0 && done1) break;
            }
        }

        uint32_t be0 = 0, be1 = 0;
        for (uint32_t i = 0; i < unpunctured_nodes; i++) {
            be0 += (channel_llrs_pair[i] >> 15) & 1;
            be1 += (channel_llrs_pair[i] >> 31) & 1;
        }

        if (be0 > 0) { total_cw0_fe++; total_cw0_be += be0; }
        if (be1 > 0) { total_cw1_fe++; total_cw1_be += be1; }

        if ((pair + 1) % 25 == 0) {
            std::cout << "  Tested " << pair + 1 << " pairs: CW0 FE=" << total_cw0_fe << " (BE=" << total_cw0_be << "), CW1 FE=" << total_cw1_fe << " (BE=" << total_cw1_be << ")" << std::endl;
        }
    }

    std::cout << "\n=======================================================" << std::endl;
    std::cout << "FINAL RESULTS (200 Total Codewords at 2.0 dB):" << std::endl;
    std::cout << "  CW0: Frame Errors = " << total_cw0_fe << ", Bit Errors = " << total_cw0_be << std::endl;
    std::cout << "  CW1: Frame Errors = " << total_cw1_fe << ", Bit Errors = " << total_cw1_be << std::endl;
    std::cout << "=======================================================" << std::endl;

    assert(total_cw0_fe == 0);
    assert(total_cw1_fe == 0);
    assert(total_cw0_be == 0);
    assert(total_cw1_be == 0);

    std::cout << "\n>>> ALL 200 CODEWORDS DECODED WITH 0 BIT ERRORS! TEST PASSED! <<<\n" << std::endl;
    return 0;
}
