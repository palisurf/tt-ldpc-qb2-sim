#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cassert>

constexpr uint32_t MAX_N = 16384;
constexpr uint32_t MAX_M = 8192;
constexpr uint32_t MAX_DEG = 12;

struct ChinnMatrix {
    uint32_t N = 0;
    uint32_t M = 0;
    uint32_t max_var_deg = 0;
    uint32_t max_check_deg = 0;
    std::vector<uint16_t> flattened_indices;
};

ChinnMatrix parse_chinn_out(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) throw std::runtime_error("Could not open matrix file: " + filepath);

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
    if (raw_lines.empty()) throw std::runtime_error("Matrix file empty: " + filepath);

    Edge dim_line = raw_lines.back();
    ChinnMatrix chinn;
    chinn.M = dim_line.r;
    chinn.N = dim_line.c;

    std::vector<std::vector<uint16_t>> check_nodes(chinn.M);
    std::vector<uint32_t> var_deg(chinn.N, 0);

    for (size_t i = 0; i + 1 < raw_lines.size(); i++) {
        const auto& edge = raw_lines[i];
        if (std::abs(edge.val) > 1e-6f) {
            uint32_t m_idx = edge.r - 1;
            uint32_t vn_idx = edge.c - 1;
            check_nodes[m_idx].push_back(static_cast<uint16_t>(vn_idx));
            var_deg[vn_idx]++;
        }
    }

    chinn.max_check_deg = 0;
    for (uint32_t m = 0; m < chinn.M; m++) {
        if (check_nodes[m].size() > chinn.max_check_deg) {
            chinn.max_check_deg = static_cast<uint32_t>(check_nodes[m].size());
        }
    }
    chinn.max_var_deg = 0;
    for (uint32_t n = 0; n < chinn.N; n++) {
        if (var_deg[n] > chinn.max_var_deg) chinn.max_var_deg = var_deg[n];
    }

    chinn.flattened_indices.assign(chinn.M * chinn.max_check_deg, 0xFFFF);
    for (uint32_t m = 0; m < chinn.M; m++) {
        for (size_t d = 0; d < check_nodes[m].size(); d++) {
            chinn.flattened_indices[m * chinn.max_check_deg + d] = check_nodes[m][d];
        }
    }
    return chinn;
}

inline uint32_t xorshift32(uint32_t& state) {
    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return state = x;
}

inline float fast_ln(float s) {
    union { float f; uint32_t u; } pun;
    pun.f = s;
    int32_t exp = static_cast<int32_t>((pun.u >> 23) & 0xFF) - 127;
    pun.u = (pun.u & 0x007FFFFF) | 0x3F800000;
    float m = pun.f - 1.0f;

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
    union { float f; uint32_t u; } pun;
    pun.f = x;
    pun.u = 0x5F3759DF - (pun.u >> 1);
    float y = pun.f;
    float xhalf = 0.5f * x;
    y = y * (1.5f - xhalf * y * y);
    y = y * (1.5f - xhalf * y * y);
    return y;
}

inline float fast_sqrt(float x) { return x * fast_rsqrt(x); }

inline void generate_awgn_llr_pair(
    uint32_t& prng_state, float mu_llr, float sigma_llr, float& llr0, float& llr1
) {
    constexpr float INT32_TO_UNIT = 4.656612875245797e-10f;
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

int main(int argc, char** argv) {
    std::string matrix_file = "matrices/AR4JA_r45_4c_128c_r12.chinn.out";
    if (argc > 1) matrix_file = argv[1];

    std::cout << "Loading matrix: " << matrix_file << std::endl;
    ChinnMatrix chinn = parse_chinn_out(matrix_file);
    std::cout << "Dimensions: N=" << chinn.N << ", M=" << chinn.M 
              << ", max_check_deg=" << chinn.max_check_deg << std::endl;

    uint32_t P_punctured = 512;
    uint32_t unpunctured_nodes = chinn.N - P_punctured;
    double rate = static_cast<double>(chinn.N - chinn.M) / static_cast<double>(unpunctured_nodes);
    std::cout << "Punctured=" << P_punctured << ", Unpunctured=" << unpunctured_nodes 
              << ", Rate=" << rate << std::endl;

    double eb_n0_db = 2.0;
    double snr_lin = std::pow(10.0, eb_n0_db / 10.0);
    float mu_llr = static_cast<float>(4.0 * rate * snr_lin);
    float sigma_llr = static_cast<float>(std::sqrt(8.0 * rate * snr_lin));

    constexpr float alpha = 0.75f;
    constexpr uint32_t max_iter = 16;
    uint32_t prng_state = 133742;

    std::vector<float> channel_llrs(chinn.N);
    std::vector<std::vector<float>> r_msg(chinn.M, std::vector<float>(MAX_DEG, 0.0f));
    const uint16_t* h_col_idx = chinn.flattened_indices.data();
    uint32_t max_check_deg = chinn.max_check_deg;

    uint32_t num_codewords = 100;
    uint32_t total_bit_errors = 0;
    uint32_t total_frame_errors = 0;

    for (uint32_t cw = 0; cw < num_codewords; cw++) {
        // STEP 1: AWGN
        for (uint32_t i = 0; i < unpunctured_nodes; i += 2) {
            generate_awgn_llr_pair(prng_state, mu_llr, sigma_llr, channel_llrs[i], channel_llrs[i + 1]);
        }
        // STEP 2: Puncturing
        for (uint32_t i = unpunctured_nodes; i < chinn.N; i++) {
            channel_llrs[i] = 0.0f;
        }
        if (cw < 2) {
            std::cout << "HOST CW " << cw << " pre-decoding llrs[0..3]: "
                      << channel_llrs[0] << ", " << channel_llrs[1] << ", "
                      << channel_llrs[2] << ", " << channel_llrs[3] 
                      << " (prng_state=" << prng_state << ")" << std::endl;
        }

        // Reset check memory
        for (uint32_t m = 0; m < chinn.M; m++) {
            for (uint32_t d = 0; d < max_check_deg; d++) {
                r_msg[m][d] = 0.0f;
            }
        }

        // STEP 3: Layered Min-Sum
        for (uint32_t iter = 0; iter < max_iter; iter++) {
            for (uint32_t m = 0; m < chinn.M; m++) {
                float min1 = 999.0f, min2 = 999.0f;
                uint32_t min_idx = 0, global_sign = 0;
                float q_val[MAX_DEG];
                uint32_t actual_deg = 0;

                for (uint32_t d = 0; d < max_check_deg; d++) {
                    uint16_t vn = h_col_idx[m * max_check_deg + d];
                    if (vn == 0xFFFF) break;
                    actual_deg++;

                    q_val[d] = channel_llrs[vn] - r_msg[m][d];

                    union { float f; uint32_t u; } q_pun;
                    q_pun.f = q_val[d];
                    uint32_t sign = q_pun.u >> 31;
                    global_sign ^= sign;
                    q_pun.u &= 0x7FFFFFFF;
                    float abs_q = q_pun.f;

                    if (abs_q < min1) {
                        min2 = min1; min1 = abs_q; min_idx = d;
                    } else if (abs_q < min2) {
                        min2 = abs_q;
                    }
                }

                for (uint32_t d = 0; d < actual_deg; d++) {
                    uint16_t vn = h_col_idx[m * max_check_deg + d];
                    float current_min = (d == min_idx) ? min2 : min1;
                    union { float f; uint32_t u; } q_pun;
                    q_pun.f = q_val[d];
                    uint32_t node_sign = q_pun.u >> 31;
                    uint32_t msg_sign = global_sign ^ node_sign;

                    float r_mag = alpha * current_min;
                    union { float f; uint32_t u; } r_pun;
                    r_pun.f = r_mag;
                    r_pun.u |= (msg_sign << 31);
                    float r_new = r_pun.f;

                    channel_llrs[vn] = q_val[d] + r_new;
                    r_msg[m][d] = r_new;
                }
                if (cw == 0 && iter == 0 && m == 0) {
                    std::cout << "HOST m=0: min1=" << min1 << " min2=" << min2 
                              << " min_idx=" << min_idx << " global_sign=" << global_sign << std::endl;
                    for (uint32_t d = 0; d < actual_deg; d++) {
                        uint16_t vn = h_col_idx[d];
                        std::cout << "  d=" << d << " vn=" << vn << " q=" << q_val[d] 
                                  << " r=" << r_msg[0][d] << " new_llr=" << channel_llrs[vn] << std::endl;
                    }
                }
            }

            // STEP 4: Syndrome Check
            uint32_t syndrome_errors = 0;
            for (uint32_t m = 0; m < chinn.M; m++) {
                uint32_t row_parity = 0;
                for (uint32_t d = 0; d < max_check_deg; d++) {
                    uint16_t vn = h_col_idx[m * max_check_deg + d];
                    if (vn == 0xFFFF) break;
                    union { float f; uint32_t u; } vn_pun;
                    vn_pun.f = channel_llrs[vn];
                    row_parity ^= (vn_pun.u >> 31);
                }
                syndrome_errors |= row_parity;
                if (syndrome_errors != 0) break;
            }
            if (cw == 0) {
                std::cout << "HOST iter " << iter << ": syndrome_errors=" << syndrome_errors 
                          << ", llr[803]=" << channel_llrs[803] 
                          << ", llr[2347]=" << channel_llrs[2347] << std::endl;
            }
            if (syndrome_errors == 0) {
                if (cw < 10) {
                    std::cout << "HOST CW " << cw << " converged in " << iter + 1 << " iterations." << std::endl;
                }
                break; // Early termination on valid codeword
            }
        }

        // STEP 5: Tally errors
        uint32_t cw_bit_errors = 0;
        for (uint32_t i = 0; i < unpunctured_nodes; i++) {
            union { float f; uint32_t u; } vn_pun;
            vn_pun.f = channel_llrs[i];
            cw_bit_errors += (vn_pun.u >> 31);
        }
        if (cw_bit_errors > 0) {
            total_bit_errors += cw_bit_errors;
            total_frame_errors++;
        }
        if (cw < 10) {
            std::cout << "CW " << cw << " bit errors: " << cw_bit_errors << std::endl;
        }
    }

    std::cout << "--- EXACT C++ KERNEL CODE RUN ON HOST ---" << std::endl;
    std::cout << "Eb/N0: " << eb_n0_db << " dB, Blocks: " << num_codewords << std::endl;
    std::cout << "Bit Errors: " << total_bit_errors 
              << ", Frame Errors: " << total_frame_errors 
              << " (" << (double)total_frame_errors / num_codewords * 100.0 << "%)" << std::endl;
    return 0;
}
