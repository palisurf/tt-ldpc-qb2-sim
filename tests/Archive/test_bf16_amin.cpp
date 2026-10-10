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

constexpr uint32_t MAX_N = 4096;
constexpr uint32_t MAX_M = 2048;
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
        if (var_deg[n] > chinn.max_var_deg) {
            chinn.max_var_deg = var_deg[n];
        }
    }

    chinn.flattened_indices.assign(chinn.M * chinn.max_check_deg, 0xFFFF);
    for (uint32_t m = 0; m < chinn.M; m++) {
        for (size_t d = 0; d < check_nodes[m].size(); d++) {
            chinn.flattened_indices[m * chinn.max_check_deg + d] = check_nodes[m][d];
        }
    }
    return chinn;
}

// ---------------------------------------------------------
// PRNG: 128-bit Xoshiro128+
// ---------------------------------------------------------
struct Xoshiro128Plus {
    uint32_t s[4];
    static inline uint32_t rotl(const uint32_t x, int k) {
        return (x << k) | (x >> (32 - k));
    }
    inline uint32_t next() {
        const uint32_t result = s[0] + s[3];
        const uint32_t t = s[1] << 9;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 11);
        return result;
    }
    void seed(uint32_t seed_val) {
        uint32_t z = seed_val;
        for (int i = 0; i < 4; i++) {
            z += 0x9e3779b9;
            uint32_t sm = z ^ (z >> 16);
            sm *= 0x21f0aaad;
            sm ^= sm >> 15;
            sm *= 0x735a2d97;
            sm ^= sm >> 15;
            s[i] = (sm == 0) ? (0x12345678 + i) : sm;
        }
    }
};

inline void generate_awgn_pair(Xoshiro128Plus& rng, float mu, float sigma, float& l0, float& l1) {
    constexpr float INT32_TO_UNIT = 4.656612875245797e-10f;
    float v1, v2, s;
    do {
        int32_t u1 = static_cast<int32_t>(rng.next());
        int32_t u2 = static_cast<int32_t>(rng.next());
        v1 = static_cast<float>(u1) * INT32_TO_UNIT;
        v2 = static_cast<float>(u2) * INT32_TO_UNIT;
        s = v1 * v1 + v2 * v2;
    } while (s >= 1.0f || s <= 1e-15f);

    float factor = std::sqrt(-2.0f * std::log(s) / s);
    l0 = mu + sigma * (v1 * factor);
    l1 = mu + sigma * (v2 * factor);
}

// ---------------------------------------------------------
// Bfloat16 Precision Representation
// ---------------------------------------------------------
struct bf16 {
    uint16_t u = 0;
    constexpr bf16() = default;
    explicit bf16(float f) {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        // IEEE 754 round to nearest even
        uint32_t lsb = (bits >> 16) & 1u;
        uint32_t bias = 0x7FFFu + lsb;
        bits += bias;
        u = static_cast<uint16_t>(bits >> 16);
    }
    constexpr operator float() const {
        uint32_t bits = static_cast<uint32_t>(u) << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }
};

// Jones Approximate-Min* piecewise linear correction
inline float jones_approx_correction(float delta) {
    if (delta < 0.5f) {
        return 0.69315f - 0.40f * delta;
    } else if (delta < 2.0f) {
        return 0.49315f - 0.328f * (delta - 0.5f);
    }
    return 0.0f;
}

// 1. FP32 Normalized Min-Sum
void decode_nms_fp32(
    const ChinnMatrix& chinn, uint32_t P_punctured, uint32_t max_iter, float alpha,
    float* channel_llrs, float r_msg[MAX_M][MAX_DEG],
    uint32_t& bit_errs, uint32_t& frame_errs, uint32_t& iters_used
) {
    const uint16_t* h_col_idx = chinn.flattened_indices.data();
    uint32_t unpunctured = chinn.N - P_punctured;

    for (uint32_t iter = 0; iter < max_iter; iter++) {
        iters_used = iter + 1;
        for (uint32_t m = 0; m < chinn.M; m++) {
            float min1 = 999.0f, min2 = 999.0f;
            uint32_t min_idx = 0, global_sign = 0;
            float q_val[MAX_DEG];
            uint32_t actual_deg = 0;

            for (uint32_t d = 0; d < chinn.max_check_deg; d++) {
                uint16_t vn = h_col_idx[m * chinn.max_check_deg + d];
                if (vn == 0xFFFF) break;
                actual_deg++;
                q_val[d] = channel_llrs[vn] - r_msg[m][d];
                uint32_t sign = std::signbit(q_val[d]) ? 1 : 0;
                global_sign ^= sign;
                float abs_q = std::abs(q_val[d]);
                if (abs_q < min1) {
                    min2 = min1; min1 = abs_q; min_idx = d;
                } else if (abs_q < min2) {
                    min2 = abs_q;
                }
            }

            for (uint32_t d = 0; d < actual_deg; d++) {
                uint16_t vn = h_col_idx[m * chinn.max_check_deg + d];
                float current_min = (d == min_idx) ? min2 : min1;
                uint32_t node_sign = std::signbit(q_val[d]) ? 1 : 0;
                uint32_t msg_sign = global_sign ^ node_sign;
                float r_mag = alpha * current_min;
                float r_new = msg_sign ? -r_mag : r_mag;
                channel_llrs[vn] = q_val[d] + r_new;
                r_msg[m][d] = r_new;
            }
        }

        uint32_t syndrome_errors = 0;
        for (uint32_t m = 0; m < chinn.M; m++) {
            uint32_t row_parity = 0;
            for (uint32_t d = 0; d < chinn.max_check_deg; d++) {
                uint16_t vn = h_col_idx[m * chinn.max_check_deg + d];
                if (vn == 0xFFFF) break;
                row_parity ^= (std::signbit(channel_llrs[vn]) ? 1 : 0);
            }
            syndrome_errors |= row_parity;
            if (syndrome_errors != 0) break;
        }
        if (syndrome_errors == 0) break;
    }

    bit_errs = 0;
    for (uint32_t i = 0; i < unpunctured; i++) {
        bit_errs += (std::signbit(channel_llrs[i]) ? 1 : 0);
    }
    frame_errs = (bit_errs > 0) ? 1 : 0;
}

// 2. BF16 Normalized Min-Sum
void decode_nms_bf16(
    const ChinnMatrix& chinn, uint32_t P_punctured, uint32_t max_iter, float alpha,
    bf16* channel_llrs, bf16 r_msg[MAX_M][MAX_DEG],
    uint32_t& bit_errs, uint32_t& frame_errs, uint32_t& iters_used
) {
    const uint16_t* h_col_idx = chinn.flattened_indices.data();
    uint32_t unpunctured = chinn.N - P_punctured;

    for (uint32_t iter = 0; iter < max_iter; iter++) {
        iters_used = iter + 1;
        for (uint32_t m = 0; m < chinn.M; m++) {
            float min1 = 999.0f, min2 = 999.0f;
            uint32_t min_idx = 0, global_sign = 0;
            float q_val[MAX_DEG];
            uint32_t actual_deg = 0;

            for (uint32_t d = 0; d < chinn.max_check_deg; d++) {
                uint16_t vn = h_col_idx[m * chinn.max_check_deg + d];
                if (vn == 0xFFFF) break;
                actual_deg++;
                q_val[d] = static_cast<float>(channel_llrs[vn]) - static_cast<float>(r_msg[m][d]);
                uint32_t sign = std::signbit(q_val[d]) ? 1 : 0;
                global_sign ^= sign;
                float abs_q = std::abs(q_val[d]);
                if (abs_q < min1) {
                    min2 = min1; min1 = abs_q; min_idx = d;
                } else if (abs_q < min2) {
                    min2 = abs_q;
                }
            }

            for (uint32_t d = 0; d < actual_deg; d++) {
                uint16_t vn = h_col_idx[m * chinn.max_check_deg + d];
                float current_min = (d == min_idx) ? min2 : min1;
                uint32_t node_sign = std::signbit(q_val[d]) ? 1 : 0;
                uint32_t msg_sign = global_sign ^ node_sign;
                float r_mag = alpha * current_min;
                float r_new = msg_sign ? -r_mag : r_mag;
                channel_llrs[vn] = bf16(q_val[d] + r_new);
                r_msg[m][d] = bf16(r_new);
            }
        }

        uint32_t syndrome_errors = 0;
        for (uint32_t m = 0; m < chinn.M; m++) {
            uint32_t row_parity = 0;
            for (uint32_t d = 0; d < chinn.max_check_deg; d++) {
                uint16_t vn = h_col_idx[m * chinn.max_check_deg + d];
                if (vn == 0xFFFF) break;
                row_parity ^= (std::signbit(static_cast<float>(channel_llrs[vn])) ? 1 : 0);
            }
            syndrome_errors |= row_parity;
            if (syndrome_errors != 0) break;
        }
        if (syndrome_errors == 0) break;
    }

    bit_errs = 0;
    for (uint32_t i = 0; i < unpunctured; i++) {
        bit_errs += (std::signbit(static_cast<float>(channel_llrs[i])) ? 1 : 0);
    }
    frame_errs = (bit_errs > 0) ? 1 : 0;
}

// 3. FP32 Approximate-Min*
void decode_amin_fp32(
    const ChinnMatrix& chinn, uint32_t P_punctured, uint32_t max_iter,
    float* channel_llrs, float r_msg[MAX_M][MAX_DEG],
    uint32_t& bit_errs, uint32_t& frame_errs, uint32_t& iters_used
) {
    const uint16_t* h_col_idx = chinn.flattened_indices.data();
    uint32_t unpunctured = chinn.N - P_punctured;

    for (uint32_t iter = 0; iter < max_iter; iter++) {
        iters_used = iter + 1;
        for (uint32_t m = 0; m < chinn.M; m++) {
            float min1 = 999.0f, min2 = 999.0f, min3 = 999.0f;
            uint32_t min_idx = 0, global_sign = 0;
            float q_val[MAX_DEG];
            uint32_t actual_deg = 0;

            for (uint32_t d = 0; d < chinn.max_check_deg; d++) {
                uint16_t vn = h_col_idx[m * chinn.max_check_deg + d];
                if (vn == 0xFFFF) break;
                actual_deg++;
                q_val[d] = channel_llrs[vn] - r_msg[m][d];
                uint32_t sign = std::signbit(q_val[d]) ? 1 : 0;
                global_sign ^= sign;
                float abs_q = std::abs(q_val[d]);
                if (abs_q < min1) {
                    min3 = min2; min2 = min1; min1 = abs_q; min_idx = d;
                } else if (abs_q < min2) {
                    min3 = min2; min2 = abs_q;
                } else if (abs_q < min3) {
                    min3 = abs_q;
                }
            }

            float delta1 = min2 - min1;
            float delta2 = (actual_deg > 2) ? (min3 - min2) : delta1;

            float corr1 = jones_approx_correction(delta1);
            float corr2 = jones_approx_correction(delta2);

            float r_mag_all = std::max(0.0f, min1 - corr1);
            float r_mag_min = std::max(0.0f, min2 - corr2);

            for (uint32_t d = 0; d < actual_deg; d++) {
                uint16_t vn = h_col_idx[m * chinn.max_check_deg + d];
                float r_mag = (d == min_idx) ? r_mag_min : r_mag_all;
                uint32_t node_sign = std::signbit(q_val[d]) ? 1 : 0;
                uint32_t msg_sign = global_sign ^ node_sign;
                float r_new = msg_sign ? -r_mag : r_mag;
                channel_llrs[vn] = q_val[d] + r_new;
                r_msg[m][d] = r_new;
            }
        }

        uint32_t syndrome_errors = 0;
        for (uint32_t m = 0; m < chinn.M; m++) {
            uint32_t row_parity = 0;
            for (uint32_t d = 0; d < chinn.max_check_deg; d++) {
                uint16_t vn = h_col_idx[m * chinn.max_check_deg + d];
                if (vn == 0xFFFF) break;
                row_parity ^= (std::signbit(channel_llrs[vn]) ? 1 : 0);
            }
            syndrome_errors |= row_parity;
            if (syndrome_errors != 0) break;
        }
        if (syndrome_errors == 0) break;
    }

    bit_errs = 0;
    for (uint32_t i = 0; i < unpunctured; i++) {
        bit_errs += (std::signbit(channel_llrs[i]) ? 1 : 0);
    }
    frame_errs = (bit_errs > 0) ? 1 : 0;
}

// 4. BF16 Approximate-Min*
void decode_amin_bf16(
    const ChinnMatrix& chinn, uint32_t P_punctured, uint32_t max_iter,
    bf16* channel_llrs, bf16 r_msg[MAX_M][MAX_DEG],
    uint32_t& bit_errs, uint32_t& frame_errs, uint32_t& iters_used
) {
    const uint16_t* h_col_idx = chinn.flattened_indices.data();
    uint32_t unpunctured = chinn.N - P_punctured;

    for (uint32_t iter = 0; iter < max_iter; iter++) {
        iters_used = iter + 1;
        for (uint32_t m = 0; m < chinn.M; m++) {
            float min1 = 999.0f, min2 = 999.0f, min3 = 999.0f;
            uint32_t min_idx = 0, global_sign = 0;
            float q_val[MAX_DEG];
            uint32_t actual_deg = 0;

            for (uint32_t d = 0; d < chinn.max_check_deg; d++) {
                uint16_t vn = h_col_idx[m * chinn.max_check_deg + d];
                if (vn == 0xFFFF) break;
                actual_deg++;
                q_val[d] = static_cast<float>(channel_llrs[vn]) - static_cast<float>(r_msg[m][d]);
                uint32_t sign = std::signbit(q_val[d]) ? 1 : 0;
                global_sign ^= sign;
                float abs_q = std::abs(q_val[d]);
                if (abs_q < min1) {
                    min3 = min2; min2 = min1; min1 = abs_q; min_idx = d;
                } else if (abs_q < min2) {
                    min3 = min2; min2 = abs_q;
                } else if (abs_q < min3) {
                    min3 = abs_q;
                }
            }

            float delta1 = min2 - min1;
            float delta2 = (actual_deg > 2) ? (min3 - min2) : delta1;

            float corr1 = jones_approx_correction(delta1);
            float corr2 = jones_approx_correction(delta2);

            float r_mag_all = std::max(0.0f, min1 - corr1);
            float r_mag_min = std::max(0.0f, min2 - corr2);

            for (uint32_t d = 0; d < actual_deg; d++) {
                uint16_t vn = h_col_idx[m * chinn.max_check_deg + d];
                float r_mag = (d == min_idx) ? r_mag_min : r_mag_all;
                uint32_t node_sign = std::signbit(q_val[d]) ? 1 : 0;
                uint32_t msg_sign = global_sign ^ node_sign;
                float r_new = msg_sign ? -r_mag : r_mag;
                channel_llrs[vn] = bf16(q_val[d] + r_new);
                r_msg[m][d] = bf16(r_new);
            }
        }

        uint32_t syndrome_errors = 0;
        for (uint32_t m = 0; m < chinn.M; m++) {
            uint32_t row_parity = 0;
            for (uint32_t d = 0; d < chinn.max_check_deg; d++) {
                uint16_t vn = h_col_idx[m * chinn.max_check_deg + d];
                if (vn == 0xFFFF) break;
                row_parity ^= (std::signbit(static_cast<float>(channel_llrs[vn])) ? 1 : 0);
            }
            syndrome_errors |= row_parity;
            if (syndrome_errors != 0) break;
        }
        if (syndrome_errors == 0) break;
    }

    bit_errs = 0;
    for (uint32_t i = 0; i < unpunctured; i++) {
        bit_errs += (std::signbit(static_cast<float>(channel_llrs[i])) ? 1 : 0);
    }
    frame_errs = (bit_errs > 0) ? 1 : 0;
}

int main(int argc, char** argv) {
    std::string matrix_file = "matrices/AR4JA_r45_4c_128c_r12.chinn.out";
    ChinnMatrix chinn = parse_chinn_out(matrix_file);
    uint32_t P_punctured = 512;
    uint32_t unpunctured = chinn.N - P_punctured;
    double rate = static_cast<double>(chinn.N - chinn.M) / unpunctured;

    double eb_n0_db = 1.6;
    if (argc > 1) eb_n0_db = std::stod(argv[1]);
    uint32_t num_codewords = 10000;
    if (argc > 2) num_codewords = std::stoul(argv[2]);
    uint32_t max_iter = 16;
    if (argc > 3) max_iter = std::stoul(argv[3]);

    double snr_lin = std::pow(10.0, eb_n0_db / 10.0);
    float mu_llr = static_cast<float>(4.0 * rate * snr_lin);
    float sigma_llr = static_cast<float>(std::sqrt(8.0 * rate * snr_lin));

    std::cout << "=================================================================\n";
    std::cout << "   Precision Benchmark: FP32 vs Bfloat16 (NMS & Approx-Min*)     \n";
    std::cout << "=================================================================\n";
    std::cout << "Eb/N0: " << eb_n0_db << " dB | Codewords: " << num_codewords << "\n";
    std::cout << "Code: N=" << chinn.N << ", M=" << chinn.M << ", P=" << P_punctured 
              << " (Rate = " << rate << ")\n\n";

    struct Result {
        std::string name;
        uint64_t bit_errors = 0;
        uint64_t frame_errors = 0;
        uint64_t total_iters = 0;
        double elapsed_sec = 0.0;
    };

    std::vector<Result> results = {
        {"NMS FP32 (alpha=0.75)"},
        {"NMS BF16 (alpha=0.75)"},
        {"Approx-Min* FP32 (PWL)"},
        {"Approx-Min* BF16 (PWL)"}
    };

    static float llr_init[MAX_N];
    static float llr_fp32[MAX_N];
    static bf16 llr_bf16[MAX_N];
    static float r_msg_fp32[MAX_M][MAX_DEG];
    static bf16 r_msg_bf16[MAX_M][MAX_DEG];

    Xoshiro128Plus rng_gen;
    rng_gen.seed(1337);

    for (size_t dec_idx = 0; dec_idx < results.size(); dec_idx++) {
        Xoshiro128Plus rng = rng_gen; // Identical channel realizations for all decoders
        auto t0 = std::chrono::high_resolution_clock::now();

        for (uint32_t cw = 0; cw < num_codewords; cw++) {
            for (uint32_t i = 0; i < unpunctured; i += 2) {
                generate_awgn_pair(rng, mu_llr, sigma_llr, llr_init[i], llr_init[i + 1]);
            }
            for (uint32_t i = unpunctured; i < chinn.N; i++) {
                llr_init[i] = 0.0f;
            }

            uint32_t b_err = 0, f_err = 0, iters = 0;

            if (dec_idx == 0) {
                std::memcpy(llr_fp32, llr_init, chinn.N * sizeof(float));
                std::memset(r_msg_fp32, 0, sizeof(r_msg_fp32));
                decode_nms_fp32(chinn, P_punctured, max_iter, 0.75f, llr_fp32, r_msg_fp32, b_err, f_err, iters);
            } else if (dec_idx == 1) {
                for (uint32_t i = 0; i < chinn.N; i++) llr_bf16[i] = bf16(llr_init[i]);
                std::memset(r_msg_bf16, 0, sizeof(r_msg_bf16));
                decode_nms_bf16(chinn, P_punctured, max_iter, 0.75f, llr_bf16, r_msg_bf16, b_err, f_err, iters);
            } else if (dec_idx == 2) {
                std::memcpy(llr_fp32, llr_init, chinn.N * sizeof(float));
                std::memset(r_msg_fp32, 0, sizeof(r_msg_fp32));
                decode_amin_fp32(chinn, P_punctured, max_iter, llr_fp32, r_msg_fp32, b_err, f_err, iters);
            } else if (dec_idx == 3) {
                for (uint32_t i = 0; i < chinn.N; i++) llr_bf16[i] = bf16(llr_init[i]);
                std::memset(r_msg_bf16, 0, sizeof(r_msg_bf16));
                decode_amin_bf16(chinn, P_punctured, max_iter, llr_bf16, r_msg_bf16, b_err, f_err, iters);
            }

            results[dec_idx].bit_errors += b_err;
            results[dec_idx].frame_errors += f_err;
            results[dec_idx].total_iters += iters;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        results[dec_idx].elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
    }

    std::cout << "---------------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(26) << "Algorithm & Precision"
              << std::setw(14) << "Bit Errors"
              << std::setw(14) << "Frame Errors"
              << std::setw(14) << "FER"
              << std::setw(12) << "Avg Iters"
              << std::setw(12) << "Time (s)" << "\n";
    std::cout << "---------------------------------------------------------------------------------------\n";

    for (const auto& r : results) {
        double ber = static_cast<double>(r.bit_errors) / (static_cast<double>(num_codewords) * unpunctured);
        double fer = static_cast<double>(r.frame_errors) / num_codewords;
        double avg_it = static_cast<double>(r.total_iters) / num_codewords;

        std::cout << std::left << std::setw(26) << r.name
                  << std::setw(14) << r.bit_errors
                  << std::setw(14) << r.frame_errors
                  << std::setw(14) << std::scientific << std::setprecision(4) << fer
                  << std::fixed << std::setprecision(2)
                  << std::setw(12) << avg_it
                  << std::setprecision(3)
                  << std::setw(12) << r.elapsed_sec << "\n";
    }
    std::cout << "---------------------------------------------------------------------------------------\n";
    return 0;
}
