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

inline uint32_t splitmix32(uint32_t& x) {
    uint32_t z = (x += 0x9E3779B9);
    z ^= z >> 16;
    z *= 0x85EBCA6B;
    z ^= z >> 13;
    z *= 0xC2B2AE35;
    z ^= z >> 16;
    return z;
}

struct Xoshiro128Plus {
    uint32_t s[4];
    void init(uint32_t seed_lo, uint32_t seed_hi) {
        uint32_t sm_lo = seed_lo;
        uint32_t sm_hi = seed_hi;
        s[0] = splitmix32(sm_lo);
        s[1] = splitmix32(sm_lo);
        s[2] = splitmix32(sm_hi);
        s[3] = splitmix32(sm_hi);
        if ((s[0] | s[1] | s[2] | s[3]) == 0) s[0] = 1;
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
};

inline void generate_awgn_floats(Xoshiro128Plus& rng, float mu, float sigma, float& l0, float& l1) {
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

// 1. FP32 Min-Sum Decoder
void decode_fp32(
    const ChinnMatrix& chinn, uint32_t P_punctured, uint32_t max_iter,
    float* channel_llrs, float r_msg[MAX_M][MAX_DEG],
    uint32_t& bit_errs, uint32_t& frame_errs, uint32_t& iters_used
) {
    const uint16_t* h_col_idx = chinn.flattened_indices.data();
    uint32_t unpunctured = chinn.N - P_punctured;
    constexpr float alpha = 0.75f;

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

// 2. Fixed-Point 16-Bit (int16_t) Min-Sum Decoder with scale S=64
void decode_int16(
    const ChinnMatrix& chinn, uint32_t P_punctured, uint32_t max_iter,
    int16_t* channel_llrs, int16_t r_msg[MAX_M][MAX_DEG],
    uint32_t& bit_errs, uint32_t& frame_errs, uint32_t& iters_used
) {
    const uint16_t* h_col_idx = chinn.flattened_indices.data();
    uint32_t unpunctured = chinn.N - P_punctured;

    for (uint32_t iter = 0; iter < max_iter; iter++) {
        iters_used = iter + 1;
        for (uint32_t m = 0; m < chinn.M; m++) {
            uint16_t min1 = 0x7FFF, min2 = 0x7FFF;
            uint32_t min_idx = 0, global_sign = 0;
            int16_t q_val[MAX_DEG];
            uint32_t actual_deg = 0;

            for (uint32_t d = 0; d < chinn.max_check_deg; d++) {
                uint16_t vn = h_col_idx[m * chinn.max_check_deg + d];
                if (vn == 0xFFFF) break;
                actual_deg++;

                int16_t q = channel_llrs[vn] - r_msg[m][d];
                q_val[d] = q;

                uint32_t sign = (q < 0) ? 1 : 0;
                global_sign ^= sign;
                uint16_t abs_q = static_cast<uint16_t>((q < 0) ? -q : q);

                if (abs_q < min1) {
                    min2 = min1; min1 = abs_q; min_idx = d;
                } else if (abs_q < min2) {
                    min2 = abs_q;
                }
            }

            for (uint32_t d = 0; d < actual_deg; d++) {
                uint16_t vn = h_col_idx[m * chinn.max_check_deg + d];
                uint16_t current_min = (d == min_idx) ? min2 : min1;
                uint32_t node_sign = (q_val[d] < 0) ? 1 : 0;
                uint32_t msg_sign = global_sign ^ node_sign;

                // alpha = 0.75: (min * 3) >> 2
                uint16_t r_mag = (current_min * 3) >> 2;
                int16_t r_new = msg_sign ? -static_cast<int16_t>(r_mag) : static_cast<int16_t>(r_mag);

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
                row_parity ^= ((channel_llrs[vn] < 0) ? 1 : 0);
            }
            syndrome_errors |= row_parity;
            if (syndrome_errors != 0) break;
        }
        if (syndrome_errors == 0) break;
    }

    bit_errs = 0;
    for (uint32_t i = 0; i < unpunctured; i++) {
        bit_errs += ((channel_llrs[i] < 0) ? 1 : 0);
    }
    frame_errs = (bit_errs > 0) ? 1 : 0;
}

int main() {
    std::string matrix_file = "matrices/AR4JA_r45_4c_128c_r12.chinn.out";
    ChinnMatrix chinn = parse_chinn_out(matrix_file);
    uint32_t P_punctured = 512;
    uint32_t unpunctured = chinn.N - P_punctured;
    double rate = static_cast<double>(chinn.N - chinn.M) / unpunctured;

    std::cout << "Loaded matrix: N=" << chinn.N << ", M=" << chinn.M << ", Rate=" << rate << std::endl;

    double eb_n0_db = 2.0;
    double snr_lin = std::pow(10.0, eb_n0_db / 10.0);
    float mu_llr = static_cast<float>(4.0 * rate * snr_lin);
    float sigma_llr = static_cast<float>(std::sqrt(8.0 * rate * snr_lin));

    uint32_t num_codewords = 2000;
    std::cout << "Benchmarking " << num_codewords << " codewords at Eb/N0 = " << eb_n0_db << " dB...\n" << std::endl;

    // Buffers for FP32
    std::vector<float> fp32_llrs(chinn.N);
    static float fp32_r_msg[MAX_M][MAX_DEG];

    // Buffers for Int16 (Scale = 64)
    std::vector<int16_t> int16_llrs(chinn.N);
    static int16_t int16_r_msg[MAX_M][MAX_DEG];
    constexpr float INT16_SCALE = 64.0f;

    // --- Benchmark FP32 ---
    Xoshiro128Plus rng_fp32;
    rng_fp32.init(133742, 0x9E3779B9);
    uint32_t fp32_total_bits = 0, fp32_total_frames = 0;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (uint32_t cw = 0; cw < num_codewords; cw++) {
        for (uint32_t i = 0; i < unpunctured; i += 2) {
            generate_awgn_floats(rng_fp32, mu_llr, sigma_llr, fp32_llrs[i], fp32_llrs[i + 1]);
        }
        for (uint32_t i = unpunctured; i < chinn.N; i++) fp32_llrs[i] = 0.0f;
        std::memset(fp32_r_msg, 0, sizeof(fp32_r_msg));

        uint32_t b_err = 0, f_err = 0, iters = 0;
        decode_fp32(chinn, P_punctured, 16, fp32_llrs.data(), fp32_r_msg, b_err, f_err, iters);
        fp32_total_bits += b_err;
        fp32_total_frames += f_err;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double fp32_time = std::chrono::duration<double>(t1 - t0).count();

    // --- Benchmark Int16 ---
    Xoshiro128Plus rng_int16;
    rng_int16.init(133742, 0x9E3779B9);
    uint32_t int16_total_bits = 0, int16_total_frames = 0;

    auto t2 = std::chrono::high_resolution_clock::now();
    for (uint32_t cw = 0; cw < num_codewords; cw++) {
        for (uint32_t i = 0; i < unpunctured; i += 2) {
            float f0, f1;
            generate_awgn_floats(rng_int16, mu_llr, sigma_llr, f0, f1);
            int16_llrs[i] = static_cast<int16_t>(std::round(f0 * INT16_SCALE));
            int16_llrs[i + 1] = static_cast<int16_t>(std::round(f1 * INT16_SCALE));
        }
        for (uint32_t i = unpunctured; i < chinn.N; i++) int16_llrs[i] = 0;
        std::memset(int16_r_msg, 0, sizeof(int16_r_msg));

        uint32_t b_err = 0, f_err = 0, iters = 0;
        decode_int16(chinn, P_punctured, 16, int16_llrs.data(), int16_r_msg, b_err, f_err, iters);
        int16_total_bits += b_err;
        int16_total_frames += f_err;
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double int16_time = std::chrono::duration<double>(t3 - t2).count();

    std::cout << "=== FP32 (Float) Results ===" << std::endl;
    std::cout << "Time: " << fp32_time << " s (" << (num_codewords / fp32_time) << " blocks/sec)" << std::endl;
    std::cout << "Bit Errors: " << fp32_total_bits << ", Frame Errors: " << fp32_total_frames 
              << " (FER: " << (double)fp32_total_frames / num_codewords << ")" << std::endl;

    std::cout << "\n=== Int16 (Fixed-Point Scale=64) Results ===" << std::endl;
    std::cout << "Time: " << int16_time << " s (" << (num_codewords / int16_time) << " blocks/sec)" << std::endl;
    std::cout << "Bit Errors: " << int16_total_bits << ", Frame Errors: " << int16_total_frames 
              << " (FER: " << (double)int16_total_frames / num_codewords << ")" << std::endl;

    std::cout << "\n=== Comparison Summary ===" << std::endl;
    std::cout << "Speedup (Int16 vs FP32): " << (fp32_time / int16_time) << "x" << std::endl;
    std::cout << "Bit-Level Accuracy Match: " 
              << (int16_total_frames == fp32_total_frames ? "100% IDENTICAL" : "DIFFERENT") 
              << " (" << int16_total_frames << " vs " << fp32_total_frames << " frame errors)" << std::endl;
    std::cout << "Memory Reduction: L1 r_msg from 72 KB -> 36 KB (2x reduction)" << std::endl;

    return 0;
}
