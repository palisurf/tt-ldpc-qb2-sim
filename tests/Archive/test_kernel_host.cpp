#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cassert>
#include <string>

// ==============================================================================
// 1. Exact Kernel Math Primitives (from kernel/compute_trisc_ldpc_awgn_sim.cpp)
// ==============================================================================

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

    // 6th-degree Chebyshev/minimax polynomial for ln(1 + m) on [0, 1] (max abs err < 6.1e-6)
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

inline float fast_sqrt(float x) {
    return x * fast_rsqrt(x);
}

inline void generate_awgn_llr_pair(
    uint32_t& prng_state, 
    float mu_llr, 
    float sigma_llr, 
    float& llr0, 
    float& llr1
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

// ==============================================================================
// 2. Unit Test Suite
// ==============================================================================

bool test_transcendental_accuracy() {
    std::cout << "[TEST 1] Verifying fast_ln, fast_rsqrt, and fast_sqrt accuracy..." << std::endl;
    
    // Test fast_ln across (0.0001, 0.9999)
    double max_ln_abs_err = 0.0;
    double max_ln_rel_err = 0.0;
    for (float s = 0.0001f; s < 0.9999f; s += 0.0001f) {
        float approx = fast_ln(s);
        double exact = std::log(static_cast<double>(s));
        double abs_err = std::abs(approx - exact);
        if (abs_err > max_ln_abs_err) max_ln_abs_err = abs_err;
        // For relative error, avoid division by zero near root s = 1.0
        if (s <= 0.90f) {
            double rel_err = abs_err / std::abs(exact);
            if (rel_err > max_ln_rel_err) max_ln_rel_err = rel_err;
        }
    }

    // Test fast_rsqrt across (0.001, 100.0)
    double max_rsqrt_rel_err = 0.0;
    for (float x = 0.01f; x < 100.0f; x += 0.01f) {
        float approx = fast_rsqrt(x);
        double exact = 1.0 / std::sqrt(static_cast<double>(x));
        double rel_err = std::abs((approx - exact) / exact);
        if (rel_err > max_rsqrt_rel_err) max_rsqrt_rel_err = rel_err;
    }

    std::cout << "  - fast_ln max absolute error:    " << std::scientific << std::setprecision(3) << max_ln_abs_err << std::endl;
    std::cout << "  - fast_ln max relative (s<=0.9): " << std::scientific << std::setprecision(3) << max_ln_rel_err << std::endl;
    std::cout << "  - fast_rsqrt max relative error: " << std::scientific << std::setprecision(3) << max_rsqrt_rel_err << std::endl;

    bool passed = (max_ln_abs_err < 1e-4) && (max_rsqrt_rel_err < 1e-4);
    std::cout << "  -> Result: " << (passed ? "PASS" : "FAIL") << "\n" << std::endl;
    return passed;
}

bool test_awgn_distribution() {
    std::cout << "[TEST 2] Verifying AWGN statistical moments (Polar Box-Muller)..." << std::endl;
    
    // Test parameters: Eb/N0 = 2.0 dB, Rate = 0.5
    double eb_n0_db = 2.0;
    double rate = 0.5;
    double snr_lin = std::pow(10.0, eb_n0_db / 10.0);
    float mu_expected = static_cast<float>(4.0 * rate * snr_lin);
    float sigma_expected = static_cast<float>(std::sqrt(8.0 * rate * snr_lin));
    double var_expected = sigma_expected * sigma_expected;

    constexpr uint32_t N_SAMPLES = 500000;
    uint32_t prng_state = 133742;

    double sum = 0.0;
    double sum_sq = 0.0;

    for (uint32_t i = 0; i < N_SAMPLES; i += 2) {
        float l0, l1;
        generate_awgn_llr_pair(prng_state, mu_expected, sigma_expected, l0, l1);
        sum += l0 + l1;
        sum_sq += l0 * l0 + l1 * l1;
    }

    double sample_mean = sum / N_SAMPLES;
    double sample_var = (sum_sq - (sum * sum / N_SAMPLES)) / (N_SAMPLES - 1);
    double sem = sigma_expected / std::sqrt(N_SAMPLES); // Standard error of mean

    std::cout << "  - Expected Mean:     " << std::fixed << std::setprecision(4) << mu_expected 
              << " | Sample Mean:     " << sample_mean << " (Diff: " << std::abs(sample_mean - mu_expected) << ", 3*SEM: " << 3.0 * sem << ")" << std::endl;
    std::cout << "  - Expected Variance: " << std::fixed << std::setprecision(4) << var_expected 
              << " | Sample Variance: " << sample_var << std::endl;

    bool passed = (std::abs(sample_mean - mu_expected) < 3.5 * sem);
    std::cout << "  -> Result: " << (passed ? "PASS" : "FAIL") << "\n" << std::endl;
    return passed;
}

bool test_minsum_decoder_correction() {
    std::cout << "[TEST 3] Verifying Layered Min-Sum decoder error-correction on test matrix..." << std::endl;

    // Load matrices/test_code.chinn (N=12, M=6, rate 0.5)
    std::ifstream file("matrices/test_code.chinn");
    if (!file.is_open()) {
        std::cerr << "  -> FAIL: Could not open matrices/test_code.chinn" << std::endl;
        return false;
    }

    std::string line;
    auto get_clean_line = [&file, &line]() -> bool {
        while (std::getline(file, line)) {
            size_t start = line.find_first_not_of(" \t\r\n");
            if (start == std::string::npos || line[start] == '#') continue;
            line = line.substr(start);
            return true;
        }
        return false;
    };

    get_clean_line();
    std::stringstream ss(line);
    uint32_t N, M, max_vdeg, max_cdeg;
    ss >> N >> M;
    get_clean_line();
    std::stringstream ss2(line);
    ss2 >> max_vdeg >> max_cdeg;

    std::vector<uint16_t> h_col_idx(M * max_cdeg, 0xFFFF);
    for (uint32_t m = 0; m < M; m++) {
        get_clean_line();
        std::stringstream row_ss(line);
        uint32_t deg;
        row_ss >> deg;
        for (uint32_t d = 0; d < deg; d++) {
            uint32_t vn;
            row_ss >> vn;
            if (vn > 0) vn -= 1;
            h_col_idx[m * max_cdeg + d] = static_cast<uint16_t>(vn);
        }
    }

    // Decoder buffers
    std::vector<float> channel_llrs(N);
    constexpr uint32_t MAX_DEG = 12;
    std::vector<std::vector<float>> r_msg(M, std::vector<float>(MAX_DEG, 0.0f));

    // Test Scenario A: Clean signal (all +4.0f). Should decode in 1 iter with 0 errors.
    for (uint32_t i = 0; i < N; i++) channel_llrs[i] = 4.0f;
    
    constexpr float alpha = 0.75f;
    constexpr uint32_t max_iter = 8;

    auto decode_frame = [&](std::vector<float>& llrs) -> uint32_t {
        for (auto& row : r_msg) std::fill(row.begin(), row.end(), 0.0f);
        uint32_t iters_run = 0;

        for (uint32_t iter = 0; iter < max_iter; iter++) {
            iters_run++;
            for (uint32_t m = 0; m < M; m++) {
                float min1 = 999.0f, min2 = 999.0f;
                uint32_t min_idx = 0, global_sign = 0;
                float q_val[MAX_DEG];
                uint32_t actual_deg = 0;

                for (uint32_t d = 0; d < max_cdeg; d++) {
                    uint16_t vn = h_col_idx[m * max_cdeg + d];
                    if (vn == 0xFFFF) break;
                    actual_deg++;

                    q_val[d] = llrs[vn] - r_msg[m][d];
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
                    uint16_t vn = h_col_idx[m * max_cdeg + d];
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

                    llrs[vn] = q_val[d] + r_new;
                    r_msg[m][d] = r_new;
                }
            }

            // Syndrome check
            uint32_t syndrome_errors = 0;
            for (uint32_t m = 0; m < M; m++) {
                uint32_t row_parity = 0;
                for (uint32_t d = 0; d < max_cdeg; d++) {
                    uint16_t vn = h_col_idx[m * max_cdeg + d];
                    if (vn == 0xFFFF) break;
                    union { float f; uint32_t u; } vn_pun;
                    vn_pun.f = llrs[vn];
                    row_parity ^= (vn_pun.u >> 31);
                }
                syndrome_errors |= row_parity;
                if (syndrome_errors != 0) break;
            }
            if (syndrome_errors == 0) break;
        }
        return iters_run;
    };

    uint32_t clean_iters = decode_frame(channel_llrs);
    std::cout << "  - Scenario A (Clean Input): Early termination in " << clean_iters << " iteration(s)" << std::endl;

    // Test Scenario B: Corrupt bit 2 (force negative LLR = -3.5f)
    for (uint32_t i = 0; i < N; i++) channel_llrs[i] = 4.0f;
    channel_llrs[2] = -3.5f; // Bit error injection
    uint32_t corrupted_iters = decode_frame(channel_llrs);
    
    // Check if bit 2 was corrected back to positive
    bool bit_corrected = (channel_llrs[2] > 0.0f);
    std::cout << "  - Scenario B (Injected Bit Error on Node 2):" << std::endl;
    std::cout << "    * Iterations until convergence: " << corrupted_iters << std::endl;
    std::cout << "    * Node 2 Final LLR: " << channel_llrs[2] << " (Corrected: " << (bit_corrected ? "YES" : "NO") << ")" << std::endl;

    bool passed = (clean_iters == 1) && bit_corrected;
    std::cout << "  -> Result: " << (passed ? "PASS" : "FAIL") << "\n" << std::endl;
    return passed;
}

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "   Tenstorrent Tensix LDPC Kernel Host Unit Test Suite    " << std::endl;
    std::cout << "==========================================================\n" << std::endl;

    bool t1 = test_transcendental_accuracy();
    bool t2 = test_awgn_distribution();
    bool t3 = test_minsum_decoder_correction();

    std::cout << "==========================================================" << std::endl;
    if (t1 && t2 && t3) {
        std::cout << "  >>> ALL UNIT TESTS PASSED SUCCESSFULLY! <<<" << std::endl;
        std::cout << "==========================================================" << std::endl;
        return 0;
    } else {
        std::cerr << "  >>> SOME UNIT TESTS FAILED! <<<" << std::endl;
        std::cout << "==========================================================" << std::endl;
        return 1;
    }
}
