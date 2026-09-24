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

// Exact Jacobian correction: g(delta) = ln(1 + e^(-delta))
inline float exact_jacobian_correction(float delta) {
    return std::log(1.0f + std::exp(-delta));
}

// Jones Approximate-Min* piecewise linear correction (shift-and-add friendly, zero LUT)
inline float jones_approx_correction(float delta) {
    // Exact at delta=0 is ln(2)=0.693, drops to 0 at delta ~ 2.0
    if (delta < 0.5f) {
        return 0.693f - 0.40f * delta;
    } else if (delta < 2.0f) {
        return 0.493f - 0.328f * (delta - 0.5f);
    }
    return 0.0f;
}

// 1. Normalized Min-Sum (NMS)
void decode_nms(
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

// 2. Christopher Jones Approximate-Min* Decoder (Two Outgoing Magnitudes)
void decode_amin_star(
    const ChinnMatrix& chinn, uint32_t P_punctured, uint32_t max_iter, bool use_exact_correction,
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

            // Approximate-Min* two outgoing magnitudes:
            // For all d != min_idx: remaining minimum is min1, second minimum is min2 -> delta1 = min2 - min1
            // For d == min_idx:     remaining minimum is min2, second minimum is min3 -> delta2 = min3 - min2
            float delta1 = min2 - min1;
            float delta2 = (actual_deg > 2) ? (min3 - min2) : delta1;

            float corr1 = use_exact_correction ? exact_jacobian_correction(delta1) : jones_approx_correction(delta1);
            float corr2 = use_exact_correction ? exact_jacobian_correction(delta2) : jones_approx_correction(delta2);

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

int main(int argc, char** argv) {
    std::string matrix_file = "matrices/AR4JA_r45_4c_128c_r12.chinn.out";
    ChinnMatrix chinn = parse_chinn_out(matrix_file);
    uint32_t P_punctured = 512;
    uint32_t unpunctured = chinn.N - P_punctured;
    double rate = static_cast<double>(chinn.N - chinn.M) / unpunctured;

    double eb_n0_db = 1.6; // Waterfall region where difference is prominent
    if (argc > 1) eb_n0_db = std::stod(argv[1]);

    uint32_t num_codewords = 10000;
    if (argc > 2) num_codewords = std::stoul(argv[2]);

    std::cout << "=================================================================" << std::endl;
    std::cout << "   AR4JA Rate-1/2 Decoding Benchmark: NMS vs Approximate-Min*   " << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << "Eb/N0: " << eb_n0_db << " dB | Codewords: " << num_codewords << std::endl;
    std::cout << "Code: N=" << chinn.N << ", M=" << chinn.M << ", P=" << P_punctured << " (Rate = " << rate << ")" << std::endl;

    double snr_lin = std::pow(10.0, eb_n0_db / 10.0);
    float mu_llr = static_cast<float>(4.0 * rate * snr_lin);
    float sigma_llr = static_cast<float>(std::sqrt(8.0 * rate * snr_lin));

    // Pre-generate identical AWGN channel realizations for all algorithms
    std::vector<std::vector<float>> initial_channel(num_codewords, std::vector<float>(chinn.N, 0.0f));
    Xoshiro128Plus rng;
    rng.init(133742, 0x9E3779B9);
    for (uint32_t cw = 0; cw < num_codewords; cw++) {
        for (uint32_t i = 0; i < unpunctured; i += 2) {
            generate_awgn_floats(rng, mu_llr, sigma_llr, initial_channel[cw][i], initial_channel[cw][i + 1]);
        }
        for (uint32_t i = unpunctured; i < chinn.N; i++) initial_channel[cw][i] = 0.0f;
    }

    std::vector<float> channel_llrs(chinn.N);
    static float r_msg[MAX_M][MAX_DEG];

    // 1. Normalized Min-Sum (alpha = 0.75)
    uint32_t nms75_bits = 0, nms75_frames = 0, nms75_total_iters = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (uint32_t cw = 0; cw < num_codewords; cw++) {
        channel_llrs = initial_channel[cw];
        std::memset(r_msg, 0, sizeof(r_msg));
        uint32_t b = 0, f = 0, it = 0;
        decode_nms(chinn, P_punctured, 16, 0.75f, channel_llrs.data(), r_msg, b, f, it);
        nms75_bits += b; nms75_frames += f; nms75_total_iters += it;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double nms75_sec = std::chrono::duration<double>(t1 - t0).count();

    // 2. Normalized Min-Sum (alpha = 0.80)
    uint32_t nms80_bits = 0, nms80_frames = 0, nms80_total_iters = 0;
    auto t2 = std::chrono::high_resolution_clock::now();
    for (uint32_t cw = 0; cw < num_codewords; cw++) {
        channel_llrs = initial_channel[cw];
        std::memset(r_msg, 0, sizeof(r_msg));
        uint32_t b = 0, f = 0, it = 0;
        decode_nms(chinn, P_punctured, 16, 0.80f, channel_llrs.data(), r_msg, b, f, it);
        nms80_bits += b; nms80_frames += f; nms80_total_iters += it;
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double nms80_sec = std::chrono::duration<double>(t3 - t2).count();

    // 3. Christopher Jones Approximate-Min* (Piecewise Linear / Shift-Add friendly)
    uint32_t amin_bits = 0, amin_frames = 0, amin_total_iters = 0;
    auto t4 = std::chrono::high_resolution_clock::now();
    for (uint32_t cw = 0; cw < num_codewords; cw++) {
        channel_llrs = initial_channel[cw];
        std::memset(r_msg, 0, sizeof(r_msg));
        uint32_t b = 0, f = 0, it = 0;
        decode_amin_star(chinn, P_punctured, 16, false, channel_llrs.data(), r_msg, b, f, it);
        amin_bits += b; amin_frames += f; amin_total_iters += it;
    }
    auto t5 = std::chrono::high_resolution_clock::now();
    double amin_sec = std::chrono::duration<double>(t5 - t4).count();

    // 4. Approximate-Min* (Exact Log-Jacobian Correction)
    uint32_t amin_exact_bits = 0, amin_exact_frames = 0, amin_exact_total_iters = 0;
    auto t6 = std::chrono::high_resolution_clock::now();
    for (uint32_t cw = 0; cw < num_codewords; cw++) {
        channel_llrs = initial_channel[cw];
        std::memset(r_msg, 0, sizeof(r_msg));
        uint32_t b = 0, f = 0, it = 0;
        decode_amin_star(chinn, P_punctured, 16, true, channel_llrs.data(), r_msg, b, f, it);
        amin_exact_bits += b; amin_exact_frames += f; amin_exact_total_iters += it;
    }
    auto t7 = std::chrono::high_resolution_clock::now();
    double amin_exact_sec = std::chrono::duration<double>(t7 - t6).count();

    std::cout << "\n-------------------------------------------------------------------------------------" << std::endl;
    std::cout << std::left << std::setw(28) << "Algorithm" 
              << std::setw(12) << "Bit Errors" 
              << std::setw(14) << "Frame Errors" 
              << std::setw(14) << "FER" 
              << std::setw(12) << "Avg Iters"
              << std::setw(10) << "Time (s)" << std::endl;
    std::cout << "-------------------------------------------------------------------------------------" << std::endl;

    std::cout << std::left << std::setw(28) << "Normalized Min-Sum (a=0.75)" 
              << std::setw(12) << nms75_bits 
              << std::setw(14) << nms75_frames 
              << std::setw(14) << (double)nms75_frames / num_codewords 
              << std::setw(12) << std::fixed << std::setprecision(2) << (double)nms75_total_iters / num_codewords 
              << std::setw(10) << std::setprecision(3) << nms75_sec << std::endl;

    std::cout << std::left << std::setw(28) << "Normalized Min-Sum (a=0.80)" 
              << std::setw(12) << nms80_bits 
              << std::setw(14) << nms80_frames 
              << std::setw(14) << (double)nms80_frames / num_codewords 
              << std::setw(12) << std::fixed << std::setprecision(2) << (double)nms80_total_iters / num_codewords 
              << std::setw(10) << std::setprecision(3) << nms80_sec << std::endl;

    std::cout << std::left << std::setw(28) << "Approx-Min* (Jones PWL)" 
              << std::setw(12) << amin_bits 
              << std::setw(14) << amin_frames 
              << std::setw(14) << (double)amin_frames / num_codewords 
              << std::setw(12) << std::fixed << std::setprecision(2) << (double)amin_total_iters / num_codewords 
              << std::setw(10) << std::setprecision(3) << amin_sec << std::endl;

    std::cout << std::left << std::setw(28) << "Approx-Min* (Exact Jacobi)" 
              << std::setw(12) << amin_exact_bits 
              << std::setw(14) << amin_exact_frames 
              << std::setw(14) << (double)amin_exact_frames / num_codewords 
              << std::setw(12) << std::fixed << std::setprecision(2) << (double)amin_exact_total_iters / num_codewords 
              << std::setw(10) << std::setprecision(3) << amin_exact_sec << std::endl;

    std::cout << "-------------------------------------------------------------------------------------" << std::endl;

    double fer_reduction = (1.0 - (double)amin_frames / nms75_frames) * 100.0;
    std::cout << "\n>>> Approximate-Min* (Jones PWL) Frame Error Reduction vs NMS (a=0.75): " 
              << std::fixed << std::setprecision(1) << fer_reduction << "% reduction in frame errors!" << std::endl;

    return 0;
}
