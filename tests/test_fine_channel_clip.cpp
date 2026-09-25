#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
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
    void init(uint32_t seed_lo, uint32_t seed_hi) {
        auto splitmix = [](uint32_t& x) {
            uint32_t z = (x += 0x9E3779B9);
            z ^= z >> 16; z *= 0x85EBCA6B;
            z ^= z >> 13; z *= 0xC2B2AE35;
            z ^= z >> 16; return z;
        };
        uint32_t x1 = seed_lo, x2 = seed_hi;
        s[0] = splitmix(x1); s[1] = splitmix(x1);
        s[2] = splitmix(x2); s[3] = splitmix(x2);
        if ((s[0] | s[1] | s[2] | s[3]) == 0) s[0] = 1;
    }
};

inline void polar_box_muller(Xoshiro128Plus& rng, float mu, float sigma, float& out0, float& out1) {
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
    out0 = mu + sigma * (v1 * factor);
    out1 = mu + sigma * (v2 * factor);
}

struct bf16 {
    uint16_t v;
    bf16() : v(0) {}
    explicit bf16(float f) {
        uint32_t u;
        std::memcpy(&u, &f, sizeof(u));
        uint32_t lsb = (u >> 16) & 1;
        u += 0x7FFF + lsb;
        v = static_cast<uint16_t>(u >> 16);
    }
    explicit operator float() const {
        uint32_t u = static_cast<uint32_t>(v) << 16;
        float f;
        std::memcpy(&f, &u, sizeof(f));
        return f;
    }
};

inline float jones_approx_correction(float delta) {
    if (delta < 0.5f) {
        return 0.69315f - 0.40f * delta;
    } else if (delta < 2.0f) {
        return 0.49315f - 0.328f * (delta - 0.5f);
    } else {
        return 0.0f;
    }
}

void decode_amin(
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
                
                float q = static_cast<float>(channel_llrs[vn]) - static_cast<float>(r_msg[m][d]);
                q_val[d] = q;

                uint32_t sign = std::signbit(q) ? 1 : 0;
                global_sign ^= sign;
                float abs_q = std::abs(q);

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

                float new_llr = q_val[d] + r_new;
                channel_llrs[vn] = bf16(new_llr);
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
    if (argc > 1) matrix_file = argv[1];

    ChinnMatrix chinn = parse_chinn_out(matrix_file);
    uint32_t P_punctured = 512;
    uint32_t unpunctured = chinn.N - P_punctured;
    double rate = static_cast<double>(chinn.N - chinn.M) / unpunctured;

    double eb_n0_db = 2.0;
    double snr_lin = std::pow(10.0, eb_n0_db / 10.0);
    float mu_llr = static_cast<float>(4.0 * rate * snr_lin);
    float sigma_llr = static_cast<float>(std::sqrt(8.0 * rate * snr_lin));

    uint32_t num_codewords = 50000;
    std::cout << "========================================================================================\n";
    std::cout << "  Fine-Grained Channel LLR Clipping Sweep: Eb/N0 = " << eb_n0_db << " dB (" << num_codewords << " codewords)\n";
    std::cout << "========================================================================================\n";
    std::cout << std::left << std::setw(24) << "Configuration"
              << " | " << std::setw(10) << "Bit Errors"
              << " | " << std::setw(12) << "Frame Errors"
              << " | " << std::setw(12) << "BER"
              << " | " << std::setw(12) << "FER"
              << " | " << std::setw(10) << "Avg Iters"
              << " | " << std::setw(10) << "Delta FE" << "\n";
    std::cout << std::string(100, '-') << "\n";

    struct Cfg { std::string name; float l_ch_max; };
    std::vector<Cfg> cfgs = {
        {"Baseline (Unclipped)", 0.0f},
        {"L_ch = 6.0",          6.0f},
        {"L_ch = 5.0",          5.0f},
        {"L_ch = 4.5",          4.5f},
        {"L_ch = 4.2",          4.2f},
        {"L_ch = 4.0",          4.0f},
        {"L_ch = 3.8",          3.8f},
        {"L_ch = 3.6",          3.6f},
        {"L_ch = 3.4",          3.4f}
    };

    std::vector<bf16> channel_llrs(chinn.N);
    static bf16 r_msg[MAX_M][MAX_DEG];
    uint64_t baseline_fe = 0;

    for (size_t idx = 0; idx < cfgs.size(); idx++) {
        const auto& c = cfgs[idx];
        Xoshiro128Plus rng;
        rng.init(1337, 0x9E3779B9);

        uint64_t total_be = 0;
        uint64_t total_fe = 0;
        uint64_t total_iters = 0;

        for (uint32_t cw = 0; cw < num_codewords; cw++) {
            for (uint32_t i = 0; i < unpunctured; i += 2) {
                float l0, l1;
                polar_box_muller(rng, mu_llr, sigma_llr, l0, l1);

                if (c.l_ch_max > 0.0f) {
                    if (l0 > c.l_ch_max) l0 = c.l_ch_max;
                    else if (l0 < -c.l_ch_max) l0 = -c.l_ch_max;
                    if (l1 > c.l_ch_max) l1 = c.l_ch_max;
                    else if (l1 < -c.l_ch_max) l1 = -c.l_ch_max;
                }

                channel_llrs[i] = bf16(l0);
                channel_llrs[i + 1] = bf16(l1);
            }
            for (uint32_t i = unpunctured; i < chinn.N; i++) {
                channel_llrs[i] = bf16(0.0f);
            }
            std::memset(r_msg, 0, sizeof(r_msg));

            uint32_t be = 0, fe = 0, iters = 0;
            decode_amin(
                chinn, P_punctured, 16,
                channel_llrs.data(), r_msg, be, fe, iters
            );
            total_be += be;
            total_fe += fe;
            total_iters += iters;
        }

        if (idx == 0) baseline_fe = total_fe;

        double ber = static_cast<double>(total_be) / (static_cast<double>(num_codewords) * unpunctured);
        double fer = static_cast<double>(total_fe) / num_codewords;
        double avg_iter = static_cast<double>(total_iters) / num_codewords;
        int64_t delta_fe = static_cast<int64_t>(total_fe) - static_cast<int64_t>(baseline_fe);

        std::cout << std::left << std::setw(24) << c.name
                  << " | " << std::setw(10) << total_be
                  << " | " << std::setw(12) << total_fe
                  << " | " << std::scientific << std::setprecision(4) << std::setw(12) << ber
                  << " | " << std::scientific << std::setprecision(4) << std::setw(12) << fer
                  << " | " << std::fixed << std::setprecision(2) << std::setw(10) << avg_iter
                  << " | " << std::showpos << std::setw(9) << delta_fe << std::noshowpos << "\n";
    }
    std::cout << "========================================================================================\n";
    return 0;
}
