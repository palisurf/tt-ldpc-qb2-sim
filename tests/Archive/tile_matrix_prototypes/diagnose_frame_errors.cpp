#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "../engines/tile_matrix_4cw/ldpc_tile_matrix_core.h"

// Compare Natural Schedule vs Tiled Schedule on the failing batch
struct ParityCheckMatrix {
    uint32_t M = 0;
    uint32_t N = 0;
    uint32_t K = 0;
    uint32_t P = 0;
    std::vector<std::vector<uint32_t>> check_to_var;
};

bool parse_chinn_out(const std::string& filepath, ParityCheckMatrix& pcm) {
    std::ifstream infile(filepath);
    if (!infile.is_open()) return false;
    struct RawEdge { uint32_t r, c; float val; };
    std::vector<RawEdge> raw_lines;
    uint32_t r, c; float val;
    while (infile >> r >> c >> val) raw_lines.push_back({r, c, val});
    if (raw_lines.empty()) return false;
    const auto& dim_line = raw_lines.back();
    pcm.M = dim_line.r;
    pcm.N = dim_line.c;
    pcm.K = pcm.N - pcm.M;
    pcm.P = pcm.M / 3;
    pcm.check_to_var.resize(pcm.M);
    for (size_t i = 0; i + 1 < raw_lines.size(); ++i) {
        if (std::abs(raw_lines[i].val) > 1e-6f) {
            pcm.check_to_var[raw_lines[i].r - 1].push_back(raw_lines[i].c - 1);
        }
    }
    return true;
}

struct TileDesc {
    std::vector<uint32_t> check_indices;
    uint8_t  check_degree;
    uint8_t  num_checks;
    uint16_t num_edges;
};

static void partition_checks(
    const std::vector<uint32_t>& check_list,
    const ParityCheckMatrix& pcm,
    uint32_t target_tile_count,
    uint32_t max_tile_size,
    std::vector<std::vector<uint32_t>>& out_tiles
) {
    std::unordered_map<uint32_t, std::vector<uint32_t>> var_to_c;
    for (uint32_t c : check_list) {
        for (uint32_t v : pcm.check_to_var[c]) var_to_c[v].push_back(c);
    }
    std::vector<std::pair<uint32_t, uint32_t>> check_conflicts;
    for (uint32_t c : check_list) {
        std::unordered_set<uint32_t> conflicts;
        for (uint32_t v : pcm.check_to_var[c]) {
            for (uint32_t other : var_to_c[v]) {
                if (other != c) conflicts.insert(other);
            }
        }
        check_conflicts.push_back({c, static_cast<uint32_t>(conflicts.size())});
    }
    std::sort(check_conflicts.begin(), check_conflicts.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    out_tiles.assign(target_tile_count, std::vector<uint32_t>());
    std::vector<std::unordered_set<uint32_t>> tile_vars(target_tile_count);
    for (const auto& item : check_conflicts) {
        uint32_t c = item.first;
        std::unordered_set<uint32_t> c_vars(pcm.check_to_var[c].begin(), pcm.check_to_var[c].end());
        int best_tile = -1;
        size_t min_size = max_tile_size + 1;
        for (uint32_t t = 0; t < target_tile_count; ++t) {
            if (out_tiles[t].size() < max_tile_size) {
                bool conflict = false;
                for (uint32_t v : c_vars) {
                    if (tile_vars[t].count(v)) { conflict = true; break; }
                }
                if (!conflict && out_tiles[t].size() < min_size) {
                    min_size = out_tiles[t].size();
                    best_tile = static_cast<int>(t);
                }
            }
        }
        out_tiles[best_tile].push_back(c);
        for (uint32_t v : c_vars) tile_vars[best_tile].insert(v);
    }
}

std::vector<TileDesc> generate_tile_schedule(const ParityCheckMatrix& pcm) {
    std::vector<uint32_t> deg6_checks, deg3_checks;
    for (uint32_t m = 0; m < pcm.M; ++m) {
        if (pcm.check_to_var[m].size() == 6) deg6_checks.push_back(m);
        else if (pcm.check_to_var[m].size() == 3) deg3_checks.push_back(m);
    }
    std::vector<std::vector<uint32_t>> deg6_tiles, deg3_tiles;
    partition_checks(deg6_checks, pcm, 26, CHECKS_PER_TILE_DEG6, deg6_tiles);
    partition_checks(deg3_checks, pcm, 7, CHECKS_PER_TILE_DEG3, deg3_tiles);
    std::vector<TileDesc> schedule;
    for (auto& t : deg6_tiles) {
        schedule.push_back({t, 6, static_cast<uint8_t>(t.size()), static_cast<uint16_t>(t.size() * 6)});
    }
    for (auto& t : deg3_tiles) {
        schedule.push_back({t, 3, static_cast<uint8_t>(t.size()), static_cast<uint16_t>(t.size() * 3)});
    }
    return schedule;
}

inline float approx_min_star(float a, float b) {
    float min_v = (a < b) ? a : b;
    float delta = std::fabs(a - b);
    float g = 0.0f;
    if (delta < 0.5f) g = 0.69315f - 0.40f * delta;
    else if (delta < 2.0f) g = 0.49315f - 0.328f * (delta - 0.5f);
    float res = min_v - g;
    return (res > 0.0f) ? res : 0.0f;
}

struct PRNG128 {
    uint32_t s[4];
    static inline uint32_t splitmix32(uint32_t& x) {
        uint32_t z = (x += 0x9e3779b9);
        z ^= z >> 16; z *= 0x21f0aaad; z ^= z >> 15; z *= 0x735a2d97; z ^= z >> 15;
        return z;
    }
    void init(uint32_t seed) {
        uint32_t sm = seed;
        s[0] = splitmix32(sm); s[1] = splitmix32(sm);
        s[2] = splitmix32(sm); s[3] = splitmix32(sm);
        if ((s[0] | s[1] | s[2] | s[3]) == 0) s[0] = 1;
    }
    inline uint32_t next() {
        const uint32_t result = s[0] + s[3];
        const uint32_t t = s[1] << 9;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t; s[3] = (s[3] << 11) | (s[3] >> 21);
        return result;
    }
    inline float next_uniform() {
        uint32_t val = next();
        if (val == 0) val = 1;
        return static_cast<float>(val) * 2.3283064365386963e-10f;
    }
};

bool check_syndrome(const ParityCheckMatrix& pcm, const std::vector<float>& vnodes) {
    for (uint32_t m = 0; m < pcm.M; ++m) {
        uint32_t parity = 0;
        for (uint32_t v : pcm.check_to_var[m]) {
            if (vnodes[v] < 0.0f) parity ^= 1;
        }
        if (parity != 0) return false;
    }
    return true;
}

// Decode single codeword with custom check ordering
bool decode_single_cw(
    const ParityCheckMatrix& pcm,
    const std::vector<uint32_t>& check_order,
    std::vector<float> vnodes, // by-value copy
    uint32_t max_iter,
    float r_max,
    uint32_t& out_iters,
    uint32_t& out_bit_errors
) {
    std::vector<std::vector<float>> r_msg(pcm.M);
    for (uint32_t m = 0; m < pcm.M; ++m) r_msg[m].resize(pcm.check_to_var[m].size(), 0.0f);

    for (uint32_t iter = 0; iter < max_iter; ++iter) {
        if (check_syndrome(pcm, vnodes)) {
            out_iters = iter;
            out_bit_errors = 0;
            return true;
        }

        for (uint32_t m : check_order) {
            uint32_t deg = pcm.check_to_var[m].size();
            const auto& vars = pcm.check_to_var[m];

            float q[8], signs[8], mags[8];
            float total_sign = 1.0f;
            uint32_t min_idx = 0;
            float min_mag = 99999.0f;

            for (uint32_t d = 0; d < deg; ++d) {
                uint32_t v = vars[d];
                float r_old = (iter == 0) ? 0.0f : r_msg[m][d];
                q[d] = vnodes[v] - r_old;
                signs[d] = (q[d] < 0.0f) ? -1.0f : 1.0f;
                mags[d] = std::fabs(q[d]);
                total_sign *= signs[d];
                if (mags[d] < min_mag) {
                    min_mag = mags[d];
                    min_idx = d;
                }
            }

            uint32_t first_idx = (min_idx == 0) ? 1 : 0;
            float m_others = mags[first_idx];
            for (uint32_t d = first_idx + 1; d < deg; ++d) {
                if (d == min_idx) continue;
                m_others = approx_min_star(m_others, mags[d]);
            }
            float m_all = approx_min_star(m_others, min_mag);

            for (uint32_t d = 0; d < deg; ++d) {
                uint32_t v = vars[d];
                float edge_sign = total_sign * signs[d];
                float edge_mag = (d == min_idx) ? m_others : m_all;
                if (r_max > 0.0f && edge_mag > r_max) edge_mag = r_max;
                float r_new = edge_sign * edge_mag;
                r_msg[m][d] = r_new;
                vnodes[v] = q[d] + r_new;
            }
        }

        uint32_t unsat = 0;
        for (uint32_t m = 0; m < pcm.M; ++m) {
            uint32_t p = 0;
            for (uint32_t v : pcm.check_to_var[m]) if (vnodes[v] < 0.0f) p ^= 1;
            if (p != 0) unsat++;
        }
        std::cout << "    [DIAG iter " << iter + 1 << "] unsat=" << unsat << ", v[1064]=" << vnodes[1064] << std::endl;
        if (unsat == 0) {
            out_iters = iter + 1;
            out_bit_errors = 0;
            return true;
        }
    }

    out_iters = max_iter;
    out_bit_errors = 0;
    for (uint32_t v = 0; v < pcm.K; ++v) {
        if (vnodes[v] < 0.0f) out_bit_errors++;
    }
    return false;
}

int main() {
    ParityCheckMatrix pcm;
    parse_chinn_out("matrices/AR4JA_r45_4c_128c_r12.chinn.out", pcm);

    // Natural schedule: 0, 1, 2, ..., 1535 (Scalar_5w order)
    std::vector<uint32_t> natural_order(pcm.M);
    for (uint32_t m = 0; m < pcm.M; ++m) natural_order[m] = m;

    // Deg6 then Deg3 schedule (Current tile partitioner order)
    std::vector<uint32_t> deg6_then_deg3;
    for (uint32_t m = 0; m < pcm.M; ++m) if (pcm.check_to_var[m].size() == 6) deg6_then_deg3.push_back(m);
    for (uint32_t m = 0; m < pcm.M; ++m) if (pcm.check_to_var[m].size() == 3) deg6_then_deg3.push_back(m);

    double eb_n0_db = 2.5;
    float code_rate = 0.5f;
    float eb_n0_lin = std::pow(10.0f, static_cast<float>(eb_n0_db) / 10.0f);
    float es_n0 = code_rate * eb_n0_lin;
    float noise_var = 1.0f / (2.0f * es_n0);
    float noise_std = std::sqrt(noise_var);
    float mu_llr = 2.0f / noise_var;
    float sigma_llr = 2.0f / noise_std;

    PRNG128 prng;
    prng.init(42 + 0 * 1000); // CW 0 PRNG stream

    // Advance to Batch 8
    for (int b = 0; b < 8; ++b) {
        for (int i = 0; i < 2048; ++i) prng.next_uniform();
    }

    // Generate Batch 23 CW 2 channel LLRs
    std::vector<float> init_vnodes(pcm.N, 0.0f);
    for (uint32_t v = 0; v < 2048; v += 2) {
        float u1 = prng.next_uniform();
        float u2 = prng.next_uniform();
        float r = std::sqrt(-2.0f * std::log(u1));
        float theta = 6.283185307f * u2;
        init_vnodes[v + 0] = mu_llr + sigma_llr * (r * std::cos(theta));
        init_vnodes[v + 1] = mu_llr + sigma_llr * (r * std::sin(theta));
    }
    for (uint32_t v = 2048; v < 2560; ++v) init_vnodes[v] = 0.0f;

    std::cout << "Testing Batch 23, CW 2 on different configurations:" << std::endl;
    
    // Config 1: Deg6-then-Deg3 (tile order), no r_max clip, max_iter = 50
    uint32_t iters = 0, bit_errs = 0;
    bool conv = decode_single_cw(pcm, deg6_then_deg3, init_vnodes, 50, 0.0f, iters, bit_errs);
    std::cout << "1. Deg6-then-Deg3, unclipped R, MaxIter=50: " << (conv ? "CONVERGED" : "FAILED") 
              << ", Iters=" << iters << ", BitErrors=" << bit_errs << std::endl;

    // Config 2: Natural Order (0..1535, scalar_5w order), no r_max clip, max_iter = 50
    conv = decode_single_cw(pcm, natural_order, init_vnodes, 50, 0.0f, iters, bit_errs);
    std::cout << "2. Natural Order (0..1535), unclipped R, MaxIter=50: " << (conv ? "CONVERGED" : "FAILED") 
              << ", Iters=" << iters << ", BitErrors=" << bit_errs << std::endl;

    // Config 3: Natural Order, R_max = 6.5 (scalar_5w with R clipping), max_iter = 50
    conv = decode_single_cw(pcm, natural_order, init_vnodes, 50, 6.5f, iters, bit_errs);
    std::cout << "3. Natural Order (0..1535), R_max=6.5, MaxIter=50: " << (conv ? "CONVERGED" : "FAILED") 
              << ", Iters=" << iters << ", BitErrors=" << bit_errs << std::endl;

    // Config 4: Deg6-then-Deg3, R_max = 6.5, max_iter = 50
    conv = decode_single_cw(pcm, deg6_then_deg3, init_vnodes, 50, 6.5f, iters, bit_errs);
    std::cout << "4. Deg6-then-Deg3, R_max=6.5, MaxIter=50: " << (conv ? "CONVERGED" : "FAILED") 
              << ", Iters=" << iters << ", BitErrors=" << bit_errs << std::endl;

    // Config 5: Partitioned Tiles Order (exactly from test_end_to_end_4cw_decoder.cpp)
    auto schedule = generate_tile_schedule(pcm);
    std::vector<uint32_t> partitioned_order;
    for (const auto& t : schedule) {
        for (uint32_t c : t.check_indices) partitioned_order.push_back(c);
    }
    conv = decode_single_cw(pcm, partitioned_order, init_vnodes, 50, 0.0f, iters, bit_errs);
    std::cout << "5. Partitioned Tiles Order, unclipped R, MaxIter=50: " << (conv ? "CONVERGED" : "FAILED") 
              << ", Iters=" << iters << ", BitErrors=" << bit_errs << std::endl;

    conv = decode_single_cw(pcm, partitioned_order, init_vnodes, 50, 6.5f, iters, bit_errs);
    std::cout << "6. Partitioned Tiles Order, R_max=6.5, MaxIter=50: " << (conv ? "CONVERGED" : "FAILED") 
              << ", Iters=" << iters << ", BitErrors=" << bit_errs << std::endl;

    return 0;
}
