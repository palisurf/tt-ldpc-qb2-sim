#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <iomanip>
#include <chrono>
#include <random>
#include <fstream>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <cstring>
#include <atomic>
#include <thread>
#include <omp.h>

#include "../engines/tile_matrix_4cw/ldpc_tile_matrix_core.h"

// ==============================================================================
// Step 1 Algorithmic Validation: End-to-End 4-Codeword Tile Decoder Simulation
// Architecture: Decoupled Access-Execute (DAE) with 33 Snapped Tiles & CW Freezing
// Code: CCSDS AR4JA Rate-1/2 (K=1024, N=2560, M=1536, P=512)
// ==============================================================================

struct ParityCheckMatrix {
    uint32_t M = 0;
    uint32_t N = 0;
    uint32_t K = 0;
    uint32_t P = 0;
    uint32_t circulant_p = 0;
    std::vector<std::vector<uint32_t>> check_to_var;
    std::vector<std::vector<uint32_t>> var_to_check;
};

struct TileDesc {
    std::vector<uint32_t> check_indices;
    uint8_t  check_degree;
    uint8_t  num_checks;
    uint16_t num_edges;
};

// ------------------------------------------------------------------------------
// Matrix Parser
// ------------------------------------------------------------------------------
bool parse_chinn_out(const std::string& filepath, ParityCheckMatrix& pcm) {
    std::ifstream infile(filepath);
    if (!infile.is_open()) {
        std::cerr << "[ERROR] Cannot open matrix: " << filepath << std::endl;
        return false;
    }

    struct RawEdge { uint32_t r, c; float val; };
    std::vector<RawEdge> raw_lines;
    uint32_t r, c; float val;
    while (infile >> r >> c >> val) {
        raw_lines.push_back({r, c, val});
    }

    if (raw_lines.empty()) return false;

    const auto& dim_line = raw_lines.back();
    pcm.M = dim_line.r;
    pcm.N = dim_line.c;
    pcm.K = pcm.N - pcm.M;
    pcm.circulant_p = pcm.M / 3;
    pcm.P = pcm.circulant_p;

    pcm.check_to_var.resize(pcm.M);
    pcm.var_to_check.resize(pcm.N);

    for (size_t i = 0; i + 1 < raw_lines.size(); ++i) {
        const auto& edge = raw_lines[i];
        if (std::abs(edge.val) > 1e-6f) {
            uint32_t m_idx = edge.r - 1;
            uint32_t v_idx = edge.c - 1;
            pcm.check_to_var[m_idx].push_back(v_idx);
            pcm.var_to_check[v_idx].push_back(m_idx);
        }
    }

    return true;
}

// ------------------------------------------------------------------------------
// Balanced Graph Partitioner for 33 Collision-Free Tiles
// ------------------------------------------------------------------------------
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
        assert(best_tile != -1);
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

    // Proportionally interleave the 26 degree-6 tiles and 7 degree-3 tiles (Total: 33 tiles)
    // Cadence: ~3.71 deg-6 tiles per deg-3 tile (3 or 4 deg-6 tiles followed by 1 deg-3 tile)
    std::vector<TileDesc> schedule;
    schedule.reserve(33);

    size_t i6 = 0, i3 = 0;
    while (i6 < deg6_tiles.size() || i3 < deg3_tiles.size()) {
        size_t quota_i6 = (i3 + 1 < deg3_tiles.size()) 
                          ? ((i3 + 1) * deg6_tiles.size() / deg3_tiles.size()) 
                          : deg6_tiles.size();
        while (i6 < quota_i6 && i6 < deg6_tiles.size()) {
            auto& t = deg6_tiles[i6++];
            schedule.push_back({t, 6, static_cast<uint8_t>(t.size()), static_cast<uint16_t>(t.size() * 6)});
        }
        if (i3 < deg3_tiles.size()) {
            auto& t = deg3_tiles[i3++];
            schedule.push_back({t, 3, static_cast<uint8_t>(t.size()), static_cast<uint16_t>(t.size() * 3)});
        }
    }
    return schedule;
}

// ------------------------------------------------------------------------------
// Dr. Jones' 2-Region Approximate-Min* Binary Operator
// ------------------------------------------------------------------------------
inline float approx_min_star(float a, float b) {
    float min_v = (a < b) ? a : b;
    float delta = std::fabs(a - b);
    float g = 0.0f;
    if (delta < 0.5f) {
        g = 0.69315f - 0.40f * delta;
    } else if (delta < 2.0f) {
        g = 0.49315f - 0.328f * (delta - 0.5f);
    }
    float res = min_v - g;
    return (res > 0.0f) ? res : 0.0f;
}

// ------------------------------------------------------------------------------
// Vector Cartesian Box-Muller Generator for 4 Parallel Codewords
// ------------------------------------------------------------------------------
constexpr float TWO_PI = 6.28318530717958647692f;
constexpr float INT32_TO_UNIT = 2.3283064365386963e-10f;

struct PRNG128 {
    uint32_t s[4];
    void init(uint32_t seed) {
        uint32_t z = seed;
        for (int i = 0; i < 4; ++i) {
            z += 0x9e3779b9;
            z = (z ^ (z >> 16)) * 0x85ebca6b;
            z = (z ^ (z >> 13)) * 0xc2b2ae35;
            s[i] = z ^ (z >> 16);
            if (s[i] == 0) s[i] = 1;
        }
    }
    inline uint32_t next() {
        const uint32_t result = s[0] + s[3];
        const uint32_t t = s[1] << 9;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t;
        s[3] = (s[3] << 11) | (s[3] >> 21);
        return result;
    }
    inline float next_uniform() {
        uint32_t val = next();
        if (val == 0) val = 1;
        return static_cast<float>(val) * INT32_TO_UNIT;
    }
};

void generate_channel_llrs(
    PRNG128 prng[4],
    float mu_llr,
    float sigma_llr,
    Packed4CW_BF16* vnodes
) {
    // 1. Transmitted nodes (0..2047) -> BPSK + AWGN
    for (int cw = 0; cw < 4; ++cw) {
        for (uint32_t v = 0; v < 2048; v += 2) {
            float u1 = prng[cw].next_uniform();
            float u2 = prng[cw].next_uniform();
            float r = std::sqrt(-2.0f * std::log(u1));
            float theta = TWO_PI * u2;
            float z0 = r * std::cos(theta);
            float z1 = r * std::sin(theta);

            float llr0 = mu_llr + sigma_llr * z0;
            float llr1 = mu_llr + sigma_llr * z1;

            vnodes[v + 0].cw[cw] = fp32_to_bf16(llr0);
            if (v + 1 < 2048) {
                vnodes[v + 1].cw[cw] = fp32_to_bf16(llr1);
            }
        }
    }

    // 2. Punctured nodes (2048..2559) -> strictly 0.0 LLR
    for (uint32_t v = 2048; v < 2560; ++v) {
        vnodes[v].set_all(fp32_to_bf16(0.0f));
    }
}

// ------------------------------------------------------------------------------
// Complete End-to-End Decoder for a 4-Codeword Batch
// ------------------------------------------------------------------------------
struct BatchResult {
    uint32_t iterations_taken[4];
    bool converged[4];
    uint32_t bit_errors[4];
    uint32_t batch_iterations;
};

BatchResult decode_4cw_batch(
    const ParityCheckMatrix& pcm,
    const std::vector<TileDesc>& schedule,
    Packed4CW_BF16* vnodes,
    std::vector<std::vector<Packed4CW_BF16>>& r_messages, // r_messages[check_idx][edge_idx]
    uint32_t max_iter,
    float post_clip_max,
    uint32_t batch_idx
) {
    BatchResult res{};
    for (int c = 0; c < 4; ++c) {
        res.iterations_taken[c] = max_iter;
        res.converged[c] = false;
        res.bit_errors[c] = 0;
    }

    uint8_t active_mask = 0x0F; // All 4 codewords active initially

    // Check initial channel LLR stats for CW 2
    int init_neg[4] = {0, 0, 0, 0};
    for (int c = 0; c < 4; ++c) {
        for (uint32_t v = 0; v < 2048; ++v) {
            if (bf16_to_fp32(vnodes[v].cw[c]) < 0.0f) init_neg[c]++;
        }
    }

    for (uint32_t iter = 0; iter < max_iter; ++iter) {
        if (active_mask == 0) {
            // All 4 codewords have converged! Early termination
            res.batch_iterations = iter;
            return res;
        }

        // Execute each of the 33 scheduled tiles in row-layered order
        for (const auto& tile : schedule) {
            uint32_t deg = tile.check_degree;

            for (uint32_t m : tile.check_indices) {
                // Incoming variable connections for check m
                const auto& vars = pcm.check_to_var[m];

                // Process each active codeword in SIMD lockstep
                for (int cw = 0; cw < 4; ++cw) {
                    if (!(active_mask & (1 << cw))) continue; // Frozen codeword!

                    float q[8];
                    float signs[8];
                    float mags[8];
                    float total_sign = 1.0f;
                    uint32_t min_idx = 0;
                    float min_mag = 99999.0f;

                    // 1. Extrinsic extraction: Q = L - R_old
                    for (uint32_t d = 0; d < deg; ++d) {
                        uint32_t v = vars[d];
                        float l_val = bf16_to_fp32(vnodes[v].cw[cw]);

                        // Iteration-1 Virtual Zero Bypass: R_old is zero
                        float r_old = (iter == 0) ? 0.0f : bf16_to_fp32(r_messages[m][d].cw[cw]);
                        q[d] = l_val - r_old;

                        signs[d] = (q[d] < 0.0f) ? -1.0f : 1.0f;
                        mags[d] = std::fabs(q[d]);
                        total_sign *= signs[d];
                        if (mags[d] < min_mag) {
                            min_mag = mags[d];
                            min_idx = d;
                        }
                    }

                    // 2. True Recursive Approximate-Min* Check Update
                    uint32_t first_idx = (min_idx == 0) ? 1 : 0;
                    float m_others = mags[first_idx];
                    for (uint32_t d = first_idx + 1; d < deg; ++d) {
                        if (d == min_idx) continue;
                        m_others = approx_min_star(m_others, mags[d]);
                    }

                    float m_all = approx_min_star(m_others, min_mag);

                    // 3. Posterior Accumulation & In-Place Scatter: L_new = Q + R_new
                    for (uint32_t d = 0; d < deg; ++d) {
                        uint32_t v = vars[d];
                        float edge_sign = total_sign * signs[d];
                        float edge_mag = (d == min_idx) ? m_others : m_all;
                        float r_new = edge_sign * edge_mag;

                        // Save R_new in check message array
                        r_messages[m][d].cw[cw] = fp32_to_bf16(r_new);

                        // Accumulate posterior
                        float l_new = q[d] + r_new;

                        // Post-clipping (only active if post_clip_max > 0.0f)
                        if (post_clip_max > 0.0f) {
                            if (l_new > post_clip_max) l_new = post_clip_max;
                            if (l_new < -post_clip_max) l_new = -post_clip_max;
                        }

                        vnodes[v].cw[cw] = fp32_to_bf16(l_new);
                    }
                }
            }
        }

        // Parity Check & Early Termination Evaluation after each iteration
        for (int cw = 0; cw < 4; ++cw) {
            if (!(active_mask & (1 << cw))) continue;

            uint32_t unsatisfied_count = 0;
            int single_unsatisfied_m = -1;
            for (uint32_t m = 0; m < pcm.M; ++m) {
                uint32_t parity = 0;
                for (uint32_t v : pcm.check_to_var[m]) {
                    float l_val = bf16_to_fp32(vnodes[v].cw[cw]);
                    if (l_val < 0.0f) parity ^= 1; // Hard decision: 1 if LLR < 0
                }
                if (parity != 0) {
                    unsatisfied_count++;
                    single_unsatisfied_m = m;
                }
            }

            if (unsatisfied_count == 0) {
                res.converged[cw] = true;
                res.iterations_taken[cw] = iter + 1;
                // FREEZE codeword cw: clear bit from active_mask
                active_mask &= ~(1 << cw);
            }
        }
    }

    res.batch_iterations = max_iter;

    // Evaluate residual bit errors on information bits (v in [0, K-1])
    for (int cw = 0; cw < 4; ++cw) {
        for (uint32_t v = 0; v < pcm.K; ++v) {
            float l_val = bf16_to_fp32(vnodes[v].cw[cw]);
            if (l_val < 0.0f) {
                res.bit_errors[cw]++;
            }
        }
    }

    return res;
}

// ------------------------------------------------------------------------------
// Main Simulation Harness
// ------------------------------------------------------------------------------
// Main Multi-Threaded Simulation Harness (OpenMP Parallel)
// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------
// Main Multi-Threaded Simulation Harness (OpenMP Parallel Waterfall Sweep)
// ------------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string matrix_path = "matrices/AR4JA_r45_4c_128c_r12.chinn.out";
    double snr_start = 1.0, snr_stop = 2.5, snr_step = 0.1;
    uint64_t max_blocks = 10000000;      // 10 Million codewords per SNR
    uint32_t min_frame_errors = 25;      // Target frame errors for statistical confidence
    uint32_t max_iter = 200;             // Max iterations per codeword
    int requested_threads = 0;           // 0 = auto-detect all available cores
    float post_clip = 0.0f;              // 0.0 = unclipped, matching baseline

    if (argc > 1) {
        std::string s = argv[1];
        auto c1 = s.find(':');
        auto c2 = s.rfind(':');
        if (c1 != std::string::npos && c2 != std::string::npos && c1 != c2) {
            snr_start = std::stod(s.substr(0, c1));
            snr_stop  = std::stod(s.substr(c1 + 1, c2 - c1 - 1));
            snr_step  = std::stod(s.substr(c2 + 1));
        } else {
            snr_start = snr_stop = std::stod(s);
        }
    }
    if (argc > 2) max_blocks = std::stoull(argv[2]);
    if (argc > 3) min_frame_errors = std::stoul(argv[3]);
    if (argc > 4) max_iter = std::stoul(argv[4]);
    if (argc > 5) requested_threads = std::stoi(argv[5]);
    if (argc > 6) post_clip = std::stof(argv[6]);

    int num_threads = (requested_threads > 0) ? requested_threads : omp_get_max_threads();
    omp_set_num_threads(num_threads);

    std::cout << "=================================================================" << std::endl;
    std::cout << " End-to-End 4-Codeword Tile Decoder: Multi-Threaded Waterfall    " << std::endl;
    std::cout << " Code   : CCSDS AR4JA Rate-1/2 (K=1024, N=2560, M=1536)          " << std::endl;
    std::cout << " Threads: " << num_threads << " OpenMP parallel worker threads" << std::endl;
    std::cout << " Range  : [" << std::fixed << std::setprecision(2) << snr_start << " : " << snr_stop << " : " << snr_step << "] dB" << std::endl;
    std::cout << " Target : " << max_blocks << " blocks or " << min_frame_errors << " frame errors per SNR" << std::endl;
    std::cout << " MaxIter: " << max_iter << std::endl;
    std::cout << "=================================================================" << std::endl;

    ParityCheckMatrix pcm;
    if (!parse_chinn_out(matrix_path, pcm)) return 1;

    auto schedule = generate_tile_schedule(pcm);
    std::cout << "[INFO] Scheduled " << schedule.size() << " collision-free tiles per iteration." << std::endl;

    std::string csv_path = "Results/results_e2e_4cw_ar4ja_r12_waterfall_1.0_2.5dB.csv";
    std::ofstream csv_file(csv_path, std::ios::app);
    if (csv_file.tellp() == 0) {
        csv_file << "EbN0_dB,Total_Blocks,Total_Bit_Errors,Total_Block_Errors,BER,FER,Mean_Iterations,Throughput_Msps,Host_Wall_s\n";
        csv_file.flush();
    }

    for (double eb_n0_db = snr_start; eb_n0_db <= snr_stop + 1e-4; eb_n0_db += snr_step) {
        float code_rate = static_cast<float>(pcm.K) / (pcm.N - pcm.P); // 1024 / 2048 = 0.5
        float eb_n0_lin = std::pow(10.0f, static_cast<float>(eb_n0_db) / 10.0f);
        float es_n0 = code_rate * eb_n0_lin;
        float noise_var = 1.0f / (2.0f * es_n0);
        float noise_std = std::sqrt(noise_var);
        float mu_llr = 2.0f / noise_var;
        float sigma_llr = 2.0f / noise_std;

        std::atomic<uint64_t> total_codewords{0};
        std::atomic<uint64_t> total_frame_errors{0};
        std::atomic<uint64_t> total_bit_errors{0};
        std::atomic<uint64_t> total_single_cw_iterations{0};
        std::atomic<uint64_t> total_batch_iterations{0};
        std::atomic<bool> stop_flag{false};

        auto t_start = std::chrono::high_resolution_clock::now();

        std::cout << "\n>>> Simulating Eb/N0 = " << std::fixed << std::setprecision(2) << eb_n0_db << " dB..." << std::endl;

        // Background progress monitor thread
        std::thread monitor([&]() {
            auto last_time = std::chrono::high_resolution_clock::now();
            uint64_t last_cws = 0;
            while (!stop_flag.load(std::memory_order_relaxed)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                auto now = std::chrono::high_resolution_clock::now();
                double dt = std::chrono::duration<double>(now - last_time).count();
                uint64_t cws = total_codewords.load(std::memory_order_relaxed);
                uint64_t fe = total_frame_errors.load(std::memory_order_relaxed);
                uint64_t be = total_bit_errors.load(std::memory_order_relaxed);
                uint64_t iters = total_single_cw_iterations.load(std::memory_order_relaxed);

                double fer = (cws > 0) ? static_cast<double>(fe) / cws : 0.0;
                double ber = (cws > 0) ? static_cast<double>(be) / (cws * pcm.K) : 0.0;
                double mean_iter = (cws > 0) ? static_cast<double>(iters) / cws : 0.0;
                double speed_cws = (dt > 0.0) ? (cws - last_cws) / dt : 0.0;
                double speed_msps = speed_cws * 2048.0 / 1e6;

                std::cout << "\r[" << std::fixed << std::setprecision(2) << eb_n0_db << " dB] Blocks: " 
                          << std::setw(8) << cws << "/" << max_blocks
                          << " | FE: " << std::setw(3) << fe << "/" << min_frame_errors
                          << " | BE: " << std::setw(6) << be
                          << " | FER: " << std::scientific << std::setprecision(3) << fer
                          << " | BER: " << ber
                          << " | Iter: " << std::fixed << std::setprecision(2) << mean_iter
                          << " | Speed: " << std::setprecision(1) << (speed_cws / 1000.0) << " kCW/s (" << speed_msps << " Msps)"
                          << std::flush;

                last_time = now;
                last_cws = cws;
            }
        });

        // Parallel Worker Threads
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            PRNG128 prng[4];
            uint32_t snr_seed_offset = static_cast<uint32_t>(eb_n0_db * 100.0) * 10007U;
            for (int c = 0; c < 4; ++c) {
                prng[c].init(1234567U + snr_seed_offset + tid * 1000003U + c * 31337U);
            }

            std::vector<Packed4CW_BF16> vnodes(pcm.N);
            std::vector<std::vector<Packed4CW_BF16>> r_messages(pcm.M);
            for (uint32_t m = 0; m < pcm.M; ++m) {
                r_messages[m].resize(pcm.check_to_var[m].size());
            }

            uint32_t chunk_batches = (eb_n0_db < 1.7) ? 25 : 250;
            uint32_t local_batch_id = 0;

            while (!stop_flag.load(std::memory_order_relaxed)) {
                uint32_t local_fe = 0;
                uint32_t local_be = 0;
                uint64_t local_cw_iters = 0;
                uint64_t local_batch_iters = 0;

                for (uint32_t b = 0; b < chunk_batches; ++b) {
                    generate_channel_llrs(prng, mu_llr, sigma_llr, vnodes.data());
                    BatchResult res = decode_4cw_batch(pcm, schedule, vnodes.data(), r_messages, max_iter, post_clip, local_batch_id++);

                    local_batch_iters += res.batch_iterations;
                    for (int c = 0; c < 4; ++c) {
                        local_cw_iters += res.iterations_taken[c];
                        if (res.bit_errors[c] > 0 || !res.converged[c]) {
                            local_fe++;
                            local_be += res.bit_errors[c];
                        }
                    }
                }

                uint64_t cur_cws = total_codewords.fetch_add(chunk_batches * 4, std::memory_order_relaxed) + chunk_batches * 4;
                uint64_t cur_fe = total_frame_errors.fetch_add(local_fe, std::memory_order_relaxed) + local_fe;
                total_bit_errors.fetch_add(local_be, std::memory_order_relaxed);
                total_single_cw_iterations.fetch_add(local_cw_iters, std::memory_order_relaxed);
                total_batch_iterations.fetch_add(local_batch_iters, std::memory_order_relaxed);

                if (cur_fe >= min_frame_errors || cur_cws >= max_blocks) {
                    stop_flag.store(true, std::memory_order_relaxed);
                }
            }
        }

        stop_flag.store(true);
        if (monitor.joinable()) monitor.join();

        auto t_end = std::chrono::high_resolution_clock::now();
        double total_sec = std::chrono::duration<double>(t_end - t_start).count();

        uint64_t final_cws = total_codewords.load();
        uint64_t final_fe = total_frame_errors.load();
        uint64_t final_be = total_bit_errors.load();
        uint64_t final_cw_iters = total_single_cw_iterations.load();
        uint64_t final_batch_iters = total_batch_iterations.load();

        double mean_single_cw_iters = (final_cws > 0) ? static_cast<double>(final_cw_iters) / final_cws : 0.0;
        double fer = (final_cws > 0) ? static_cast<double>(final_fe) / final_cws : 0.0;
        double ber = (final_cws > 0) ? static_cast<double>(final_be) / (final_cws * pcm.K) : 0.0;
        double throughput_msps = (final_cws * (pcm.N - pcm.P)) / (total_sec * 1e6);

        std::cout << "\n[COMPLETED " << std::fixed << std::setprecision(2) << eb_n0_db << " dB]"
                  << " Blocks: " << final_cws
                  << " | FE: " << final_fe
                  << " | BE: " << final_be
                  << " | FER: " << std::scientific << std::setprecision(4) << fer
                  << " | BER: " << ber
                  << " | MeanIter: " << std::fixed << std::setprecision(2) << mean_single_cw_iters
                  << " | Tput: " << std::setprecision(2) << throughput_msps << " Msps"
                  << " | Time: " << std::setprecision(1) << total_sec << " s"
                  << std::endl;

        csv_file << std::fixed << std::setprecision(2) << eb_n0_db << ","
                 << final_cws << ","
                 << final_be << ","
                 << final_fe << ","
                 << std::scientific << std::setprecision(6) << ber << ","
                 << fer << ","
                 << std::fixed << std::setprecision(3) << mean_single_cw_iters << ","
                 << std::setprecision(2) << throughput_msps << ","
                 << std::setprecision(2) << total_sec << "\n";
        csv_file.flush();
    }

    std::cout << "\n[SUCCESS] Full waterfall sweep completed! Results saved to " << csv_path << std::endl;
    return 0;
}
