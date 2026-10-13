#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <fstream>
#include <string>

// Include our SFPI and PRNG primitives
#include "../engines/tile_matrix_4cw/sfpi_host_emu.h"

// ------------------------------------------------------------------------------
// Polar Marsaglia Generator (as implemented in test_tile_matrix_40tile_lockstep_stepper_sfpi.cpp)
// ------------------------------------------------------------------------------
struct HostXoshiro128Plus {
    uint32_t s[4];
    static inline uint32_t rotl(const uint32_t x, int k) {
        return (x << k) | (x >> (32 - k));
    }
    void init(uint32_t seed) {
        uint32_t z = seed + 0x9E3779B9U;
        for (int i = 0; i < 4; ++i) {
            z = (z ^ (z >> 16)) * 0x85EBCA6BU;
            z = (z ^ (z >> 13)) * 0xC2B2AE35U;
            s[i] = z ^ (z >> 16);
            z += 0x9E3779B9U;
        }
    }
    uint32_t next() {
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
};

inline void generate_polar_marsaglia_pair(HostXoshiro128Plus& rng, float& z0, float& z1) {
    float u, v, s;
    do {
        u = static_cast<float>(rng.next()) / 2147483648.0f - 1.0f;
        v = static_cast<float>(rng.next()) / 2147483648.0f - 1.0f;
        s = u * u + v * v;
    } while (s >= 1.0f || s == 0.0f);
    float mul = std::sqrt(-2.0f * std::log(s) / s);
    z0 = u * mul;
    z1 = v * mul;
}

// ------------------------------------------------------------------------------
// Tensix Hardware Vector Box-Muller Primitives
// ------------------------------------------------------------------------------
namespace TensixHw {

inline float to_bf16_trunc(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    u &= 0xFFFF0000;
    float res;
    std::memcpy(&res, &u, sizeof(res));
    return res;
}

inline float to_bf16_rne(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    uint32_t lsb = (u >> 16) & 1;
    u += 0x7FFF + lsb;
    u &= 0xFFFF0000;
    float res;
    std::memcpy(&res, &u, sizeof(res));
    return res;
}

inline float sfpu_fast_ln(float a) {
    a = std::max(a, 1e-7f);
    uint32_t u;
    std::memcpy(&u, &a, sizeof(u));
    int exp_val = static_cast<int>((u >> 23) & 0xFF) - 127;
    uint32_t m_bits = (u & 0x007FFFFF) | 0x3F800000;
    float m;
    std::memcpy(&m, &m_bits, sizeof(m));
    float z = m - 1.0f;

    float p = to_bf16_trunc(-0.0528f);
    p = p * z + to_bf16_trunc(0.1264f);
    p = p * z - to_bf16_trunc(0.1678f);
    p = p * z + to_bf16_trunc(0.2057f);
    p = p * z - to_bf16_trunc(0.2490f);
    p = p * z + to_bf16_trunc(0.3328f);
    p = p * z - 0.5f;

    float ln_m = z * (p * z + 1.0f);
    float ln_2 = to_bf16_trunc(0.69314718f);
    return ln_m + static_cast<float>(exp_val) * ln_2;
}

inline float sfpu_fast_sqrt(float val) {
    if (val <= 0.0f) return 0.0f;
    uint32_t u;
    std::memcpy(&u, &val, sizeof(u));
    uint32_t magic = 0x5F3759DF - (u >> 1);
    float approx;
    std::memcpy(&approx, &magic, sizeof(approx));
    float neg_half_val = val * -0.5f;
    approx = ((approx * approx) * neg_half_val + 1.5f) * approx;
    approx = ((approx * approx) * neg_half_val + 1.5f) * approx;
    return approx * val;
}

struct VectorXoshiro128Plus {
    uint32_t s[4];
    static inline uint32_t rotl(const uint32_t x, int k) {
        return (x << k) | (x >> (32 - k));
    }
    static inline uint32_t arx_hash(uint32_t x, uint32_t k) {
        x ^= k;
        x = (x ^ (x >> 16)) * 0x85EBCA6BU;
        x = (x ^ (x >> 13)) * 0xC2B2AE35U;
        return x ^ (x >> 16);
    }
    void init(uint32_t seed) {
        s[0] = arx_hash(seed, 0x9E3779B9U);
        s[1] = arx_hash(seed, 0x85EBCA6BU);
        s[2] = arx_hash(seed, 0xC2B2AE35U);
        s[3] = arx_hash(seed, 0x27D4EB2FU);
        if ((s[0] | s[1] | s[2] | s[3]) == 0) s[0] = 1;
    }
    uint32_t next() {
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
};

inline void generate_tensix_bm_pair(
    VectorXoshiro128Plus& rng,
    float& z0,
    float& z1,
    bool truncate_u = true,
    bool truncate_output = true,
    bool fast_trig = true,
    bool fast_ln_sqrt = true
) {
    uint32_t raw1 = rng.next();
    uint32_t u_bits1 = ((raw1 >> 9) & 0x007FFFFF) | 0x3F800000;
    float u1;
    std::memcpy(&u1, &u_bits1, sizeof(u1));
    u1 -= 1.0f;
    u1 = std::max(u1, 1e-7f);
    if (truncate_u) u1 = to_bf16_trunc(u1);

    uint32_t raw2 = rng.next();
    uint32_t u_bits2 = ((raw2 >> 9) & 0x007FFFFF) | 0x3F800000;
    float u2;
    std::memcpy(&u2, &u_bits2, sizeof(u2));
    u2 -= 1.0f;
    u2 = std::max(u2, 1e-7f);
    if (truncate_u) u2 = to_bf16_trunc(u2);

    float r;
    if (fast_ln_sqrt) {
        float ln_u1 = sfpu_fast_ln(u1);
        r = sfpu_fast_sqrt(ln_u1 * -2.0f);
    } else {
        r = std::sqrt(-2.0f * std::log(u1));
    }

    float theta = u2 * 6.2831853f - 3.14159265f;
    float cos_val, sin_val;

    if (fast_trig) {
        float s_th = theta * theta;
        cos_val = to_bf16_trunc(0.0000248f);
        cos_val = cos_val * s_th - to_bf16_trunc(0.0013889f);
        cos_val = cos_val * s_th + to_bf16_trunc(0.0416667f);
        cos_val = cos_val * s_th - 0.5f;
        cos_val = cos_val * s_th + 1.0f;

        sin_val = to_bf16_trunc(0.00000276f);
        sin_val = sin_val * s_th - to_bf16_trunc(0.0001984f);
        sin_val = sin_val * s_th + to_bf16_trunc(0.0083333f);
        sin_val = sin_val * s_th - to_bf16_trunc(0.1666667f);
        sin_val = sin_val * s_th + 1.0f;
        sin_val = sin_val * theta;
    } else {
        cos_val = std::cos(theta);
        sin_val = std::sin(theta);
    }

    z0 = r * cos_val;
    z1 = r * sin_val;

    if (truncate_output) {
        z0 = to_bf16_trunc(z0);
        z1 = to_bf16_trunc(z1);
    }
}

} // namespace TensixHw

// ------------------------------------------------------------------------------
// Statistical Accumulator Structure
// ------------------------------------------------------------------------------
struct Statistics {
    uint64_t count = 0;
    double sum = 0.0;
    double sum_sq = 0.0;
    double sum_cube = 0.0;
    double sum_quad = 0.0;
    double min_val = 1e9;
    double max_val = -1e9;

    uint64_t tail_1_0 = 0;
    uint64_t tail_2_0 = 0;
    uint64_t tail_2_5 = 0;
    uint64_t tail_3_0 = 0;
    uint64_t tail_3_5 = 0;
    uint64_t tail_4_0 = 0;
    uint64_t tail_4_5 = 0;
    uint64_t tail_5_0 = 0;

    uint64_t neg_tail_1_0 = 0;
    uint64_t neg_tail_2_0 = 0;
    uint64_t neg_tail_3_0 = 0;
    uint64_t neg_tail_4_0 = 0;
    uint64_t neg_tail_5_0 = 0;

    static constexpr int NUM_BINS = 1200;
    static constexpr double BIN_MIN = -6.0;
    static constexpr double BIN_MAX = 6.0;
    static constexpr double BIN_WIDTH = (BIN_MAX - BIN_MIN) / NUM_BINS;
    uint64_t histogram[NUM_BINS] = {0};

    void add(float z) {
        double d = static_cast<double>(z);
        count++;
        sum += d;
        double d2 = d * d;
        sum_sq += d2;
        sum_cube += d2 * d;
        sum_quad += d2 * d2;
        if (d < min_val) min_val = d;
        if (d > max_val) max_val = d;

        double abs_d = std::abs(d);
        if (abs_d >= 1.0) tail_1_0++;
        if (abs_d >= 2.0) tail_2_0++;
        if (abs_d >= 2.5) tail_2_5++;
        if (abs_d >= 3.0) tail_3_0++;
        if (abs_d >= 3.5) tail_3_5++;
        if (abs_d >= 4.0) tail_4_0++;
        if (abs_d >= 4.5) tail_4_5++;
        if (abs_d >= 5.0) tail_5_0++;

        if (d <= -1.0) neg_tail_1_0++;
        if (d <= -2.0) neg_tail_2_0++;
        if (d <= -3.0) neg_tail_3_0++;
        if (d <= -4.0) neg_tail_4_0++;
        if (d <= -5.0) neg_tail_5_0++;

        int bin = static_cast<int>((d - BIN_MIN) / BIN_WIDTH);
        if (bin >= 0 && bin < NUM_BINS) {
            histogram[bin]++;
        }
    }

    void merge(const Statistics& other) {
        count += other.count;
        sum += other.sum;
        sum_sq += other.sum_sq;
        sum_cube += other.sum_cube;
        sum_quad += other.sum_quad;
        if (other.min_val < min_val) min_val = other.min_val;
        if (other.max_val > max_val) max_val = other.max_val;

        tail_1_0 += other.tail_1_0;
        tail_2_0 += other.tail_2_0;
        tail_2_5 += other.tail_2_5;
        tail_3_0 += other.tail_3_0;
        tail_3_5 += other.tail_3_5;
        tail_4_0 += other.tail_4_0;
        tail_4_5 += other.tail_4_5;
        tail_5_0 += other.tail_5_0;

        neg_tail_1_0 += other.neg_tail_1_0;
        neg_tail_2_0 += other.neg_tail_2_0;
        neg_tail_3_0 += other.neg_tail_3_0;
        neg_tail_4_0 += other.neg_tail_4_0;
        neg_tail_5_0 += other.neg_tail_5_0;

        for (int i = 0; i < NUM_BINS; ++i) {
            histogram[i] += other.histogram[i];
        }
    }

    double mean() const { return sum / static_cast<double>(count); }
    double variance() const {
        double m = mean();
        return (sum_sq / static_cast<double>(count)) - (m * m);
    }
    double stddev() const { return std::sqrt(std::max(0.0, variance())); }
    double skewness() const {
        double m = mean();
        double s = stddev();
        if (s == 0.0) return 0.0;
        double s3 = s * s * s;
        double n = static_cast<double>(count);
        double m3 = (sum_cube / n) - 3.0 * m * (sum_sq / n) + 2.0 * m * m * m;
        return m3 / s3;
    }
    double excess_kurtosis() const {
        double m = mean();
        double s = stddev();
        if (s == 0.0) return 0.0;
        double s4 = s * s * s * s;
        double n = static_cast<double>(count);
        double m4 = (sum_quad / n) - 4.0 * m * (sum_cube / n) + 6.0 * m * m * (sum_sq / n) - 3.0 * m * m * m * m;
        return (m4 / s4) - 3.0;
    }
};

// ------------------------------------------------------------------------------
// Cross-Lane Spatial Correlation Benchmarking (32 Lanes)
// ------------------------------------------------------------------------------
void benchmark_cross_lane_correlation(uint32_t num_batches = 500000) {
    std::cout << "\n========================================================================\n";
    std::cout << " 32-LANE SIMD SPATIAL CROSS-CORRELATION ANALYSIS (Tensix ARX-Seeded)\n";
    std::cout << " Batches: " << num_batches << " (16 draws/lane/batch = " << (num_batches * 16) << " draws/lane)\n";
    std::cout << "========================================================================\n" << std::endl;

    const int NUM_LANES = 32;
    std::vector<double> sum(NUM_LANES, 0.0);
    std::vector<double> sum_sq(NUM_LANES, 0.0);
    // Upper triangle of covariance matrix: 32 x 32
    std::vector<std::vector<double>> sum_cross(NUM_LANES, std::vector<double>(NUM_LANES, 0.0));
    uint64_t total_draws_per_lane = static_cast<uint64_t>(num_batches) * 16ULL;

    #pragma omp parallel
    {
        std::vector<double> local_sum(NUM_LANES, 0.0);
        std::vector<double> local_sum_sq(NUM_LANES, 0.0);
        std::vector<std::vector<double>> local_cross(NUM_LANES, std::vector<double>(NUM_LANES, 0.0));

        #pragma omp for schedule(static)
        for (uint32_t b = 0; b < num_batches; ++b) {
            TensixHw::VectorXoshiro128Plus rngs[NUM_LANES];
            uint32_t seed_base = 0x12345678U + b * 31337U;
            for (int l = 0; l < NUM_LANES; ++l) {
                rngs[l].init(seed_base + (2 * l));
            }

            for (int p = 0; p < 16; ++p) {
                float z[NUM_LANES];
                for (int l = 0; l < NUM_LANES; ++l) {
                    float z0, z1;
                    TensixHw::generate_tensix_bm_pair(rngs[l], z0, z1);
                    z[l] = z0; // Examine lane variates
                }

                for (int l = 0; l < NUM_LANES; ++l) {
                    double dl = static_cast<double>(z[l]);
                    local_sum[l] += dl;
                    local_sum_sq[l] += dl * dl;
                    for (int l2 = l + 1; l2 < NUM_LANES; ++l2) {
                        local_cross[l][l2] += dl * static_cast<double>(z[l2]);
                    }
                }
            }
        }

        #pragma omp critical
        {
            for (int l = 0; l < NUM_LANES; ++l) {
                sum[l] += local_sum[l];
                sum_sq[l] += local_sum_sq[l];
                for (int l2 = l + 1; l2 < NUM_LANES; ++l2) {
                    sum_cross[l][l2] += local_cross[l][l2];
                }
            }
        }
    }

    double n = static_cast<double>(total_draws_per_lane);
    std::vector<double> mean(NUM_LANES, 0.0);
    std::vector<double> stddev(NUM_LANES, 0.0);
    for (int l = 0; l < NUM_LANES; ++l) {
        mean[l] = sum[l] / n;
        double var = (sum_sq[l] / n) - (mean[l] * mean[l]);
        stddev[l] = std::sqrt(std::max(0.0, var));
    }

    double max_rho = 0.0;
    double sum_abs_rho = 0.0;
    int num_pairs = 0;
    int pairs_above_3sigma = 0;
    int pairs_above_005 = 0;
    int pairs_above_010 = 0;
    double three_sigma_floor = 3.0 / std::sqrt(n);

    for (int l1 = 0; l1 < NUM_LANES; ++l1) {
        for (int l2 = l1 + 1; l2 < NUM_LANES; ++l2) {
            double cov = (sum_cross[l1][l2] / n) - (mean[l1] * mean[l2]);
            double rho = cov / (stddev[l1] * stddev[l2]);
            double abs_rho = std::abs(rho);

            if (abs_rho > max_rho) max_rho = abs_rho;
            sum_abs_rho += abs_rho;
            num_pairs++;
            if (abs_rho > three_sigma_floor) pairs_above_3sigma++;
            if (abs_rho > 0.05) pairs_above_005++;
            if (abs_rho > 0.10) pairs_above_010++;
        }
    }

    double mean_rho = sum_abs_rho / num_pairs;

    std::cout << "32-Lane Correlation Results across " << num_pairs << " unique lane pairs:\n";
    std::cout << "  3-Sigma White Noise Floor: " << std::fixed << std::setprecision(6) << three_sigma_floor << "\n";
    std::cout << "  Max |rho|:                 " << std::fixed << std::setprecision(6) << max_rho << "\n";
    std::cout << "  Mean |rho|:                " << std::fixed << std::setprecision(6) << mean_rho << "\n";
    std::cout << "  Lane pairs > 3-Sigma:      " << pairs_above_3sigma << " / " << num_pairs << " ("
              << (100.0 * pairs_above_3sigma / num_pairs) << "%)\n";
    std::cout << "  Lane pairs > 0.05:         " << pairs_above_005 << " / " << num_pairs << "\n";
    std::cout << "  Lane pairs > 0.10:         " << pairs_above_010 << " / " << num_pairs << "\n";
}

// ------------------------------------------------------------------------------
// Main Study Runner
// ------------------------------------------------------------------------------
int main(int argc, char** argv) {
    uint64_t total_samples = 100000000ULL; // 100 Million variates default
    if (argc > 1) {
        total_samples = std::stoull(argv[1]);
    }
    int num_threads = omp_get_max_threads();
    if (argc > 2) {
        num_threads = std::stoi(argv[2]);
        omp_set_num_threads(num_threads);
    }

    std::cout << "========================================================================\n";
    std::cout << " COMPREHENSIVE AWGN NOISE MODEL COMPARISON STUDY\n";
    std::cout << " Total Variates Per Model: " << total_samples << " (" << (total_samples / 1000000) << " Million)\n";
    std::cout << " OpenMP Threads:           " << num_threads << "\n";
    std::cout << " Target Distribution:      Standard Normal N(0, 1)\n";
    std::cout << "========================================================================\n" << std::endl;

    const std::vector<std::string> model_names = {
        "1. Polar Marsaglia (Host Golden)",
        "2. Tensix Vector BM (Both Z0 & Z1)",
        "3. Tensix BM Z0 (r*cos: CW0/CW2)",
        "4. Tensix BM Z1 (r*sin: CW1/CW3)",
        "5. Tensix BM (No U-Trunc)",
        "6. Tensix BM (Exact Math)",
        "7. Tensix BM (RNE Rounding)"
    };

    std::vector<Statistics> model_stats(model_names.size());

    uint64_t pairs_per_thread = (total_samples / 2) / num_threads;
    uint64_t actual_total_samples = pairs_per_thread * num_threads * 2;

    for (size_t m = 0; m < model_names.size(); ++m) {
        std::cout << "Running Model [" << model_names[m] << "] over " << actual_total_samples << " samples..." << std::flush;
        auto t0 = std::chrono::high_resolution_clock::now();

        std::vector<Statistics> thread_stats(num_threads);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            HostXoshiro128Plus polar_rng;
            polar_rng.init(123456789U + tid * 1013904223U);

            TensixHw::VectorXoshiro128Plus tensix_rng;
            tensix_rng.init(0x12345678U + tid * 7919U);

            Statistics& lstat = thread_stats[tid];

            if (m == 2 || m == 3) {
                // Models 2 & 3: Evaluate 100M variates of Z0 or Z1 exclusively
                for (uint64_t p = 0; p < actual_total_samples / num_threads; ++p) {
                    float z0, z1;
                    TensixHw::generate_tensix_bm_pair(tensix_rng, z0, z1, true, true, true, true);
                    if (m == 2) lstat.add(z0);
                    else lstat.add(z1);
                }
            } else {
                for (uint64_t p = 0; p < pairs_per_thread; ++p) {
                    float z0, z1;
                    if (m == 0) {
                        generate_polar_marsaglia_pair(polar_rng, z0, z1);
                    } else if (m == 1) {
                        TensixHw::generate_tensix_bm_pair(tensix_rng, z0, z1, true, true, true, true);
                    } else if (m == 4) {
                        TensixHw::generate_tensix_bm_pair(tensix_rng, z0, z1, false, true, true, true);
                    } else if (m == 5) {
                        TensixHw::generate_tensix_bm_pair(tensix_rng, z0, z1, true, true, false, false);
                    } else if (m == 6) {
                        TensixHw::generate_tensix_bm_pair(tensix_rng, z0, z1, true, false, true, true);
                        z0 = TensixHw::to_bf16_rne(z0);
                        z1 = TensixHw::to_bf16_rne(z1);
                    }
                    lstat.add(z0);
                    lstat.add(z1);
                }
            }
        }

        for (int tid = 0; tid < num_threads; ++tid) {
            model_stats[m].merge(thread_stats[tid]);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
        double rate_msps = (actual_total_samples / 1e6) / elapsed_s;
        std::cout << " Done in " << std::fixed << std::setprecision(2) << elapsed_s << " s (" << rate_msps << " Msps)\n";
    }

    // --------------------------------------------------------------------------
    // Print Detailed Statistical Report
    // --------------------------------------------------------------------------
    std::cout << "\n========================================================================================================================\n";
    std::cout << "                                  STATISTICAL MOMENTS COMPARISON TABLE                                 \n";
    std::cout << "========================================================================================================================\n";
    std::cout << std::left << std::setw(32) << "Metric"
              << std::setw(14) << "Ideal N(0,1)"
              << std::setw(14) << "Polar (Host)"
              << std::setw(14) << "Tensix (Both)"
              << std::setw(14) << "Tensix Z0(cos)"
              << std::setw(14) << "Tensix Z1(sin)"
              << std::setw(14) << "No U-Trunc"
              << std::setw(14) << "Exact Math"
              << std::setw(14) << "RNE Rounding" << "\n";
    std::cout << std::string(144, '-') << "\n";

    auto print_row_d = [&](const std::string& name, double ideal, auto fn) {
        std::cout << std::left << std::setw(32) << name
                  << std::setw(14) << std::fixed << std::setprecision(6) << ideal;
        for (size_t m = 0; m < model_names.size(); ++m) {
            std::cout << std::setw(14) << std::fixed << std::setprecision(6) << fn(model_stats[m]);
        }
        std::cout << "\n";
    };

    print_row_d("Mean (mu)", 0.0, [](const Statistics& s) { return s.mean(); });
    print_row_d("Variance (sigma^2)", 1.0, [](const Statistics& s) { return s.variance(); });
    print_row_d("Standard Deviation (sigma)", 1.0, [](const Statistics& s) { return s.stddev(); });

    std::cout << std::left << std::setw(32) << "Effective SNR Shift (dB)"
              << std::setw(14) << "0.0000 dB";
    for (size_t m = 0; m < model_names.size(); ++m) {
        double var = model_stats[m].variance();
        double d_db = -10.0 * std::log10(var);
        std::cout << std::setw(14) << (std::to_string(d_db).substr(0, 8) + " dB");
    }
    std::cout << "\n";

    print_row_d("Skewness (gamma_1)", 0.0, [](const Statistics& s) { return s.skewness(); });
    print_row_d("Excess Kurtosis (gamma_2)", 0.0, [](const Statistics& s) { return s.excess_kurtosis(); });
    print_row_d("Min Value", -6.0, [](const Statistics& s) { return s.min_val; });
    print_row_d("Max Value", +6.0, [](const Statistics& s) { return s.max_val; });

    // --------------------------------------------------------------------------
    // Tail Probabilities Table
    // --------------------------------------------------------------------------
    std::cout << "\n========================================================================================================================\n";
    std::cout << "                                  TAIL EXCEEDANCE PROBABILITIES P(|Z| >= T)                              \n";
    std::cout << "========================================================================================================================\n";
    std::cout << std::left << std::setw(20) << "Threshold T"
              << std::setw(16) << "Theoretical"
              << std::setw(16) << "Polar (Host)"
              << std::setw(16) << "Tensix (Both)"
              << std::setw(16) << "Tensix Z0(cos)"
              << std::setw(16) << "Tensix Z1(sin)"
              << std::setw(16) << "Ratio(Z0/Z1)"
              << std::setw(16) << "Ratio(HW/Polar)" << "\n";
    std::cout << std::string(132, '-') << "\n";

    auto print_tail_detail = [&](const std::string& name, double ideal, auto count_fn) {
        double p_polar = static_cast<double>(count_fn(model_stats[0])) / model_stats[0].count;
        double p_both  = static_cast<double>(count_fn(model_stats[1])) / model_stats[1].count;
        double p_z0    = static_cast<double>(count_fn(model_stats[2])) / model_stats[2].count;
        double p_z1    = static_cast<double>(count_fn(model_stats[3])) / model_stats[3].count;
        double r_z0_z1 = p_z1 > 0 ? (p_z0 / p_z1) : 0.0;
        double r_hw_pol = p_polar > 0 ? (p_both / p_polar) : 0.0;

        std::cout << std::left << std::setw(20) << name
                  << std::setw(16) << std::scientific << std::setprecision(4) << ideal
                  << std::setw(16) << std::scientific << std::setprecision(4) << p_polar
                  << std::setw(16) << std::scientific << std::setprecision(4) << p_both
                  << std::setw(16) << std::scientific << std::setprecision(4) << p_z0
                  << std::setw(16) << std::scientific << std::setprecision(4) << p_z1
                  << std::setw(16) << std::fixed << std::setprecision(4) << r_z0_z1 << "x"
                  << std::setw(16) << std::fixed << std::setprecision(4) << r_hw_pol << "x\n";
    };

    print_tail_detail("|Z| >= 1.0 sigma", 3.173105e-01, [](const Statistics& s) { return s.tail_1_0; });
    print_tail_detail("|Z| >= 2.0 sigma", 4.550026e-02, [](const Statistics& s) { return s.tail_2_0; });
    print_tail_detail("|Z| >= 2.5 sigma", 1.241933e-02, [](const Statistics& s) { return s.tail_2_5; });
    print_tail_detail("|Z| >= 3.0 sigma", 2.699796e-03, [](const Statistics& s) { return s.tail_3_0; });
    print_tail_detail("|Z| >= 3.5 sigma", 4.652582e-04, [](const Statistics& s) { return s.tail_3_5; });
    print_tail_detail("|Z| >= 4.0 sigma", 6.334248e-05, [](const Statistics& s) { return s.tail_4_0; });
    print_tail_detail("|Z| >= 4.5 sigma", 6.795346e-06, [](const Statistics& s) { return s.tail_4_5; });
    print_tail_detail("|Z| >= 5.0 sigma", 5.733031e-07, [](const Statistics& s) { return s.tail_5_0; });

    // Directional negative tail
    std::cout << "\nDirectional Left Tail P(Z <= -T) (Inverted Channel Bit Probabilities):\n";
    std::cout << std::string(132, '-') << "\n";
    print_tail_detail("Z <= -1.0 sigma", 1.586553e-01, [](const Statistics& s) { return s.neg_tail_1_0; });
    print_tail_detail("Z <= -2.0 sigma", 2.275013e-02, [](const Statistics& s) { return s.neg_tail_2_0; });
    print_tail_detail("Z <= -3.0 sigma", 1.349898e-03, [](const Statistics& s) { return s.neg_tail_3_0; });
    print_tail_detail("Z <= -4.0 sigma", 3.167124e-05, [](const Statistics& s) { return s.neg_tail_4_0; });
    print_tail_detail("Z <= -5.0 sigma", 2.866516e-07, [](const Statistics& s) { return s.neg_tail_5_0; });

    // --------------------------------------------------------------------------
    // Run 32-Lane Spatial Cross-Correlation Analysis
    // --------------------------------------------------------------------------
    benchmark_cross_lane_correlation(200000); // 200,000 batches = 3.2M draws per lane

    std::cout << "========================================================================\n" << std::endl;
    return 0;
}
