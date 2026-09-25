#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <thread>
#include <atomic>
#include <cstring>
#include <cstdint>

// ==============================================================================
// Exact Math Primitives from kernel/compute_trisc_ldpc_awgn_sim.cpp
// ==============================================================================

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
        if ((s[0] | s[1] | s[2] | s[3]) == 0) {
            s[0] = 1;
        }
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
    Xoshiro128Plus& rng, 
    float mu_llr, 
    float sigma_llr, 
    float& llr0, 
    float& llr1
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

// Theoretical BPSK BER in AWGN: Pb = 0.5 * erfc(sqrt(Eb/N0))
inline double theoretical_bpsk_ber(double eb_n0_db) {
    double snr_lin = std::pow(10.0, eb_n0_db / 10.0);
    return 0.5 * std::erfc(std::sqrt(snr_lin));
}

// Worker thread for parallel AWGN simulation
struct ThreadResult {
    uint64_t total_bits = 0;
    uint64_t fp32_bit_errors = 0;
    uint64_t bf16_bit_errors = 0;
};

void sim_worker(
    uint32_t seed_lo,
    uint32_t seed_hi,
    float mu_llr,
    float sigma_llr,
    uint64_t target_bits,
    uint64_t target_errors,
    std::atomic<uint64_t>& shared_errors,
    std::atomic<uint64_t>& shared_bits,
    ThreadResult& result
) {
    Xoshiro128Plus rng;
    rng.init(seed_lo, seed_hi);

    constexpr uint64_t BATCH_PAIRS = 50000; // 100,000 bits per batch
    uint64_t local_bits = 0;
    uint64_t local_fp32_errs = 0;
    uint64_t local_bf16_errs = 0;

    while (shared_errors.load(std::memory_order_relaxed) < target_errors &&
           shared_bits.load(std::memory_order_relaxed) < target_bits) {

        uint64_t batch_fp32 = 0;
        uint64_t batch_bf16 = 0;

        for (uint64_t i = 0; i < BATCH_PAIRS; i++) {
            float l0, l1;
            generate_awgn_llr_pair(rng, mu_llr, sigma_llr, l0, l1);

            // FP32 Decision: negative LLR is an error
            batch_fp32 += (l0 < 0.0f) + (l1 < 0.0f);

            // BF16 Decision: sign bit of bfloat16
            uint16_t bf0 = fp32_to_bf16(l0);
            uint16_t bf1 = fp32_to_bf16(l1);
            batch_bf16 += (bf0 >> 15) + (bf1 >> 15);
        }

        local_bits += BATCH_PAIRS * 2;
        local_fp32_errs += batch_fp32;
        local_bf16_errs += batch_bf16;

        shared_errors.fetch_add(batch_fp32, std::memory_order_relaxed);
        shared_bits.fetch_add(BATCH_PAIRS * 2, std::memory_order_relaxed);
    }

    result.total_bits = local_bits;
    result.fp32_bit_errors = local_fp32_errs;
    result.bf16_bit_errors = local_bf16_errs;
}

int main(int argc, char** argv) {
    double start_snr = 0.0;
    double stop_snr = 10.0;
    double step_snr = 0.5;
    uint64_t min_errors = 1000;
    uint64_t max_bits = 1000000000ULL; // Up to 1 billion bits per SNR point
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 8;

    if (argc >= 2) start_snr = std::stod(argv[1]);
    if (argc >= 3) stop_snr  = std::stod(argv[2]);
    if (argc >= 4) step_snr  = std::stod(argv[3]);
    if (argc >= 5) min_errors = std::stoull(argv[4]);
    if (argc >= 6) max_bits   = std::stoull(argv[5]);

    std::cout << "========================================================================================\n";
    std::cout << "  Uncoded BPSK AWGN Monte Carlo Simulation (Rate R = 1.0)\n";
    std::cout << "  Exact Kernel Noise Generator (Xoshiro128+ & Polar Box-Muller)\n";
    std::cout << "  Threads: " << num_threads << " | Min Errors: " << min_errors << " | Max Bits: " << max_bits << "\n";
    std::cout << "========================================================================================\n\n";

    std::cout << std::setw(8) << "Eb/N0(dB)" 
              << std::setw(14) << "Total Bits" 
              << std::setw(12) << "FP32 Errs" 
              << std::setw(14) << "Sim BER(FP32)" 
              << std::setw(14) << "Sim BER(BF16)" 
              << std::setw(14) << "Theory BER" 
              << std::setw(12) << "Sim/Theory" 
              << std::setw(10) << "Delta %" 
              << std::setw(10) << "Time(s)" 
              << std::endl;
    std::cout << std::string(98, '-') << std::endl;

    std::ofstream csv_file("Results/results_uncoded_bpsk_awgn.csv");
    csv_file << "ebn0_db,total_bits,fp32_errors,bf16_errors,ber_fp32,ber_bf16,ber_theory,ratio_fp32_theory\n";

    for (double snr = start_snr; snr <= stop_snr + 1e-6; snr += step_snr) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // Exact Eb/N0 to LLR mapping (from ldpc_sim.cpp with Rate = 1.0)
        double rate = 1.0;
        double snr_lin = std::pow(10.0, snr / 10.0);
        float mu_llr = static_cast<float>(4.0 * rate * snr_lin);
        float sigma_llr = static_cast<float>(std::sqrt(8.0 * rate * snr_lin));

        double theory_ber = theoretical_bpsk_ber(snr);

        std::atomic<uint64_t> shared_errors(0);
        std::atomic<uint64_t> shared_bits(0);
        std::vector<ThreadResult> results(num_threads);
        std::vector<std::thread> workers;

        for (unsigned int t = 0; t < num_threads; t++) {
            uint32_t seed_lo = 10007 + t * 65537 + static_cast<uint32_t>(snr * 100);
            uint32_t seed_hi = 99991 + t * 31337 + static_cast<uint32_t>(snr * 1000);
            workers.emplace_back(
                sim_worker,
                seed_lo,
                seed_hi,
                mu_llr,
                sigma_llr,
                max_bits,
                min_errors,
                std::ref(shared_errors),
                std::ref(shared_bits),
                std::ref(results[t])
            );
        }

        for (auto& w : workers) {
            w.join();
        }

        uint64_t total_bits = 0;
        uint64_t total_fp32_errs = 0;
        uint64_t total_bf16_errs = 0;
        for (const auto& r : results) {
            total_bits += r.total_bits;
            total_fp32_errs += r.fp32_bit_errors;
            total_bf16_errs += r.bf16_bit_errors;
        }

        double sim_ber_fp32 = static_cast<double>(total_fp32_errs) / total_bits;
        double sim_ber_bf16 = static_cast<double>(total_bf16_errs) / total_bits;
        double ratio = (theory_ber > 0.0) ? (sim_ber_fp32 / theory_ber) : 1.0;
        double delta_pct = (ratio - 1.0) * 100.0;

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();

        std::cout << std::fixed << std::setprecision(2)
                  << std::setw(8) << snr 
                  << std::setw(14) << total_bits 
                  << std::setw(12) << total_fp32_errs 
                  << std::scientific << std::setprecision(4)
                  << std::setw(14) << sim_ber_fp32 
                  << std::setw(14) << sim_ber_bf16 
                  << std::setw(14) << theory_ber 
                  << std::fixed << std::setprecision(4)
                  << std::setw(12) << ratio 
                  << std::setprecision(2) << std::showpos
                  << std::setw(9) << delta_pct << "%" << std::noshowpos
                  << std::fixed << std::setprecision(2)
                  << std::setw(10) << elapsed_sec 
                  << std::endl;

        csv_file << std::fixed << std::setprecision(2) << snr << ","
                 << total_bits << ","
                 << total_fp32_errs << ","
                 << total_bf16_errs << ","
                 << std::scientific << std::setprecision(8)
                 << sim_ber_fp32 << ","
                 << sim_ber_bf16 << ","
                 << theory_ber << ","
                 << std::fixed << std::setprecision(6)
                 << ratio << "\n";
        csv_file.flush();

        // If at high SNR we achieved fewer than 10 errors despite max_bits, break
        if (total_fp32_errs < 10 && total_bits >= max_bits) {
            std::cout << "  [INFO] Reached bit limit with < 10 errors. Terminating sweep.\n";
            break;
        }
    }

    csv_file.close();
    std::cout << std::string(98, '-') << std::endl;
    std::cout << "[DONE] Results written to Results/results_uncoded_bpsk_awgn.csv\n";
    return 0;
}
