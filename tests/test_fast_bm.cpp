#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <chrono>
#include <iomanip>
#include "engines/tile_matrix_4cw/ldpc_tile_matrix_core.h"
#include "engines/tile_matrix_4cw/box_muller_lut.h"

int main() {
    Xoshiro128Plus rng1;
    rng1.init(0x12345678);

    float mu = 3.5f;
    float sigma = 2.64575f;
    const int N = 100000;

    std::vector<float> samples_fast;
    samples_fast.reserve(N * 2);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        float l0, l1;
        fast_box_muller_pair(rng1, mu, sigma, l0, l1);
        samples_fast.push_back((l0 - mu) / sigma);
        samples_fast.push_back((l1 - mu) / sigma);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double mean = 0, var = 0, kurt = 0;
    for (float x : samples_fast) mean += x;
    mean /= samples_fast.size();
    for (float x : samples_fast) {
        double d = x - mean;
        var += d * d;
        kurt += d * d * d * d;
    }
    var /= samples_fast.size();
    kurt = (kurt / samples_fast.size()) / (var * var);

    double ns_per_pair = std::chrono::duration<double, std::nano>(t1 - t0).count() / N;

    std::cout << "FAST BOX-MULLER LUT RESULTS (200,000 samples):" << std::endl;
    std::cout << "  Mean      : " << mean << " (Expected: 0.0)" << std::endl;
    std::cout << "  Variance  : " << var << " (Expected: 1.0)" << std::endl;
    std::cout << "  Kurtosis  : " << kurt << " (Expected: 3.0)" << std::endl;
    std::cout << "  Host Speed: " << ns_per_pair << " ns/pair" << std::endl;

    return 0;
}
