#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include "../engines/tile_matrix_4cw/ldpc_tile_matrix_core.h"

inline float to_bf16(float f) {
    return bf16_to_fp32(fp32_to_bf16(f));
}

inline float approx_min_star_fp32(float a, float b) {
    float min_v = std::min(a, b);
    float max_v = std::max(a, b);
    float diff  = max_v - min_v;
    float line1 = 0.69315f - 0.4000f * diff;
    float line2 = 0.65715f - 0.3280f * diff;
    float corr  = std::max(std::max(line1, line2), 0.0f);
    return std::max(min_v - corr, 0.0f);
}

inline float approx_min_star_exact_bf16(float a, float b) {
    float min_v = std::min(a, b);
    float max_v = std::max(a, b);
    float diff  = to_bf16(max_v - min_v);

    static const float c_0_69315   = to_bf16(0.69315f);
    static const float c_neg_0_4   = to_bf16(-0.4000f);
    static const float c_0_65715   = to_bf16(0.65715f);
    static const float c_neg_0_328 = to_bf16(-0.3280f);

    float line1 = to_bf16(std::fma(c_neg_0_4, diff, c_0_69315));
    float line2 = to_bf16(std::fma(c_neg_0_328, diff, c_0_65715));

    float corr  = std::max(std::max(line1, line2), 0.0f);
    return std::max(to_bf16(min_v - corr), 0.0f);
}

int main() {
    std::cout << "Testing constants in bf16:" << std::endl;
    std::cout << "0.69315f  -> " << to_bf16(0.69315f) << " (0x" << std::hex << fp32_to_bf16(0.69315f) << std::dec << ")" << std::endl;
    std::cout << "-0.4000f  -> " << to_bf16(-0.4000f) << " (0x" << std::hex << fp32_to_bf16(-0.4000f) << std::dec << ")" << std::endl;
    std::cout << "0.65715f  -> " << to_bf16(0.65715f) << " (0x" << std::hex << fp32_to_bf16(0.65715f) << std::dec << ")" << std::endl;
    std::cout << "-0.3280f  -> " << to_bf16(-0.3280f) << " (0x" << std::hex << fp32_to_bf16(-0.3280f) << std::dec << ")" << std::endl;

    // Test a range of diffs
    int diff_count = 0;
    for (int i = 0; i < 1000; ++i) {
        float a = 2.5f + (i * 0.01f);
        float b = 2.5f + ((1000 - i) * 0.01f);
        a = to_bf16(a);
        b = to_bf16(b);

        float out_fp32 = to_bf16(approx_min_star_fp32(a, b));
        float out_bf16 = approx_min_star_exact_bf16(a, b);

        uint16_t u_fp32 = fp32_to_bf16(out_fp32);
        uint16_t u_bf16 = fp32_to_bf16(out_bf16);

        if (u_fp32 != u_bf16) {
            diff_count++;
            if (diff_count <= 10) {
                std::cout << "Diff at a=" << a << ", b=" << b 
                          << " | FP32=0x" << std::hex << u_fp32 << " (" << out_fp32 << ")"
                          << " vs BF16=0x" << u_bf16 << " (" << out_bf16 << ")" << std::dec << std::endl;
            }
        }
    }
    std::cout << "Total differences between FP32 and BF16 models: " << diff_count << " / 1000" << std::endl;

    return 0;
}
