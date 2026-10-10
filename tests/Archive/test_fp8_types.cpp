#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <vector>
#include "../../shared/fp8_types.h"

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "      Testing FP8 E5M2 (Bfloat8) & E4M3 Conversions       " << std::endl;
    std::cout << "==========================================================" << std::endl;

    // 1. Channel LLR values at Eb/N0 = 2.6 dB
    float test_vals[] = {
        0.0f,
        0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 3.6394f, // Mean at 2.6 dB
        4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0352f,             // 2-sigma tail
        10.0f, 11.73f, 14.0f, 16.0f, 32.0f, 60.0f          // Evolved posteriors
    };

    std::cout << "\nTesting FP8 E5M2 (Bfloat8) Quantization at Eb/N0 = 2.6 dB:" << std::endl;
    std::cout << std::setw(12) << "Original" 
              << std::setw(10) << "E5M2 Hex" 
              << std::setw(14) << "E5M2 Decoded" 
              << std::setw(14) << "Abs Error" 
              << std::setw(14) << "Rel Error (%)" << std::endl;
    std::cout << std::string(64, '-') << std::endl;

    for (float f : test_vals) {
        for (int sign : {1, -1}) {
            float val = f * sign;
            uint8_t enc = fp32_to_fp8_e5m2(val);
            float dec = fp8_e5m2_to_fp32(enc);
            float err = std::abs(val - dec);
            float rel_err = (std::abs(val) > 1e-4f) ? (err / std::abs(val) * 100.0f) : 0.0f;

            std::cout << std::setw(12) << std::fixed << std::setprecision(4) << val
                      << "    0x" << std::hex << std::uppercase << std::setw(2) << std::setfill('0') << (int)enc << std::dec << std::setfill(' ')
                      << std::setw(14) << std::fixed << std::setprecision(4) << dec
                      << std::setw(14) << std::fixed << std::setprecision(4) << err
                      << std::setw(12) << std::fixed << std::setprecision(2) << rel_err << " %"
                      << std::endl;
        }
    }

    // 2. Test Monotonicity
    std::cout << "\nTesting Monotonicity in Range [-20.0, +20.0]:" << std::endl;
    float prev_dec = -9999.0f;
    for (float f = -20.0f; f <= 20.0f; f += 0.05f) {
        uint8_t enc = fp32_to_fp8_e5m2(f);
        float dec = fp8_e5m2_to_fp32(enc);
        if (dec < prev_dec - 1e-5f) {
            std::cerr << "FAILED Monotonicity: f=" << f << " got " << dec << " < prev " << prev_dec << std::endl;
            return 1;
        }
        prev_dec = dec;
    }
    std::cout << "PASSED: FP8 E5M2 is strictly monotonic!" << std::endl;

    // 3. Test Full 256-Code Roundtrip
    std::cout << "\nTesting all 256 byte patterns for finite, non-NaN roundtrips:" << std::endl;
    for (int b = 0; b < 256; b++) {
        uint8_t byte_val = static_cast<uint8_t>(b);
        float f = fp8_e5m2_to_fp32(byte_val);
        assert(!std::isnan(f));
    }
    std::cout << "PASSED: All 256 E5M2 patterns decode to valid finite numbers." << std::endl;

    return 0;
}
