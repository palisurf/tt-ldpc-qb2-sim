#pragma once

#include <stdint.h>
#include <cstring>
#include <cmath>

// ==============================================================================
// 8-Bit Floating-Point Types for LDPC Belief Propagation & Channel Quantization
// ==============================================================================
// 1. FP8 E5M2 ("Bfloat8" / BF8):
//    - 1 Sign bit, 5 Exponent bits (bias 15), 2 Mantissa bits
//    - Dynamic range identical to IEEE FP16 (~10^±5, max representable: 57344.0)
//    - Matches Bfloat16's philosophy of preserving dynamic range over precision.
//
// 2. FP8 E4M3:
//    - 1 Sign bit, 4 Exponent bits (bias 7), 3 Mantissa bits
//    - Higher precision (~1 additional bit of mantissa), max representable: 448.0
// ==============================================================================

inline uint32_t float_to_raw_u32(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

inline float raw_u32_to_float(uint32_t u) {
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// ------------------------------------------------------------------------------
// FP8 E5M2 (Bfloat8) Encoder / Decoder
// ------------------------------------------------------------------------------

// Converts single-precision FP32 to FP8 E5M2 with round-to-nearest-even & clamping
inline uint8_t fp32_to_fp8_e5m2(float f) {
    uint32_t x = float_to_raw_u32(f);
    uint32_t sign = (x >> 31) & 0x01;
    uint32_t exp32 = (x >> 23) & 0xFF;
    uint32_t mant32 = x & 0x007FFFFF;

    // Handle 0.0f
    if ((x & 0x7FFFFFFF) == 0) {
        return static_cast<uint8_t>(sign << 7);
    }

    // Unbiased exponent
    int32_t exp_unbiased = static_cast<int32_t>(exp32) - 127;
    int32_t exp8 = exp_unbiased + 15;

    // Overflow / Saturation to maximum finite value (exp=30, mant=3 -> 57344.0)
    if (exp8 >= 31) {
        return static_cast<uint8_t>((sign << 7) | 0x7B); // Max finite positive/negative
    }

    // Underflow to zero (or subnormal)
    if (exp8 <= 0) {
        if (exp8 < -2) {
            return static_cast<uint8_t>(sign << 7);
        }
        // Subnormal representation
        uint32_t full_mant = mant32 | 0x00800000;
        int32_t shift = 1 - exp8 + 21; // shift to 2 mantissa bits
        uint32_t mant8 = (full_mant >> shift);
        return static_cast<uint8_t>((sign << 7) | (mant8 & 0x03));
    }

    // Normal numbers: Round to nearest even on the 21 discarded mantissa bits
    // Mantissa bits [22:21] are kept, bit 20 is round bit
    uint32_t round_bit = (mant32 >> 20) & 1;
    uint32_t sticky_bits = mant32 & 0x000FFFFF;
    uint32_t mant8 = (mant32 >> 21) & 0x03;

    if (round_bit && (sticky_bits || (mant8 & 1))) {
        mant8++;
        if (mant8 > 3) {
            mant8 = 0;
            exp8++;
            if (exp8 >= 31) {
                return static_cast<uint8_t>((sign << 7) | 0x7B); // Clamp on rounding overflow
            }
        }
    }

    return static_cast<uint8_t>((sign << 7) | ((exp8 & 0x1F) << 2) | (mant8 & 0x03));
}

// Converts FP8 E5M2 back to standard FP32
inline float fp8_e5m2_to_fp32(uint8_t u) {
    uint32_t sign = (u >> 7) & 0x01;
    uint32_t exp8 = (u >> 2) & 0x1F;
    uint32_t mant8 = u & 0x03;

    if (exp8 == 0) {
        if (mant8 == 0) {
            return raw_u32_to_float(sign << 31);
        }
        // Subnormal
        float val = static_cast<float>(mant8) * std::ldexp(1.0f, -14 - 2);
        return sign ? -val : val;
    }

    if (exp8 == 31) {
        // Reserved for Inf/NaN - map to max finite
        exp8 = 30;
        mant8 = 3;
    }

    uint32_t exp32 = (exp8 - 15 + 127) & 0xFF;
    uint32_t mant32 = mant8 << 21;
    uint32_t result_u = (sign << 31) | (exp32 << 23) | mant32;
    return raw_u32_to_float(result_u);
}

// ------------------------------------------------------------------------------
// FP8 E4M3 Encoder / Decoder (Higher precision option)
// ------------------------------------------------------------------------------

inline uint8_t fp32_to_fp8_e4m3(float f) {
    uint32_t x = float_to_raw_u32(f);
    uint32_t sign = (x >> 31) & 0x01;
    uint32_t exp32 = (x >> 23) & 0xFF;
    uint32_t mant32 = x & 0x007FFFFF;

    if ((x & 0x7FFFFFFF) == 0) {
        return static_cast<uint8_t>(sign << 7);
    }

    int32_t exp_unbiased = static_cast<int32_t>(exp32) - 127;
    int32_t exp8 = exp_unbiased + 7;

    // Max finite E4M3 is exp=15, mant=6 (448.0)
    if (exp8 >= 15) {
        return static_cast<uint8_t>((sign << 7) | 0x7E);
    }

    if (exp8 <= 0) {
        if (exp8 < -3) {
            return static_cast<uint8_t>(sign << 7);
        }
        uint32_t full_mant = mant32 | 0x00800000;
        int32_t shift = 1 - exp8 + 20;
        uint32_t mant8 = (full_mant >> shift);
        return static_cast<uint8_t>((sign << 7) | (mant8 & 0x07));
    }

    uint32_t round_bit = (mant32 >> 19) & 1;
    uint32_t sticky_bits = mant32 & 0x0007FFFF;
    uint32_t mant8 = (mant32 >> 20) & 0x07;

    if (round_bit && (sticky_bits || (mant8 & 1))) {
        mant8++;
        if (mant8 > 7) {
            mant8 = 0;
            exp8++;
            if (exp8 >= 15) {
                return static_cast<uint8_t>((sign << 7) | 0x7E);
            }
        }
    }

    return static_cast<uint8_t>((sign << 7) | ((exp8 & 0x0F) << 3) | (mant8 & 0x07));
}

inline float fp8_e4m3_to_fp32(uint8_t u) {
    uint32_t sign = (u >> 7) & 0x01;
    uint32_t exp8 = (u >> 3) & 0x0F;
    uint32_t mant8 = u & 0x07;

    if (exp8 == 0) {
        if (mant8 == 0) {
            return raw_u32_to_float(sign << 31);
        }
        float val = static_cast<float>(mant8) * std::ldexp(1.0f, -6 - 3);
        return sign ? -val : val;
    }

    if (exp8 == 15 && mant8 == 7) {
        exp8 = 15;
        mant8 = 6; // Clamp to max finite
    }

    uint32_t exp32 = (exp8 - 7 + 127) & 0xFF;
    uint32_t mant32 = mant8 << 20;
    uint32_t result_u = (sign << 31) | (exp32 << 23) | mant32;
    return raw_u32_to_float(result_u);
}
