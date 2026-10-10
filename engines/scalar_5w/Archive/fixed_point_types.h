#pragma once

#include <stdint.h>
#include <cmath>
#include <algorithm>

// ==============================================================================
// Fixed-Point Data Formats for LDPC Belief Propagation
// ==============================================================================
// 1. Signed Q4.3 (8-bit integer):
//    - 1 Sign bit, 4 Integer bits, 3 Fractional bits
//    - Scale: 2^3 = 8.0, Step size: 0.125
//    - Range: [-16.0, +15.875] (integer values -128 to +127)
//    - Ideal for: Variable node channel LLRs and evolving posteriors
//
// 2. Signed Q3.4 (8-bit integer):
//    - 1 Sign bit, 3 Integer bits, 4 Fractional bits
//    - Scale: 2^4 = 16.0, Step size: 0.0625
//    - Range: [-8.0, +7.9375] (integer values -128 to +127)
//    - Ideal for: Check node edge updates (r_m,d)
// ==============================================================================

// ------------------------------------------------------------------------------
// Q4.3 Helpers (Scale = 8.0)
// ------------------------------------------------------------------------------
inline int8_t float_to_q43(float f) {
    float scaled = f * 8.0f;
    int32_t rounded = (scaled >= 0.0f) ? static_cast<int32_t>(scaled + 0.5f) : static_cast<int32_t>(scaled - 0.5f);
    if (rounded > 127) rounded = 127;
    else if (rounded < -128) rounded = -128;
    return static_cast<int8_t>(rounded);
}

inline float q43_to_float(int8_t q) {
    return static_cast<float>(q) * 0.125f; // 1.0 / 8.0
}

// ------------------------------------------------------------------------------
// Q3.4 Helpers (Scale = 16.0)
// ------------------------------------------------------------------------------
inline int8_t float_to_q34(float f) {
    float scaled = f * 16.0f;
    int32_t rounded = (scaled >= 0.0f) ? static_cast<int32_t>(scaled + 0.5f) : static_cast<int32_t>(scaled - 0.5f);
    if (rounded > 127) rounded = 127;
    else if (rounded < -128) rounded = -128;
    return static_cast<int8_t>(rounded);
}

inline float q34_to_float(int8_t q) {
    return static_cast<float>(q) * 0.0625f; // 1.0 / 16.0
}

// ------------------------------------------------------------------------------
// Saturating Integer Arithmetic Helpers
// ------------------------------------------------------------------------------
inline int8_t sat_add_i8(int8_t a, int8_t b) {
    int32_t res = static_cast<int32_t>(a) + static_cast<int32_t>(b);
    if (res > 127) return 127;
    if (res < -128) return -128;
    return static_cast<int8_t>(res);
}

inline int8_t sat_sub_i8(int8_t a, int8_t b) {
    int32_t res = static_cast<int32_t>(a) - static_cast<int32_t>(b);
    if (res > 127) return 127;
    if (res < -128) return -128;
    return static_cast<int8_t>(res);
}
