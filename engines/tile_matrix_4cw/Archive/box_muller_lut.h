#pragma once
#include <cstdint>
#include <cstring>
#include "ldpc_tile_matrix_core.h"

constexpr uint32_t BM_MIN_EXP = 14;
constexpr uint32_t BM_MAX_EXP = 29;
constexpr uint32_t BM_N_OCTAVES = 16;
constexpr uint32_t BM_N_MANT = 32;

// Fast branchless leading zero count (zero function calls, no libgcc dependencies)
inline uint32_t fast_clz(uint32_t x) {
    uint32_t n = 0;
    if (x <= 0x0000FFFF) { n += 16; x <<= 16; }
    if (x <= 0x00FFFFFF) { n += 8;  x <<= 8;  }
    if (x <= 0x0FFFFFFF) { n += 4;  x <<= 4;  }
    if (x <= 0x3FFFFFFF) { n += 2;  x <<= 2;  }
    if (x <= 0x7FFFFFFF) { n += 1; }
    return n;
}

// One-time initialization of 528 LUT values (2,112 bytes) in L1 SRAM at kernel startup
inline void init_box_muller_lut(float* lut) {
    for (uint32_t e = 0; e < 16; ++e) {
        uint32_t exp = 14 + e;
        float s_base = static_cast<float>(1U << exp);
        float s_step = s_base * (1.0f / 32.0f);
        for (uint32_t m = 0; m <= 32; ++m) {
            float s = s_base + static_cast<float>(m) * s_step;
            float t = s * (1.0f / 1073741824.0f); // s / 2^30
            float ln_t = sfpu_fast_log(t);
            float arg = -2.0f * ln_t / t;
            float scale = (arg > 0.0f) ? sfpu_fast_sqrt(arg) * (1.0f / 32768.0f) : 0.0f;
            lut[e * 33 + m] = scale;
        }
    }
}

// High-speed Box-Muller Gaussian pair generation using L1 LUT (~25 clock cycles/pair)
inline void fast_box_muller_pair(
    const float* lut,
    Xoshiro128Plus& rng,
    float mu_llr,
    float sigma_llr,
    float& llr0,
    float& llr1
) {
    int32_t x1, x2;
    uint32_t s;
    do {
        uint32_t u = rng.next();
        x1 = static_cast<int16_t>(u & 0xFFFF);
        x2 = static_cast<int16_t>(u >> 16);
        s = static_cast<uint32_t>(x1 * x1 + x2 * x2);
    } while (s < (1U << BM_MIN_EXP) || s >= (1U << 30));

    uint32_t lz = fast_clz(s);
    uint32_t exp = 31 - lz;
    if (exp > BM_MAX_EXP) exp = BM_MAX_EXP;
    uint32_t e_idx = exp - BM_MIN_EXP;

    uint32_t m_full = (s << (lz + 1));
    uint32_t m_idx = m_full >> 27;
    uint32_t alpha_u = (m_full >> 11) & 0xFFFF;
    float alpha = static_cast<float>(alpha_u) * (1.0f / 65536.0f);

    const float* row = &lut[e_idx * 33];
    float scale0 = row[m_idx];
    float scale1 = row[m_idx + 1];
    float scale = scale0 + alpha * (scale1 - scale0);

    float z1 = static_cast<float>(x1) * scale;
    float z2 = static_cast<float>(x2) * scale;
    llr0 = mu_llr + sigma_llr * z1;
    llr1 = mu_llr + sigma_llr * z2;
}
