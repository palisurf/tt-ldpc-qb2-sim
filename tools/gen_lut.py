import numpy as np

MIN_EXP = 14
MAX_EXP = 29
N_OCTAVES = MAX_EXP - MIN_EXP + 1
N_MANT = 32

lut = np.zeros((N_OCTAVES, N_MANT + 1), dtype=np.float32)
for e_idx, exp in enumerate(range(MIN_EXP, MAX_EXP + 1)):
    for mant in range(N_MANT + 1):
        s = (1 << exp) + (mant * (1 << exp)) / float(N_MANT)
        t = s / float(1 << 30)
        lut[e_idx, mant] = np.sqrt(-2.0 * np.log(t) / t) / 32768.0

with open('engines/tile_matrix_4cw/box_muller_lut.h', 'w') as f:
    f.write('#pragma once\n')
    f.write('#include <cstdint>\n')
    f.write('#include <cstring>\n')
    f.write('#include "ldpc_tile_matrix_core.h"\n\n')
    f.write('constexpr uint32_t BM_MIN_EXP = 14;\n')
    f.write('constexpr uint32_t BM_MAX_EXP = 29;\n')
    f.write('constexpr uint32_t BM_N_OCTAVES = 16;\n')
    f.write('constexpr uint32_t BM_N_MANT = 32;\n\n')
    f.write('static const float g_box_muller_lut[16 * 33] = {\n')
    for row in range(N_OCTAVES):
        f.write(f'    // Octave exp = {MIN_EXP + row}\n    ')
        for col in range(N_MANT + 1):
            f.write(f'{lut[row, col]:.8e}f, ')
            if (col + 1) % 4 == 0:
                f.write('\n    ')
        f.write('\n')
    f.write('};\n\n')
    f.write('inline uint32_t fast_clz(uint32_t x) {\n')
    f.write('    uint32_t n = 0;\n')
    f.write('    if (x <= 0x0000FFFF) { n += 16; x <<= 16; }\n')
    f.write('    if (x <= 0x00FFFFFF) { n += 8;  x <<= 8;  }\n')
    f.write('    if (x <= 0x0FFFFFFF) { n += 4;  x <<= 4;  }\n')
    f.write('    if (x <= 0x3FFFFFFF) { n += 2;  x <<= 2;  }\n')
    f.write('    if (x <= 0x7FFFFFFF) { n += 1; }\n')
    f.write('    return n;\n')
    f.write('}\n\n')
    f.write('inline void fast_box_muller_pair(\n')
    f.write('    Xoshiro128Plus& rng,\n')
    f.write('    float mu_llr,\n')
    f.write('    float sigma_llr,\n')
    f.write('    float& llr0,\n')
    f.write('    float& llr1\n')
    f.write(') {\n')
    f.write('    int32_t x1, x2;\n')
    f.write('    uint32_t s;\n')
    f.write('    do {\n')
    f.write('        uint32_t u = rng.next();\n')
    f.write('        x1 = static_cast<int16_t>(u & 0xFFFF);\n')
    f.write('        x2 = static_cast<int16_t>(u >> 16);\n')
    f.write('        s = static_cast<uint32_t>(x1 * x1 + x2 * x2);\n')
    f.write('    } while (s < (1U << BM_MIN_EXP) || s >= (1U << 30));\n\n')
    f.write('    uint32_t lz = fast_clz(s);\n')
    f.write('    uint32_t exp = 31 - lz;\n')
    f.write('    if (exp > BM_MAX_EXP) exp = BM_MAX_EXP;\n')
    f.write('    uint32_t e_idx = exp - BM_MIN_EXP;\n\n')
    f.write('    uint32_t m_full = (s << (lz + 1));\n')
    f.write('    uint32_t m_idx = m_full >> 27;\n')
    f.write('    uint32_t alpha_u = (m_full >> 11) & 0xFFFF;\n')
    f.write('    float alpha = static_cast<float>(alpha_u) * (1.0f / 65536.0f);\n\n')
    f.write('    const float* row = &g_box_muller_lut[e_idx * 33];\n')
    f.write('    float scale0 = row[m_idx];\n')
    f.write('    float scale1 = row[m_idx + 1];\n')
    f.write('    float scale = scale0 + alpha * (scale1 - scale0);\n\n')
    f.write('    float z1 = static_cast<float>(x1) * scale;\n')
    f.write('    float z2 = static_cast<float>(x2) * scale;\n')
    f.write('    llr0 = mu_llr + sigma_llr * z1;\n')
    f.write('    llr1 = mu_llr + sigma_llr * z2;\n')
    f.write('}\n')

print('Generated box_muller_lut.h cleanly.')
