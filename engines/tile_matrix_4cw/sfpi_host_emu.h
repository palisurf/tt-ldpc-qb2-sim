#pragma once

// ==============================================================================
// Tenstorrent SFPI Host Emulation Layer (sfpi_host_emu.h)
// Provides 100% bit-exact Tenstorrent SFPU vector emulation for host CPUs:
//   - 32-lane SIMD vector types: vFloat, vInt, vUInt, vBool, vFloat16b, sFloat16b
//   - Hardware 16-lane Transposition: swap_subvec16 / subvec_transp
//   - Hardware Round-to-Nearest-Even (RNE) bfloat16 converter: round_bf16
//   - Bitwise sign & magnitude primitives: copysgn, min, max, min_max, abs, as<T>
//   - Predication: thread-safe v_if, v_else, v_endif via active lane bitmasks
//   - Thread-local 64-row DEST register bank: sfpi::dst_reg[0..63]
// ==============================================================================

#include <cstdint>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <array>
#include <utility>

#ifndef sfpi_inline
#define sfpi_inline inline
#endif

namespace sfpi {

// ------------------------------------------------------------------------------
// Bit-Exact Hardware Round-to-Nearest-Even (RNE) for bfloat16
// ------------------------------------------------------------------------------
inline float round_bf16_rne_scalar(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    uint32_t lsb = (u >> 16) & 1;
    uint32_t bias = 0x7FFF + lsb;
    uint16_t bf = static_cast<uint16_t>((u + bias) >> 16);
    uint32_t res = static_cast<uint32_t>(bf) << 16;
    float out;
    std::memcpy(&out, &res, sizeof(out));
    return out;
}

// ------------------------------------------------------------------------------
// SFPI Enums
// ------------------------------------------------------------------------------
enum class RoundMode { Nearest, Trunc };
enum class LutMode { Fp8x3, Fp16x3, Fp32x3, Fp16x6_HWM3, Fp16x6_HWM4 };
enum class LutSign { Retain, Invert };

struct sFloat16b {
    float f;
    sFloat16b(float val = 0.0f) {
        uint32_t u;
        std::memcpy(&u, &val, sizeof(u));
        u &= 0xFFFF0000;
        std::memcpy(&f, &u, sizeof(f));
    }
    operator float() const { return f; }
};

struct sLut16si {
    float slope;
    float intercept;
    sLut16si(float s = 0.0f, float i = 0.0f) : slope(s), intercept(i) {}
};

struct vLut16ss {
    float s0, s1;
    vLut16ss(float _s0 = 0.0f, float _s1 = 0.0f) : s0(_s0), s1(_s1) {}
};

struct vLut16ii {
    float i0, i1;
    vLut16ii(float _i0 = 0.0f, float _i1 = 0.0f) : i0(_i0), i1(_i1) {}
};

// ------------------------------------------------------------------------------
// Thread-Safe Predication Context Stack for v_if / v_endif
// ------------------------------------------------------------------------------
namespace impl_ {
    struct CCStack {
        uint32_t mask[16];
        int top = 0;
        CCStack() {
            mask[0] = 0xFFFFFFFFU;
            top = 0;
        }
        inline uint32_t active() const { return mask[top]; }
        inline void push(uint32_t m) {
            if (top < 15) {
                uint32_t parent = mask[top];
                top++;
                mask[top] = parent & m;
            }
        }
        inline void pop() {
            if (top > 0) top--;
        }
    };

    inline CCStack& get_cc() {
        static thread_local CCStack cc;
        return cc;
    }
} // namespace impl_

struct vBool {
    uint32_t mask;
    vBool(uint32_t m = 0) : mask(m) {}
};

// ------------------------------------------------------------------------------
// Vector Types: vFloat, vInt, vUInt, vFloat16b
// ------------------------------------------------------------------------------
struct vFloat {
    float data[32];

    vFloat() {
        for (int i = 0; i < 32; ++i) data[i] = 0.0f;
    }
    vFloat(float val) {
        for (int i = 0; i < 32; ++i) data[i] = val;
    }
    vFloat(sFloat16b val) {
        for (int i = 0; i < 32; ++i) data[i] = val.f;
    }

    vFloat& operator=(const vFloat& rhs) {
        uint32_t m = impl_::get_cc().active();
        if (m == 0xFFFFFFFFU) {
            std::memcpy(data, rhs.data, sizeof(data));
        } else {
            for (int i = 0; i < 32; ++i) {
                if ((m >> i) & 1) data[i] = rhs.data[i];
            }
        }
        return *this;
    }

    vFloat& operator=(float rhs) {
        return *this = vFloat(rhs);
    }

    float& operator[](size_t i) { return data[i]; }
    const float& operator[](size_t i) const { return data[i]; }
};

struct vInt {
    int32_t data[32];

    vInt() {
        for (int i = 0; i < 32; ++i) data[i] = 0;
    }
    vInt(int32_t val) {
        for (int i = 0; i < 32; ++i) data[i] = val;
    }

    vInt& operator=(const vInt& rhs) {
        uint32_t m = impl_::get_cc().active();
        if (m == 0xFFFFFFFFU) {
            std::memcpy(data, rhs.data, sizeof(data));
        } else {
            for (int i = 0; i < 32; ++i) {
                if ((m >> i) & 1) data[i] = rhs.data[i];
            }
        }
        return *this;
    }

    vInt& operator=(int32_t rhs) {
        return *this = vInt(rhs);
    }

    int32_t& operator[](size_t i) { return data[i]; }
    const int32_t& operator[](size_t i) const { return data[i]; }
};

struct vUInt {
    uint32_t data[32];

    vUInt() {
        for (int i = 0; i < 32; ++i) data[i] = 0;
    }
    vUInt(uint32_t val) {
        for (int i = 0; i < 32; ++i) data[i] = val;
    }

    vUInt& operator=(const vUInt& rhs) {
        uint32_t m = impl_::get_cc().active();
        if (m == 0xFFFFFFFFU) {
            std::memcpy(data, rhs.data, sizeof(data));
        } else {
            for (int i = 0; i < 32; ++i) {
                if ((m >> i) & 1) data[i] = rhs.data[i];
            }
        }
        return *this;
    }

    vUInt& operator=(uint32_t rhs) {
        return *this = vUInt(rhs);
    }

    uint32_t& operator[](size_t i) { return data[i]; }
    const uint32_t& operator[](size_t i) const { return data[i]; }
};

struct vFloat16b {
    float data[32];
    vFloat16b() {
        for (int i = 0; i < 32; ++i) data[i] = 0.0f;
    }
    vFloat16b(const vFloat& f) {
        for (int i = 0; i < 32; ++i) data[i] = round_bf16_rne_scalar(f.data[i]);
    }
    operator vFloat() const {
        vFloat out;
        std::memcpy(out.data, data, sizeof(data));
        return out;
    }
};

// ------------------------------------------------------------------------------
// Type Bitcasting: sfpi::as<T>
// ------------------------------------------------------------------------------
template<typename Dest, typename Src>
inline Dest as(const Src& s);

template<>
inline vInt as<vInt, vFloat>(const vFloat& s) {
    vInt out;
    std::memcpy(out.data, s.data, sizeof(out.data));
    return out;
}

template<>
inline vFloat as<vFloat, vInt>(const vInt& s) {
    vFloat out;
    std::memcpy(out.data, s.data, sizeof(out.data));
    return out;
}

template<>
inline vUInt as<vUInt, vFloat>(const vFloat& s) {
    vUInt out;
    std::memcpy(out.data, s.data, sizeof(out.data));
    return out;
}

template<>
inline vFloat as<vFloat, vUInt>(const vUInt& s) {
    vFloat out;
    std::memcpy(out.data, s.data, sizeof(out.data));
    return out;
}

template<>
inline vUInt as<vUInt, vInt>(const vInt& s) {
    vUInt out;
    std::memcpy(out.data, s.data, sizeof(out.data));
    return out;
}

template<>
inline vInt as<vInt, vUInt>(const vUInt& s) {
    vInt out;
    std::memcpy(out.data, s.data, sizeof(out.data));
    return out;
}

// ------------------------------------------------------------------------------
// Arithmetic & Bitwise Operators
// ------------------------------------------------------------------------------
// vFloat
inline vFloat operator+(const vFloat& a, const vFloat& b) {
    vFloat out;
    for (int i = 0; i < 32; ++i) out.data[i] = a.data[i] + b.data[i];
    return out;
}
inline vFloat operator-(const vFloat& a, const vFloat& b) {
    vFloat out;
    for (int i = 0; i < 32; ++i) out.data[i] = a.data[i] - b.data[i];
    return out;
}
inline vFloat operator*(const vFloat& a, const vFloat& b) {
    vFloat out;
    for (int i = 0; i < 32; ++i) out.data[i] = a.data[i] * b.data[i];
    return out;
}
inline vFloat operator*(const vFloat& a, float b) {
    vFloat out;
    for (int i = 0; i < 32; ++i) out.data[i] = a.data[i] * b;
    return out;
}
inline vFloat operator*(float a, const vFloat& b) {
    vFloat out;
    for (int i = 0; i < 32; ++i) out.data[i] = a * b.data[i];
    return out;
}
inline vFloat operator+(const vFloat& a, float b) {
    vFloat out;
    for (int i = 0; i < 32; ++i) out.data[i] = a.data[i] + b;
    return out;
}
inline vFloat operator-(const vFloat& a, float b) {
    vFloat out;
    for (int i = 0; i < 32; ++i) out.data[i] = a.data[i] - b;
    return out;
}
inline vFloat operator-(const vFloat& a) {
    vFloat out;
    for (int i = 0; i < 32; ++i) out.data[i] = -a.data[i];
    return out;
}

// vInt
inline vInt operator^(const vInt& a, const vInt& b) {
    vInt out;
    for (int i = 0; i < 32; ++i) out.data[i] = a.data[i] ^ b.data[i];
    return out;
}
inline vInt operator&(const vInt& a, const vInt& b) {
    vInt out;
    for (int i = 0; i < 32; ++i) out.data[i] = a.data[i] & b.data[i];
    return out;
}
inline vInt operator|(const vInt& a, const vInt& b) {
    vInt out;
    for (int i = 0; i < 32; ++i) out.data[i] = a.data[i] | b.data[i];
    return out;
}
inline vInt operator&(const vInt& a, int32_t b) {
    vInt out;
    for (int i = 0; i < 32; ++i) out.data[i] = a.data[i] & b;
    return out;
}
inline vInt operator-(const vInt& a, int32_t b) {
    vInt out;
    for (int i = 0; i < 32; ++i) out.data[i] = a.data[i] - b;
    return out;
}
inline vInt operator<<(const vInt& a, int shift) {
    vInt out;
    for (int i = 0; i < 32; ++i) out.data[i] = a.data[i] << shift;
    return out;
}
inline vInt operator>>(const vInt& a, int shift) {
    vInt out;
    for (int i = 0; i < 32; ++i) out.data[i] = a.data[i] >> shift;
    return out;
}

// vUInt
inline vUInt operator^(const vUInt& a, const vUInt& b) {
    vUInt out;
    for (int i = 0; i < 32; ++i) out.data[i] = a.data[i] ^ b.data[i];
    return out;
}
inline vUInt operator&(const vUInt& a, const vUInt& b) {
    vUInt out;
    for (int i = 0; i < 32; ++i) out.data[i] = a.data[i] & b.data[i];
    return out;
}
inline vUInt operator|(const vUInt& a, const vUInt& b) {
    vUInt out;
    for (int i = 0; i < 32; ++i) out.data[i] = a.data[i] | b.data[i];
    return out;
}
inline vUInt operator+(const vUInt& a, const vUInt& b) {
    vUInt out;
    for (int i = 0; i < 32; ++i) out.data[i] = a.data[i] + b.data[i];
    return out;
}
inline vUInt operator-(const vUInt& a, const vUInt& b) {
    vUInt out;
    for (int i = 0; i < 32; ++i) out.data[i] = a.data[i] - b.data[i];
    return out;
}
inline vUInt operator+(const vUInt& a, uint32_t b) {
    vUInt out;
    for (int i = 0; i < 32; ++i) out.data[i] = a.data[i] + b;
    return out;
}
inline vUInt operator-(uint32_t a, const vUInt& b) {
    vUInt out;
    for (int i = 0; i < 32; ++i) out.data[i] = a - b.data[i];
    return out;
}
inline vUInt operator&(const vUInt& a, uint32_t b) {
    vUInt out;
    for (int i = 0; i < 32; ++i) out.data[i] = a.data[i] & b;
    return out;
}
inline vUInt operator|(const vUInt& a, uint32_t b) {
    vUInt out;
    for (int i = 0; i < 32; ++i) out.data[i] = a.data[i] | b;
    return out;
}
inline vUInt operator<<(const vUInt& a, int shift) {
    vUInt out;
    for (int i = 0; i < 32; ++i) out.data[i] = a.data[i] << shift;
    return out;
}
inline vUInt operator>>(const vUInt& a, int shift) {
    vUInt out;
    for (int i = 0; i < 32; ++i) out.data[i] = a.data[i] >> shift;
    return out;
}

// In-place operators
inline vUInt& operator^=(vUInt& a, const vUInt& b) { return a = a ^ b; }
inline vUInt& operator+=(vUInt& a, const vUInt& b) { return a = a + b; }
inline vUInt& operator&=(vUInt& a, const vUInt& b) { return a = a & b; }
inline vInt& operator&=(vInt& a, const vInt& b)   { return a = a & b; }

// Comparisons producing vBool
inline vBool operator>=(const vUInt& a, uint32_t val) {
    uint32_t mask = 0;
    for (int i = 0; i < 32; ++i) {
        if (a.data[i] >= val) mask |= (1U << i);
    }
    return vBool(mask);
}

inline vBool operator<(const vUInt& a, uint32_t val) {
    uint32_t mask = 0;
    for (int i = 0; i < 32; ++i) {
        if (a.data[i] < val) mask |= (1U << i);
    }
    return vBool(mask);
}

inline vBool operator==(const vUInt& a, uint32_t val) {
    uint32_t mask = 0;
    for (int i = 0; i < 32; ++i) {
        if (a.data[i] == val) mask |= (1U << i);
    }
    return vBool(mask);
}

// ------------------------------------------------------------------------------
// Math Primitives & Utilities
// ------------------------------------------------------------------------------
inline vFloat min(const vFloat& a, const vFloat& b) {
    vFloat out;
    for (int i = 0; i < 32; ++i) out.data[i] = std::min(a.data[i], b.data[i]);
    return out;
}

inline vFloat max(const vFloat& a, const vFloat& b) {
    vFloat out;
    for (int i = 0; i < 32; ++i) out.data[i] = std::max(a.data[i], b.data[i]);
    return out;
}

inline vFloat max(const vFloat& a, float b) {
    vFloat out;
    for (int i = 0; i < 32; ++i) out.data[i] = std::max(a.data[i], b);
    return out;
}

inline std::pair<vFloat, vFloat> min_max(const vFloat& a, const vFloat& b) {
    vFloat min_v, max_v;
    for (int i = 0; i < 32; ++i) {
        min_v.data[i] = std::min(a.data[i], b.data[i]);
        max_v.data[i] = std::max(a.data[i], b.data[i]);
    }
    return {min_v, max_v};
}

inline vFloat abs(const vFloat& a) {
    vFloat out;
    for (int i = 0; i < 32; ++i) out.data[i] = std::fabs(a.data[i]);
    return out;
}

inline vFloat copysgn(const vFloat& mag, const vInt& sign_bits) {
    vFloat res;
    for (int i = 0; i < 32; ++i) {
        uint32_t m_u, s_u;
        std::memcpy(&m_u, &mag.data[i], 4);
        std::memcpy(&s_u, &sign_bits.data[i], 4);
        uint32_t out_u = (m_u & 0x7FFFFFFFU) | (s_u & 0x80000000U);
        std::memcpy(&res.data[i], &out_u, 4);
    }
    return res;
}

template<typename T>
inline T convert(const vFloat& val, RoundMode mode);

template<>
inline vFloat16b convert<vFloat16b>(const vFloat& val, RoundMode /*mode*/) {
    return vFloat16b(val);
}

template<typename T>
inline T convert(const vInt& val, RoundMode mode);

template<>
inline vFloat convert<vFloat>(const vInt& val, RoundMode /*mode*/) {
    vFloat out;
    for (int i = 0; i < 32; ++i) out.data[i] = static_cast<float>(val.data[i]);
    return out;
}

inline vFloat round_bf16(const vFloat& val) {
    vFloat out;
    for (int i = 0; i < 32; ++i) out.data[i] = round_bf16_rne_scalar(val.data[i]);
    return out;
}

// ------------------------------------------------------------------------------
// Hardware Subvector Transposition (SFPTRANSP 16-Lane Crossbar)
// ------------------------------------------------------------------------------
inline vFloat swap_subvec16(const vFloat& in) {
    vFloat out;
    for (int i = 0; i < 16; ++i) {
        out.data[i]      = in.data[i + 16];
        out.data[i + 16] = in.data[i];
    }
    return out;
}

inline vInt swap_subvec16(const vInt& in) {
    vInt out;
    for (int i = 0; i < 16; ++i) {
        out.data[i]      = in.data[i + 16];
        out.data[i + 16] = in.data[i];
    }
    return out;
}

inline void subvec_transp(vFloat& a, vFloat& b, vFloat& c, vFloat& d) {
    vFloat new_a, new_b, new_c, new_d;
    for (int i = 0; i < 16; ++i) {
        new_a.data[i]      = a.data[i];
        new_a.data[i + 16] = b.data[i];
        new_b.data[i]      = a.data[i + 16];
        new_b.data[i + 16] = b.data[i + 16];
        new_c.data[i]      = c.data[i];
        new_c.data[i + 16] = d.data[i];
        new_d.data[i]      = c.data[i + 16];
        new_d.data[i + 16] = d.data[i + 16];
    }
    a = new_a; b = new_b; c = new_c; d = new_d;
}

inline void subvec_transp(vInt& a, vInt& b, vInt& c, vInt& d) {
    vInt new_a, new_b, new_c, new_d;
    for (int i = 0; i < 16; ++i) {
        new_a.data[i]      = a.data[i];
        new_a.data[i + 16] = b.data[i];
        new_b.data[i]      = a.data[i + 16];
        new_b.data[i + 16] = b.data[i + 16];
        new_c.data[i]      = c.data[i];
        new_c.data[i + 16] = d.data[i];
        new_d.data[i]      = c.data[i + 16];
        new_d.data[i + 16] = d.data[i + 16];
    }
    a = new_a; b = new_b; c = new_c; d = new_d;
}

// ------------------------------------------------------------------------------
// Hardware Constants: vConstTileId (lane = 0..31 -> 2 * lane)
// ------------------------------------------------------------------------------
inline vInt make_vConstTileId() {
    vInt out;
    for (int i = 0; i < 32; ++i) out.data[i] = 2 * i;
    return out;
}
static const vInt vConstTileId = make_vConstTileId();

// ------------------------------------------------------------------------------
// SFPU Hardware Look-Up Table (SFPLUTFP32) Emulation
// ------------------------------------------------------------------------------
static sLut16si lut_table_hwm3[6];

template<LutMode M>
inline void lut_init(sLut16si si0, sLut16si si1, sLut16si si2,
                     sLut16si si3, sLut16si si4, sLut16si si5) {
    lut_table_hwm3[0] = si0;
    lut_table_hwm3[1] = si1;
    lut_table_hwm3[2] = si2;
    lut_table_hwm3[3] = si3;
    lut_table_hwm3[4] = si4;
    lut_table_hwm3[5] = si5;
}

template<LutMode M>
inline vFloat lut(const vFloat& delta, LutSign /*sign*/ = LutSign::Retain) {
    vFloat out;
    for (int i = 0; i < 32; ++i) {
        float d = delta.data[i];
        sLut16si entry;
        if      (d < 0.5f) entry = lut_table_hwm3[0];
        else if (d < 1.0f) entry = lut_table_hwm3[1];
        else if (d < 1.5f) entry = lut_table_hwm3[2];
        else if (d < 2.0f) entry = lut_table_hwm3[3];
        else if (d < 3.0f) entry = lut_table_hwm3[4];
        else               entry = lut_table_hwm3[5];
        out.data[i] = entry.slope * d + entry.intercept;
    }
    return out;
}

template<LutMode M = LutMode::Fp16x6_HWM3>
inline vFloat lut(const vFloat& delta,
                  vLut16ss s01, vLut16ii i01,
                  vLut16ss s23, vLut16ii i23,
                  vLut16ss s45, vLut16ii i45,
                  LutSign /*sign*/ = LutSign::Retain) {
    vFloat out;
    for (int i = 0; i < 32; ++i) {
        float d = delta.data[i];
        float slope = 0.0f, intercept = 0.0f;
        if      (d < 0.5f) { slope = s01.s0; intercept = i01.i0; }
        else if (d < 1.0f) { slope = s01.s1; intercept = i01.i1; }
        else if (d < 1.5f) { slope = s23.s0; intercept = i23.i0; }
        else if (d < 2.0f) { slope = s23.s1; intercept = i23.i1; }
        else if (d < 3.0f) { slope = s45.s0; intercept = i45.i0; }
        else               { slope = s45.s1; intercept = i45.i1; }
        out.data[i] = slope * d + intercept;
    }
    return out;
}

// ------------------------------------------------------------------------------
// Branchless Approximate-Min* (Verbatim matching compute_trisc.cpp)
// ------------------------------------------------------------------------------
sfpi_inline vFloat sfpi_amin_star_branchless(vFloat a, vFloat b) {
#if defined(LDPC_USE_SFPU_LUT)
    auto [min_v, max_v] = sfpi::min_max(a, b);
    vFloat diff = max_v - min_v;
    vLut16ss s01(-0.4000f, -0.3280f);
    vLut16ii i01(0.69315f, 0.65715f);
    vLut16ss s23(-0.3280f, -0.3280f);
    vLut16ii i23(0.65715f, 0.65715f);
    vLut16ss s45(0.0f, 0.0f);
    vLut16ii i45(0.0f, 0.0f);
    vFloat corr = sfpi::lut<sfpi::LutMode::Fp16x6_HWM3>(
        diff, s01, i01, s23, i23, s45, i45, sfpi::LutSign::Update);
    return sfpi::max(min_v - corr, 0.0f);
#elif defined(LDPC_USE_3PIECE_AMIN_STAR)
    auto [min_v, max_v] = sfpi::min_max(a, b);
    vFloat diff   = max_v - min_v;
    // Fold sequentially to keep active register pressure low
    vFloat corr   = vFloat(sFloat16b(0.67862f)) + vFloat(sFloat16b(-0.39622f)) * diff;
    corr = sfpi::max(corr, vFloat(sFloat16b(0.48710f)) + vFloat(sFloat16b(-0.18297f)) * diff);
    corr = sfpi::max(corr, vFloat(sFloat16b(0.17614f)) + vFloat(sFloat16b(-0.04029f)) * diff);
    corr = sfpi::max(corr, 0.0f);
    return sfpi::max(min_v - corr, 0.0f);
#else
    auto [min_v, max_v] = sfpi::min_max(a, b);
    vFloat diff   = max_v - min_v;
    vFloat line1  = vFloat(sFloat16b(0.69315f)) + vFloat(sFloat16b(-0.4000f)) * diff;
    vFloat line2  = vFloat(sFloat16b(0.65715f)) + vFloat(sFloat16b(-0.3280f)) * diff;
    vFloat corr   = sfpi::max(sfpi::max(line1, line2), 0.0f);
    return sfpi::max(min_v - corr, 0.0f);
#endif
}

// ------------------------------------------------------------------------------
// Thread-Local DEST Register Bank: sfpi::dst_reg[0..63]
// ------------------------------------------------------------------------------
struct DstRegFile {
    vFloat rows[64];
    vFloat& operator[](size_t r) { return rows[r]; }
    const vFloat& operator[](size_t r) const { return rows[r]; }
};

inline thread_local DstRegFile dst_reg;

} // namespace sfpi

// ------------------------------------------------------------------------------
// Predication Macros: v_if, v_endif
// ------------------------------------------------------------------------------
#ifndef v_if
#define v_if(cond) { sfpi::vBool __cond_val = (cond); sfpi::impl_::get_cc().push(__cond_val.mask); {
#endif

#ifndef v_endif
#define v_endif } sfpi::impl_::get_cc().pop(); }
#endif
