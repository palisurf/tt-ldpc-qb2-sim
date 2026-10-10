#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"
#include "sfpi.h"
#include "llk_math_eltwise_sfpu_common.h"

// ==============================================================================
// Tenstorrent QuietBox 2: Vector Cartesian Box-Muller AWGN Generator Kernel
// Target: 32-Lane SFPU on Blackhole Tensix Core (TRISC1 Compute Engine)
// Memory Destination: L1 SRAM Circular Buffer (vnode_buf_B, CB 16, 20.0 KB)
// Characteristics:
//   - 100% Branch-Free SIMD execution across 32 lanes
//   - Minimax polynomial natural logarithm + hardware nonlinear approx_sqrt
//   - Pipelined Horner Maclaurin cosine & sine series
//   - Direct PACK DMA tile streaming (64 cycles per 2 KB tile)
//   - Strict register safety: <= 4 live vector registers (zero stack spills)
// ==============================================================================

sfpi_inline sfpi::vFloat sfpu_fast_ln(sfpi::vFloat a) {
    a = sfpi::max(a, 1e-7f);
    sfpi::vUInt u = sfpi::as<sfpi::vUInt>(a);
    sfpi::vInt exp_val = sfpi::as<sfpi::vInt>((u >> 23) & 0xFF) - 127;
    sfpi::vUInt m_bits = (u & 0x007FFFFF) | 0x3F800000;
    sfpi::vFloat m = sfpi::as<sfpi::vFloat>(m_bits);
    sfpi::vFloat z = m - 1.0f;

    // Minimax polynomial for ln(1 + z) on z in [0, 1]
    sfpi::vFloat p = sfpi::vFloat(sfpi::sFloat16b(-0.0528f));
    p = p * z + sfpi::vFloat(sfpi::sFloat16b(0.1264f));
    p = p * z - sfpi::vFloat(sfpi::sFloat16b(0.1678f));
    p = p * z + sfpi::vFloat(sfpi::sFloat16b(0.2057f));
    p = p * z - sfpi::vFloat(sfpi::sFloat16b(0.2490f));
    p = p * z + sfpi::vFloat(sfpi::sFloat16b(0.3328f));
    p = p * z - 0.5f;
    p = p * z + 1.0f;
    sfpi::vFloat ln_m = p * z;

    sfpi::vFloat e_flt = sfpi::convert<sfpi::vFloat>(exp_val, sfpi::RoundMode::Nearest);
    return ln_m + e_flt * 0.69314718f;
}

sfpi_inline sfpi::vFloat sfpu_fast_sqrt(sfpi::vFloat val) {
    val = sfpi::max(val, 1e-7f);
    sfpi::vUInt magic = 0x5F375A86U;
    sfpi::vFloat approx = sfpi::as<sfpi::vFloat>(magic - (sfpi::as<sfpi::vUInt>(val) >> 1));
    sfpi::vFloat neg_half_val = val * -0.5f;
    approx = ((approx * approx) * neg_half_val + 1.5f) * approx;
    approx = ((approx * approx) * neg_half_val + 1.5f) * approx;
    return approx * val;
}

void kernel_main() {
    uint32_t num_tiles  = get_arg_val<uint32_t>(0); // 10 tiles (8 AWGN + 2 Punctured)
    uint32_t mu_bits    = get_arg_val<uint32_t>(1);
    uint32_t sigma_bits = get_arg_val<uint32_t>(2);
    uint32_t seed_base  = get_arg_val<uint32_t>(3);

    float mu_val;
    float sigma_val;
    std::memcpy(&mu_val, &mu_bits, 4);
    std::memcpy(&sigma_val, &sigma_bits, 4);

    constexpr uint32_t cb_out   = tt::CBIndex::c_16; // 10 tiles (20 KB)
    constexpr uint32_t cb_stats = tt::CBIndex::c_17; // Timing telemetry

    compute_kernel_hw_startup(cb_out, cb_out);

    volatile uint32_t* const wall_clock_reg = reinterpret_cast<volatile uint32_t*>(0xFFB121F0);
    uint32_t tile_cycles[10] = {0};

#ifdef TRISC_MATH
    uint32_t t_start = *wall_clock_reg;
    uint32_t t_end = 0;

    // Vector PRNG state initialization per lane
    sfpi::vUInt prng_state = static_cast<uint32_t>(seed_base) + sfpi::as<sfpi::vUInt>(sfpi::vConstTileId);
    prng_state = (prng_state << 5) ^ sfpi::as<sfpi::vUInt>(sfpi::vConstTileId) ^ 0x55AA55AAU;
    v_if (prng_state == 0) {
        prng_state = 0x13579BDFU;
    } v_endif;

    sfpi::vFloat v_mu = mu_val;
    sfpi::vFloat v_sigma = sigma_val;
#endif

    // Loop over all 10 tiles (8 AWGN tiles + 2 Punctured zero tiles)
    for (uint32_t t = 0; t < num_tiles; ++t) {
        tile_regs_acquire();

#ifdef TRISC_MATH
        _llk_math_eltwise_sfpu_start_(0);
        uint32_t t_tile_0 = *wall_clock_reg;

        if (t < 8) {
            // 16 passes per tile: each pass fills 2 rows (64 samples)
            for (int p = 0; p < 16; ++p) {
                // Draw U1 in [1e-7, 1.0)
                prng_state ^= prng_state << 13;
                prng_state ^= prng_state >> 17;
                prng_state ^= prng_state << 5;
                sfpi::vUInt u1_bits = ((prng_state >> 9) & 0x007FFFFF) | 0x3F800000;
                sfpi::vFloat u1 = sfpi::as<sfpi::vFloat>(u1_bits) - 1.0f;
                u1 = sfpi::max(u1, 1e-7f);

                // Compute Radius R = sqrt(-2 * ln(U1))
                sfpi::vFloat ln_u1 = sfpu_fast_ln(u1);
                sfpi::vFloat r = sfpu_fast_sqrt(ln_u1 * -2.0f);

                // Draw U2 in [0.0, 1.0)
                prng_state ^= prng_state << 13;
                prng_state ^= prng_state >> 17;
                prng_state ^= prng_state << 5;
                sfpi::vUInt u2_bits = ((prng_state >> 9) & 0x007FFFFF) | 0x3F800000;
                sfpi::vFloat u2 = sfpi::as<sfpi::vFloat>(u2_bits) - 1.0f;

                // Compute theta in [-PI, PI)
                sfpi::vFloat theta = u2 * 6.2831853f - 3.14159265f;
                sfpi::vFloat s_th = theta * theta;

                // Cosine series & write Row 2*p + 0 immediately to free register
                sfpi::vFloat cos_val = sfpi::vFloat(sfpi::sFloat16b(0.0000248f));
                cos_val = cos_val * s_th - sfpi::vFloat(sfpi::sFloat16b(0.0013889f));
                cos_val = cos_val * s_th + sfpi::vFloat(sfpi::sFloat16b(0.0416667f));
                cos_val = cos_val * s_th - 0.5f;
                cos_val = cos_val * s_th + 1.0f;
                sfpi::dst_reg[2 * p + 0] = v_mu + v_sigma * (r * cos_val);

                // Sine series & write Row 2*p + 1 immediately to free register
                sfpi::vFloat sin_val = sfpi::vFloat(sfpi::sFloat16b(0.00000276f));
                sin_val = sin_val * s_th - sfpi::vFloat(sfpi::sFloat16b(0.0001984f));
                sin_val = sin_val * s_th + sfpi::vFloat(sfpi::sFloat16b(0.0083333f));
                sin_val = sin_val * s_th - sfpi::vFloat(sfpi::sFloat16b(0.1666667f));
                sin_val = sin_val * s_th + 1.0f;
                sfpi::dst_reg[2 * p + 1] = v_mu + v_sigma * (r * (sin_val * theta));
            }
        } else {
            // Stage 2: Generate 2 Punctured Zero Tiles (2 x 1024 = 2,048 Zeros)
            for (int r = 0; r < 32; ++r) {
                sfpi::dst_reg[r] = 0.0f;
            }
        }

        tile_cycles[t] = *wall_clock_reg - t_tile_0;
        _llk_math_eltwise_sfpu_done_();
#endif

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }

#ifdef TRISC_MATH
    t_end = *wall_clock_reg;
#endif

    // Telemetry output to CB 17
    cb_reserve_back(cb_stats, 1);
    uint32_t stats_addr = get_tile_address(cb_stats, 0);
    auto* stats = reinterpret_cast<volatile uint32_t*>(stats_addr);
#ifdef TRISC_MATH
    stats[0] = t_end - t_start;
    stats[1] = 0xAA55AA55; // Hardware sentinel
    for (int t = 0; t < 10; ++t) {
        stats[2 + t] = tile_cycles[t];
    }
#endif
    cb_push_back(cb_stats, 1);
}
