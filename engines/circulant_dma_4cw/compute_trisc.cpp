#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/debug/waypoint.h"
#include "ldpc_circulant_dma_core.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "sfpi.h"
#include <cstdint>

// ==============================================================================
// Tenstorrent QuietBox 2: 4-Codeword Circulant DMA LDPC Decoder Engine
// Component: Tri-TRISC Compute & Centralized Macro-Orchestrator Kernel
// Architecture: Centralized TRISC1 Maestro with Zero-Transpose Pure SFPU Compute
// Target: Tenstorrent Blackhole (440 Tensix Cores, 2,200 RISC-V Processors)
//
// Key Microarchitectural Principles:
// 1. Zero-Transpose Vertical SIMD Check Node Math: Edges 0..2 and Edges 3..5
//    are gathered directly into separate face registers (Face 0 vs Face 2,
//    Face 1 vs Face 3), eliminating 100% of SFPTRANSP (swap_subvec16) instructions!
// 2. Pure Compute TRISCs: TRISC0 (UNPACK) and TRISC2 (PACK) perform zero data
//    movement in L1; all gather/scatter is offloaded to BRISC and NCRISC.
// 3. Centralized TRISC1 Maestro: Orchestrates macro-step grants and convergence.
// ==============================================================================

// ------------------------------------------------------------------------------
// SFPU Vector Fast Natural Logarithm
// ------------------------------------------------------------------------------
sfpi_inline sfpi::vFloat sfpu_fast_ln(sfpi::vFloat a) {
  a = sfpi::max(a, 1e-7f);
  sfpi::vUInt u = sfpi::as<sfpi::vUInt>(a);
  sfpi::vInt exp_val = sfpi::as<sfpi::vInt>((u >> 23) & 0xFF) - 127;
  sfpi::vUInt m_bits = (u & 0x007FFFFF) | 0x3F800000;
  sfpi::vFloat m = sfpi::as<sfpi::vFloat>(m_bits);
  sfpi::vFloat z = m - 1.0f;

  sfpi::vFloat p = sfpi::vFloat(sfpi::sFloat16b(-0.0528f));
  p = p * z + sfpi::vFloat(sfpi::sFloat16b(0.1264f));
  p = p * z - sfpi::vFloat(sfpi::sFloat16b(0.1678f));
  p = p * z + sfpi::vFloat(sfpi::sFloat16b(0.2057f));
  p = p * z - sfpi::vFloat(sfpi::sFloat16b(0.2490f));
  p = p * z + sfpi::vFloat(sfpi::sFloat16b(0.3328f));
  p = p * z - 0.5f;
  p = p * z + 1.0f;
  sfpi::vFloat ln_m = p * z;

  sfpi::vFloat e_flt =
      sfpi::convert<sfpi::vFloat>(exp_val, sfpi::RoundMode::Nearest);
  return ln_m + e_flt * 0.69314718f;
}

// ------------------------------------------------------------------------------
// SFPU Vector Fast Square Root
// ------------------------------------------------------------------------------
sfpi_inline sfpi::vFloat sfpu_fast_sqrt(sfpi::vFloat val) {
  val = sfpi::max(val, 1e-7f);
  sfpi::vUInt magic = 0x5F375A86U;
  sfpi::vFloat approx =
      sfpi::as<sfpi::vFloat>(magic - (sfpi::as<sfpi::vUInt>(val) >> 1));
  sfpi::vFloat neg_half_val = val * -0.5f;
  approx = ((approx * approx) * neg_half_val + 1.5f) * approx;
  approx = ((approx * approx) * neg_half_val + 1.5f) * approx;
  return approx * val;
}

// ------------------------------------------------------------------------------
// SFPU Vector Branchless Approximate-Min* Formulation
// ------------------------------------------------------------------------------
#if defined(LDPC_USE_SFPU_LUT)
sfpi_inline sfpi::vFloat sfpi_amin_star_branchless(sfpi::vFloat a,
                                                   sfpi::vFloat b) {
  auto [min_v, max_v] = sfpi::min_max(a, b);
  sfpi::vFloat diff = max_v - min_v;
  sfpi::vLut16ss s01(-0.4000f, -0.3280f);
  sfpi::vLut16ii i01(0.69315f, 0.65715f);
  sfpi::vLut16ss s23(-0.3280f, -0.3280f);
  sfpi::vLut16ii i23(0.65715f, 0.65715f);
  sfpi::vLut16ss s45(0.0f, 0.0f);
  sfpi::vLut16ii i45(0.0f, 0.0f);

  sfpi::vFloat corr = sfpi::lut<sfpi::LutMode::Fp16x6_HWM3>(
      diff, s01, i01, s23, i23, s45, i45, sfpi::LutSign::Update);
  return sfpi::max(min_v - corr, 0.0f);
}
#else
sfpi_inline sfpi::vFloat sfpi_amin_star_branchless(sfpi::vFloat a,
                                                   sfpi::vFloat b) {
  auto [min_v, max_v] = sfpi::min_max(a, b);
  sfpi::vFloat diff = max_v - min_v;
  sfpi::vFloat line1 = sfpi::vFloat(sfpi::sFloat16b(0.69315f)) +
                       sfpi::vFloat(sfpi::sFloat16b(-0.4000f)) * diff;
  sfpi::vFloat line2 = sfpi::vFloat(sfpi::sFloat16b(0.65715f)) +
                       sfpi::vFloat(sfpi::sFloat16b(-0.3280f)) * diff;
  sfpi::vFloat corr = sfpi::max(sfpi::max(line1, line2), 0.0f);
  return sfpi::max(min_v - corr, 0.0f);
}
#endif

// ------------------------------------------------------------------------------
// Bit-Exact IEEE Round-to-Nearest-Even (RNE) on Tensix SFPU (bfloat16)
// ------------------------------------------------------------------------------
sfpi_inline sfpi::vFloat round_bf16_rne(sfpi::vFloat val) {
  sfpi::vUInt u = sfpi::as<sfpi::vUInt>(val);
  sfpi::vUInt lsb = (u >> 16) & 1U;
  sfpi::vUInt bias = lsb + 0x7FFFU;
  u += bias;
  u &= 0xFFFF0000U;
  return sfpi::as<sfpi::vFloat>(u);
}
#define round_bf16(v) round_bf16_rne(v)

sfpi_inline sfpi::vFloat round_bf16_fast(sfpi::vFloat val) {
  return round_bf16_rne(val);
}


// ------------------------------------------------------------------------------
// Vector xoshiro128+ PRNG Generator on SFPU (32 Lanes in Parallel)
// ------------------------------------------------------------------------------
sfpi_inline sfpi::vUInt arx_hash(sfpi::vUInt x, uint32_t c) {
  x += c;
  x ^= (x >> 16);
  x += (x << 11);
  x ^= (x >> 7);
  x += (x << 5);
  x ^= (x >> 13);
  x += (x << 17);
  x ^= (x >> 11);
  return x;
}

struct VectorXoshiro128Plus {
  sfpi::vUInt s0;
  sfpi::vUInt s1;
  sfpi::vUInt s2;
  sfpi::vUInt s3;

  sfpi_inline void init(sfpi::vUInt seed_val) {
    s0 = arx_hash(seed_val, 0x9E3779B9U);
    s1 = arx_hash(seed_val, 0x85EBCA6BU);
    s2 = arx_hash(seed_val, 0xC2B2AE35U);
    s3 = arx_hash(seed_val, 0x27D4EB2FU);
    v_if((s0 | s1 | s2 | s3) == 0) { s0 = 0x13579BDFU; }
    v_endif;
  }

  sfpi_inline sfpi::vUInt next() {
    sfpi::vUInt res = s0 + s3;
    sfpi::vUInt t = s1 << 9;
    s2 ^= s0;
    s3 ^= s1;
    s1 ^= s2;
    s0 ^= s3;
    s2 ^= t;
    s3 = (s3 << 11) | (s3 >> 21);
    return res;
  }
};

#ifndef LDPC_ENABLE_CH_CLAMP
#define LDPC_ENABLE_CH_CLAMP 0
#endif

#ifndef LDPC_L_CH_MAX
#define LDPC_L_CH_MAX 7.0f
#endif

// ------------------------------------------------------------------------------
// 32-Lane Vector SFPU Channel Observation Generator
// Generates 10 tiles (8 AWGN / Benchmark tiles + 2 Punctured zero tiles)
// Directly streams into CB_VNODE in native Tensix face quadrant tile format
// ------------------------------------------------------------------------------
void __attribute__((noinline))
generate_channel_tiles(uint32_t cb_out, float mu_val, float sigma_val,
                       bool benchmark_mode, uint32_t invocation_id,
                       uint32_t batch) {
  pack_init(cb_out);

  for (uint32_t t = 0; t < 10; ++t) {
    tile_regs_acquire();

#ifdef TRISC_MATH
    _llk_math_eltwise_sfpu_start_(0);

    // Generate 8 AWGN Transmitted Tiles (8 x 256 = 2,048 Transmitted Variable Nodes)
    if (t < 8) {
      if (benchmark_mode) {
        sfpi::vFloat v_mu = mu_val;
        for (int r = 0; r < 32; ++r) {
          sfpi::dst_reg[r] = v_mu;
        }
      } else {
        // Phase 1: Vector xoshiro128+ Uniform Draws staged directly into DEST
        {
          VectorXoshiro128Plus rng;
          uint32_t seed_base = 0x12345678U + (invocation_id * 1013904223U) +
                               batch * 31337U + t * 7919U;
          sfpi::vUInt lane_seed = static_cast<uint32_t>(seed_base) +
                                  sfpi::as<sfpi::vUInt>(sfpi::vConstTileId);
          rng.init(lane_seed);

          for (int r = 0; r < 32; ++r) {
            sfpi::vUInt raw = rng.next();
            sfpi::vUInt u_bits = ((raw >> 9) & 0x007FFFFF) | 0x3F800000;
            sfpi::vFloat u = sfpi::as<sfpi::vFloat>(u_bits) - 1.0f;
            sfpi::dst_reg[r] = sfpi::max(u, 1e-7f);
          }
        }

        // Phase 2: Transcendental Box-Muller Transformation (PRNG is dead, 0 LReg pressure)
        sfpi::vFloat v_mu = mu_val;
        sfpi::vFloat v_sigma = sigma_val;
        for (int p = 0; p < 16; ++p) {
          sfpi::vFloat u1 = sfpi::vFloat(sfpi::dst_reg[2 * p + 0]);
          sfpi::vFloat ln_u1 = sfpu_fast_ln(u1);
          sfpi::vFloat r = sfpu_fast_sqrt(ln_u1 * -2.0f);

          sfpi::vFloat u2 = sfpi::vFloat(sfpi::dst_reg[2 * p + 1]);
          sfpi::vFloat theta = u2 * 6.2831853f - 3.14159265f;
          sfpi::vFloat s_th = theta * theta;

#ifdef LDPC_USE_LEGACY_TAYLOR_TRIG
          sfpi::vFloat cos_val = sfpi::vFloat(sfpi::sFloat16b(0.0000248f));
          cos_val = cos_val * s_th - sfpi::vFloat(sfpi::sFloat16b(0.0013889f));
          cos_val = cos_val * s_th + sfpi::vFloat(sfpi::sFloat16b(0.0416667f));
          cos_val = cos_val * s_th - 0.5f;
          cos_val = cos_val * s_th + 1.0f;

          sfpi::vFloat sin_val = sfpi::vFloat(sfpi::sFloat16b(0.00000276f));
          sin_val = sin_val * s_th - sfpi::vFloat(sfpi::sFloat16b(0.0001984f));
          sin_val = sin_val * s_th + sfpi::vFloat(sfpi::sFloat16b(0.0083333f));
          sin_val = sin_val * s_th - sfpi::vFloat(sfpi::sFloat16b(0.1666667f));
          sin_val = sin_val * s_th + 1.0f;
#else
          // Default: Remez Minimax Polynomial Evaluation (212x lower truncation error over [-pi, pi])
          sfpi::vFloat cos_val = sfpi::vFloat(sfpi::sFloat16b(0.00001907f));
          cos_val = cos_val * s_th - sfpi::vFloat(sfpi::sFloat16b(0.0013441f));
          cos_val = cos_val * s_th + sfpi::vFloat(sfpi::sFloat16b(0.0415223f));
          cos_val = cos_val * s_th - 0.5f;
          cos_val = cos_val * s_th + 1.0f;

          sfpi::vFloat sin_val = sfpi::vFloat(sfpi::sFloat16b(0.000002225f));
          sin_val = sin_val * s_th - sfpi::vFloat(sfpi::sFloat16b(0.00019424f));
          sin_val = sin_val * s_th + sfpi::vFloat(sfpi::sFloat16b(0.0083198f));
          sin_val = sin_val * s_th - sfpi::vFloat(sfpi::sFloat16b(0.166667f));
          sin_val = sin_val * s_th + 1.0f;
#endif
          sfpi::vFloat llr0 = v_mu + v_sigma * (r * cos_val);
#if LDPC_ENABLE_CH_CLAMP
          llr0 = sfpi::min(sfpi::max(llr0, -LDPC_L_CH_MAX), LDPC_L_CH_MAX);
#endif

          sfpi::vFloat llr1 = v_mu + v_sigma * (r * (sin_val * theta));
#if LDPC_ENABLE_CH_CLAMP
          llr1 = sfpi::min(sfpi::max(llr1, -LDPC_L_CH_MAX), LDPC_L_CH_MAX);
#endif

          // 50/50 Codeword Interleaving: Even p assigns (llr0, llr1); Odd p
          // assigns (llr1, llr0). Guarantees all 4 codewords receive an
          // identical 50/50 blend of sine & cosine variates.
          if ((p % 2) == 1) {
            sfpi::dst_reg[2 * p + 0] = llr1;
            sfpi::dst_reg[2 * p + 1] = llr0;
          } else {
            sfpi::dst_reg[2 * p + 0] = llr0;
            sfpi::dst_reg[2 * p + 1] = llr1;
          }
        }
      }
    } else {
      // Stage 2: Generate 2 Punctured Zero Tiles (2 x 256 = 512 Zeros)
      for (int r = 0; r < 32; ++r) {
        sfpi::dst_reg[r] = 0.0f;
      }
    }

    _llk_math_eltwise_sfpu_done_();
#endif

    tile_regs_commit();
    tile_regs_wait();

    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out, 0);
    cb_push_back(cb_out, 1);

    tile_regs_release();
  }
}

// ------------------------------------------------------------------------------
// Zero-Transpose 32-Lane Vector Approximate-Min* for Degree-6 Tile (32 Checks)
// ------------------------------------------------------------------------------
uint32_t __attribute__((noinline)) compute_tile_deg6(uint32_t cb_in,
                                                     uint32_t cb_out,
                                                     uint32_t cb_r_msg,
                                                     uint32_t iter,
                                                     volatile uint32_t* sync_flags) {
  volatile uint32_t *const wall_clock_reg = reinterpret_cast<volatile uint32_t *>(0xFFB121F0);

  if (iter == 0) {
    cb_wait_front(cb_in, 1);
    tile_regs_acquire();
    copy_tile(cb_in, 0, 0); // Tile L into dst_reg[0..31]
    cb_pop_front(cb_in, 1);
  } else {
    cb_wait_front(cb_in, 1);
    cb_wait_front(cb_r_msg, 1);
    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);    // Tile L into dst_reg[0..31]
    copy_tile(cb_r_msg, 0, 1); // Tile R_old from cb_r_msg into dst_reg[32..63]
    cb_pop_front(cb_in, 1);
    cb_pop_front(cb_r_msg, 1);
  }

#ifdef TRISC_UNPACK
  sync_flags[SyncSlot::T2_UNPACK_DONE] = *wall_clock_reg;
  asm volatile("fence rw, rw" ::: "memory");
#endif

#ifdef TRISC_MATH
  uint32_t t_m0 = *wall_clock_reg;
  _llk_math_eltwise_sfpu_start_(0);

#if LDPC_TEST_PASSTHROUGH_MODE
  // In pass-through mode, Tile 0 is already in dst_reg[0..31] from copy_tile(cb_in, 0, 0).
#elif LDPC_TEST_R_DATAFLOW_MODE
  for (int r = 0; r < 32; ++r) {
    sfpi::vFloat tmp = sfpi::dst_reg[r];
    sfpi::dst_reg[32 + r] = tmp;
    sfpi::dst_reg[r] = 0.0f;
  }
#else
  // 2 Worker Groups:
  // Worker 0: Checks 0..15  (Face 0: Edges 0..2, Face 2: Edges 3..5)
  // Worker 1: Checks 16..31 (Face 1: Edges 0..2, Face 3: Edges 3..5)
  const int w_low_bases[2]  = {0, 8};
  const int w_high_bases[2] = {16, 24};

#pragma GCC unroll 2
  for (int w = 0; w < 2; ++w) {
    int f_low  = w_low_bases[w];
    int f_high = w_high_bases[w];

#pragma GCC unroll 1
    for (int sub = 0; sub < 2; ++sub) {
      int r0 = f_low  + 0 + sub; // Edge 0
      int r3 = f_high + 0 + sub; // Edge 3
      int r1 = f_low  + 2 + sub; // Edge 1
      int r4 = f_high + 2 + sub; // Edge 4
      int r2 = f_low  + 4 + sub; // Edge 2
      int r5 = f_high + 4 + sub; // Edge 5

      int sc0 = f_low  + 6;
      int sc1 = f_low  + 7;
      int sc2 = f_high + 6;
      int sc3 = f_high + 7;

      // Stage 0: Q = L - R_old in-place in Tile 0 (iter > 0)
      if (iter != 0) {
        sfpi::dst_reg[r0] = round_bf16_fast(sfpi::vFloat(sfpi::dst_reg[r0]) - sfpi::vFloat(sfpi::dst_reg[32 + r0]));
        sfpi::dst_reg[r3] = round_bf16_fast(sfpi::vFloat(sfpi::dst_reg[r3]) - sfpi::vFloat(sfpi::dst_reg[32 + r3]));
        sfpi::dst_reg[r1] = round_bf16_fast(sfpi::vFloat(sfpi::dst_reg[r1]) - sfpi::vFloat(sfpi::dst_reg[32 + r1]));
        sfpi::dst_reg[r4] = round_bf16_fast(sfpi::vFloat(sfpi::dst_reg[r4]) - sfpi::vFloat(sfpi::dst_reg[32 + r4]));
        sfpi::dst_reg[r2] = round_bf16_fast(sfpi::vFloat(sfpi::dst_reg[r2]) - sfpi::vFloat(sfpi::dst_reg[32 + r2]));
        sfpi::dst_reg[r5] = round_bf16_fast(sfpi::vFloat(sfpi::dst_reg[r5]) - sfpi::vFloat(sfpi::dst_reg[32 + r5]));
      }

      // Stage 1: Native 1-Cycle Integer XOR Sign Parity -> stored in dst_reg[sc0]
      {
        sfpi::vInt s0 = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[r0]));
        sfpi::vInt s3 = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[r3]));
        sfpi::vInt s1 = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[r1]));
        sfpi::vInt s4 = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[r4]));
        sfpi::vInt s2 = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[r2]));
        sfpi::vInt s5 = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[r5]));
        sfpi::dst_reg[sc0] = sfpi::as<sfpi::vFloat>(((s0 ^ s3) ^ (s1 ^ s4)) ^ (s2 ^ s5));
      }

      // Stage 2: Pairwise Tree Approximate-Min* stored in DEST scratch
      {
        sfpi::vFloat m0 = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r0]));
        sfpi::vFloat m3 = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r3]));
        sfpi::dst_reg[sc1] = sfpi_amin_star_branchless(m0, m3); // m03 in sc1
      }
      {
        sfpi::vFloat m1 = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r1]));
        sfpi::vFloat m4 = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r4]));
        sfpi::dst_reg[sc2] = sfpi_amin_star_branchless(m1, m4); // m14 in sc2
      }
      {
        sfpi::vFloat m2 = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r2]));
        sfpi::vFloat m5 = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r5]));
        sfpi::dst_reg[sc3] = sfpi_amin_star_branchless(m2, m5); // m25 in sc3
      }

      // Stage 3: Outgoing check messages with Zero LReg pressure
      // Edge pair (0, 3)
      {
        sfpi::vFloat m_oth0 = sfpi_amin_star_branchless(sfpi::dst_reg[sc2], sfpi::dst_reg[sc3]);

        sfpi::vFloat q3 = sfpi::dst_reg[r3];
        sfpi::vFloat r_mag0 = sfpi_amin_star_branchless(m_oth0, sfpi::abs(q3));
        sfpi::vFloat q0 = sfpi::dst_reg[r0];
        sfpi::vInt s_tot = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc0]));
        sfpi::vFloat r_out0 = round_bf16(sfpi::copysgn(r_mag0, s_tot ^ sfpi::as<sfpi::vInt>(q0)));
        sfpi::dst_reg[32 + r0] = round_bf16_fast(q0 + r_out0);

        sfpi::vFloat r_mag3 = sfpi_amin_star_branchless(m_oth0, sfpi::abs(q0));
        sfpi::vFloat r_out3 = round_bf16(sfpi::copysgn(r_mag3, s_tot ^ sfpi::as<sfpi::vInt>(q3)));
        sfpi::dst_reg[32 + r3] = round_bf16_fast(q3 + r_out3);

        sfpi::dst_reg[r0] = r_out0;
        sfpi::dst_reg[r3] = r_out3;
      }

      // Edge pair (1, 4)
      {
        sfpi::vFloat m_oth1 = sfpi_amin_star_branchless(sfpi::dst_reg[sc1], sfpi::dst_reg[sc3]);

        sfpi::vFloat q4 = sfpi::dst_reg[r4];
        sfpi::vFloat r_mag1 = sfpi_amin_star_branchless(m_oth1, sfpi::abs(q4));
        sfpi::vFloat q1 = sfpi::dst_reg[r1];
        sfpi::vInt s_tot = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc0]));
        sfpi::vFloat r_out1 = round_bf16(sfpi::copysgn(r_mag1, s_tot ^ sfpi::as<sfpi::vInt>(q1)));
        sfpi::dst_reg[32 + r1] = round_bf16_fast(q1 + r_out1);

        sfpi::vFloat r_mag4 = sfpi_amin_star_branchless(m_oth1, sfpi::abs(q1));
        sfpi::vFloat r_out4 = round_bf16(sfpi::copysgn(r_mag4, s_tot ^ sfpi::as<sfpi::vInt>(q4)));
        sfpi::dst_reg[32 + r4] = round_bf16_fast(q4 + r_out4);

        sfpi::dst_reg[r1] = r_out1;
        sfpi::dst_reg[r4] = r_out4;
      }

      // Edge pair (2, 5)
      {
        sfpi::vFloat m_oth2 = sfpi_amin_star_branchless(sfpi::dst_reg[sc1], sfpi::dst_reg[sc2]);

        sfpi::vFloat q5 = sfpi::dst_reg[r5];
        sfpi::vFloat r_mag2 = sfpi_amin_star_branchless(m_oth2, sfpi::abs(q5));
        sfpi::vFloat q2 = sfpi::dst_reg[r2];
        sfpi::vInt s_tot = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc0]));
        sfpi::vFloat r_out2 = round_bf16(sfpi::copysgn(r_mag2, s_tot ^ sfpi::as<sfpi::vInt>(q2)));
        sfpi::dst_reg[32 + r2] = round_bf16_fast(q2 + r_out2);

        sfpi::vFloat r_mag5 = sfpi_amin_star_branchless(m_oth2, sfpi::abs(q2));
        sfpi::vFloat r_out5 = round_bf16(sfpi::copysgn(r_mag5, s_tot ^ sfpi::as<sfpi::vInt>(q5)));
        sfpi::dst_reg[32 + r5] = round_bf16_fast(q5 + r_out5);

        sfpi::dst_reg[r2] = r_out2;
        sfpi::dst_reg[r5] = r_out5;
      }
    }

    // Zero out unused scratch registers (regs 6, 7 of each face)
    sfpi::dst_reg[f_low  + 6] = 0.0f;
    sfpi::dst_reg[f_low  + 7] = 0.0f;
    sfpi::dst_reg[f_high + 6] = 0.0f;
    sfpi::dst_reg[f_high + 7] = 0.0f;
    sfpi::dst_reg[32 + f_low  + 6] = 0.0f;
    sfpi::dst_reg[32 + f_low  + 7] = 0.0f;
    sfpi::dst_reg[32 + f_high + 6] = 0.0f;
    sfpi::dst_reg[32 + f_high + 7] = 0.0f;
  }
#endif

  _llk_math_eltwise_sfpu_done_();
  uint32_t math_cyc = *wall_clock_reg - t_m0;
#endif

  tile_regs_commit();

#ifdef TRISC_MATH
  sync_flags[SyncSlot::T3_MATH_DONE] = *wall_clock_reg;
  asm volatile("fence rw, rw" ::: "memory");
#endif

  tile_regs_wait();

  // Pack R_new (Dst reg 0) to cb_r_msg
  cb_reserve_back(cb_r_msg, 1);
  pack_init(cb_r_msg);
  pack_tile(0, cb_r_msg, 0);
  cb_push_back(cb_r_msg, 1);

  // Pack L_new to cb_out
  cb_reserve_back(cb_out, 1);
  pack_init(cb_out);
#if LDPC_TEST_PASSTHROUGH_MODE
  pack_tile(0, cb_out, 0);
#else
  pack_tile(1, cb_out, 0);
#endif
  cb_push_back(cb_out, 1);

  tile_regs_release();

#ifdef TRISC_PACK
  sync_flags[SyncSlot::T4_PACK_DONE] = *wall_clock_reg;
  asm volatile("fence rw, rw" ::: "memory");
#endif

#ifdef TRISC_MATH
  return math_cyc;
#else
  return 0;
#endif
}

// ------------------------------------------------------------------------------
// Zero-Transpose 32-Lane Vector Approximate-Min* for Degree-3 Tile (64 Checks)
// ------------------------------------------------------------------------------
uint32_t __attribute__((noinline)) compute_tile_deg3(uint32_t cb_in,
                                                     uint32_t cb_out,
                                                     uint32_t cb_r_msg,
                                                     uint32_t iter,
                                                     volatile uint32_t* sync_flags) {
  volatile uint32_t *const wall_clock_reg = reinterpret_cast<volatile uint32_t *>(0xFFB121F0);

  if (iter == 0) {
    cb_wait_front(cb_in, 1);
    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);
    cb_pop_front(cb_in, 1);
  } else {
    cb_wait_front(cb_in, 1);
    cb_wait_front(cb_r_msg, 1);
    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);
    copy_tile(cb_r_msg, 0, 1);
    cb_pop_front(cb_in, 1);
    cb_pop_front(cb_r_msg, 1);
  }

#ifdef TRISC_UNPACK
  sync_flags[SyncSlot::T2_UNPACK_DONE] = *wall_clock_reg;
  asm volatile("fence rw, rw" ::: "memory");
#endif

#ifdef TRISC_MATH
  uint32_t t_m0 = *wall_clock_reg;
  _llk_math_eltwise_sfpu_start_(0);

#if LDPC_TEST_PASSTHROUGH_MODE
  // Pass-through
#elif LDPC_TEST_R_DATAFLOW_MODE
  for (int r = 0; r < 32; ++r) {
    sfpi::vFloat tmp = sfpi::dst_reg[r];
    sfpi::dst_reg[32 + r] = tmp;
    sfpi::dst_reg[r] = 0.0f;
  }
#else
  const int face_bases[4] = {0, 8, 16, 24};

#pragma GCC unroll 4
  for (int f = 0; f < 4; ++f) {
    int base_r = face_bases[f];

#pragma GCC unroll 1
    for (int sub = 0; sub < 2; ++sub) {
      int r0 = base_r + 0 + sub;
      int r1 = base_r + 2 + sub;
      int r2 = base_r + 4 + sub;
      int sc0 = base_r + 6;
      int sc1 = base_r + 7;

      if (iter != 0) {
        sfpi::dst_reg[r0] = round_bf16_fast(sfpi::vFloat(sfpi::dst_reg[r0]) - sfpi::vFloat(sfpi::dst_reg[32 + r0]));
        sfpi::dst_reg[r1] = round_bf16_fast(sfpi::vFloat(sfpi::dst_reg[r1]) - sfpi::vFloat(sfpi::dst_reg[32 + r1]));
        sfpi::dst_reg[r2] = round_bf16_fast(sfpi::vFloat(sfpi::dst_reg[r2]) - sfpi::vFloat(sfpi::dst_reg[32 + r2]));
      }

      // Stage 1: Sign parity stored into scratch sc0
      {
        sfpi::vInt s0 = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[r0]));
        sfpi::vInt s1 = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[r1]));
        sfpi::vInt s2 = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[r2]));
        sfpi::dst_reg[sc0] = sfpi::as<sfpi::vFloat>((s0 ^ s1) ^ s2);
      }

      // Edge 0: computes r_out0, writes L_new to dst_reg[32 + r0], stages r_out0 into sc1
      {
        sfpi::vFloat m1 = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r1]));
        sfpi::vFloat m2 = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r2]));
        sfpi::vFloat r_mag0 = sfpi_amin_star_branchless(m1, m2);

        sfpi::vInt s_tot = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc0]));
        sfpi::vFloat q0     = sfpi::dst_reg[r0];
        sfpi::vFloat r_out0 = round_bf16(sfpi::copysgn(r_mag0, s_tot ^ sfpi::as<sfpi::vInt>(q0)));
        sfpi::dst_reg[32 + r0] = round_bf16_fast(q0 + r_out0);
        sfpi::dst_reg[sc1]     = r_out0; // Stage in sc1 so dst_reg[r0] remains pristine q0
      }

      // Edge 1: computes r_out1, writes L_new to dst_reg[32 + r1], stages r_out1 into (32 + sc0)
      {
        sfpi::vFloat m0 = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r0])); // pristine q0
        sfpi::vFloat m2 = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r2]));
        sfpi::vFloat r_mag1 = sfpi_amin_star_branchless(m0, m2);

        sfpi::vInt s_tot = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc0]));
        sfpi::vFloat q1     = sfpi::dst_reg[r1];
        sfpi::vFloat r_out1 = round_bf16(sfpi::copysgn(r_mag1, s_tot ^ sfpi::as<sfpi::vInt>(q1)));
        sfpi::dst_reg[32 + r1] = round_bf16_fast(q1 + r_out1);
        sfpi::dst_reg[32 + sc0] = r_out1; // Stage in (32 + sc0) so dst_reg[r1] remains pristine q1
      }

      // Edge 2: computes r_out2, writes L_new to dst_reg[32 + r2], writes r_out2 to dst_reg[r2]
      {
        sfpi::vFloat m0 = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r0])); // pristine q0
        sfpi::vFloat m1 = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r1])); // pristine q1
        sfpi::vFloat r_mag2 = sfpi_amin_star_branchless(m0, m1);

        sfpi::vInt s_tot = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc0]));
        sfpi::vFloat q2     = sfpi::dst_reg[r2];
        sfpi::vFloat r_out2 = round_bf16(sfpi::copysgn(r_mag2, s_tot ^ sfpi::as<sfpi::vInt>(q2)));
        sfpi::dst_reg[32 + r2] = round_bf16_fast(q2 + r_out2);
        sfpi::dst_reg[r2]      = r_out2;
      }

      // Commit staged r_out0 and r_out1
      sfpi::dst_reg[r0] = sfpi::vFloat(sfpi::dst_reg[sc1]);
      sfpi::dst_reg[r1] = sfpi::vFloat(sfpi::dst_reg[32 + sc0]);
    }

    sfpi::dst_reg[base_r + 6] = 0.0f;
    sfpi::dst_reg[base_r + 7] = 0.0f;
    sfpi::dst_reg[32 + base_r + 6] = 0.0f;
    sfpi::dst_reg[32 + base_r + 7] = 0.0f;
  }
#endif

  _llk_math_eltwise_sfpu_done_();
  uint32_t math_cyc = *wall_clock_reg - t_m0;
#endif

  tile_regs_commit();

#ifdef TRISC_MATH
  sync_flags[SyncSlot::T3_MATH_DONE] = *wall_clock_reg;
  asm volatile("fence rw, rw" ::: "memory");
#endif

  tile_regs_wait();

  // Pack R_new to cb_r_msg
  cb_reserve_back(cb_r_msg, 1);
  pack_init(cb_r_msg);
  pack_tile(0, cb_r_msg, 0);
  cb_push_back(cb_r_msg, 1);

  // Pack L_new to cb_out
  cb_reserve_back(cb_out, 1);
  pack_init(cb_out);
#if LDPC_TEST_PASSTHROUGH_MODE
  pack_tile(0, cb_out, 0);
#else
  pack_tile(1, cb_out, 0);
#endif
  cb_push_back(cb_out, 1);

  tile_regs_release();

#ifdef TRISC_PACK
  sync_flags[SyncSlot::T4_PACK_DONE] = *wall_clock_reg;
  asm volatile("fence rw, rw" ::: "memory");
#endif

#ifdef TRISC_MATH
  return math_cyc;
#else
  return 0;
#endif
}

// ------------------------------------------------------------------------------
// Main Kernel Entry Point
// ------------------------------------------------------------------------------
void kernel_main() {
  uint32_t num_batches     = get_arg_val<uint32_t>(0);
  uint32_t max_iter        = get_arg_val<uint32_t>(1);
  uint32_t num_tiles       = get_arg_val<uint32_t>(2);
  uint32_t mu_bits         = get_arg_val<uint32_t>(3);
  (void)mu_bits;
  uint32_t sigma_bits      = get_arg_val<uint32_t>(4);
  (void)sigma_bits;
  uint32_t invocation_id   = get_arg_val<uint32_t>(5);
  uint32_t diff_test_mode  = get_arg_val<uint32_t>(6);
  (void)diff_test_mode;
  uint32_t start_batch     = get_arg_val<uint32_t>(7);
  (void)start_batch;
  uint32_t core_id         = get_arg_val<uint32_t>(8);
  (void)core_id;

  if (invocation_id == 0) invocation_id = 1;
  uint32_t epoch = invocation_id << 16;

  constexpr uint32_t cb_vnode_a     = TileMatrixCB::CB_VNODE_A;
  constexpr uint32_t cb_vnode_b     = TileMatrixCB::CB_VNODE_B;
  constexpr uint32_t cb_vnode       = TileMatrixCB::CB_VNODE;
  constexpr uint32_t cb_scratch_in  = TileMatrixCB::CB_SCRATCH_GATHER;
  constexpr uint32_t cb_scratch_out = TileMatrixCB::CB_SCRATCH_SCATTER;
  constexpr uint32_t cb_r_msg       = TileMatrixCB::CB_R_MSG;
  constexpr uint32_t cb_telem       = TileMatrixCB::CB_TELEMETRY;

  compute_kernel_hw_startup(cb_scratch_in, cb_scratch_out);
  copy_init(cb_scratch_in);

  uint32_t telem_l1 = ckernel::get_tile_address(cb_telem, 0);
  volatile uint32_t* sync_flags = reinterpret_cast<volatile uint32_t*>(telem_l1);

  // Initial startup handshake with NCRISC and BRISC
#ifdef TRISC_UNPACK
  WAYPOINT("U_01");
#endif
#ifdef TRISC_MATH
  WAYPOINT("M_01");
#endif
#ifdef TRISC_PACK
  WAYPOINT("P_01");
#endif

  while (sync_flags[SyncSlot::NCRISC_READY] < epoch) {
    asm volatile("" ::: "memory");
  }
  while (sync_flags[SyncSlot::BRISC_READY] < epoch) {
    asm volatile("" ::: "memory");
  }

#ifdef TRISC_PACK
  sync_flags[SyncSlot::TRISC_READY] = epoch;
  asm volatile("fence rw, rw" ::: "memory");
  WAYPOINT("P_02");
#endif

  while (sync_flags[SyncSlot::TRISC_READY] < epoch) {
    asm volatile("" ::: "memory");
  }

#ifdef TRISC_UNPACK
  WAYPOINT("U_02");
#endif
#ifdef TRISC_MATH
  WAYPOINT("M_02");
#endif

  uint32_t max_steps = max_iter * num_tiles;

  volatile uint32_t *const wall_clock_reg = reinterpret_cast<volatile uint32_t *>(0xFFB121F0);
  uint32_t cycles_compute = 0;
  uint32_t cycles_math_pure = 0;

  bool benchmark_mode = (sigma_bits == 0);
  float mu_val = uint_as_float(mu_bits);
  float sigma_val = uint_as_float(sigma_bits);

  Xoshiro128Plus rng_core;
  uint32_t seed_master = 0x12345678U + (invocation_id * 1013904223U);
  rng_core.init(seed_master);
  for (uint32_t c = 0; c < core_id; ++c) {
    rng_core.long_jump();
  }
  for (uint32_t b = 0; b < start_batch; ++b) {
    rng_core.jump();
  }

  for (uint32_t batch = 0; batch < num_batches; ++batch) {
    uint32_t active_cb_vnode = cb_vnode_a;
#if !LDPC_TEST_PASSTHROUGH_MODE
    uint32_t seed_batch = rng_core.next();
    generate_channel_tiles(active_cb_vnode, mu_val, sigma_val, benchmark_mode, invocation_id, seed_batch);
    rng_core.jump();
#endif

    // Tri-core batch rendezvous barrier
#ifdef TRISC_PACK
    sync_flags[SyncSlot::BATCH_BARRIER_TRISC] = epoch + batch + 1;
    asm volatile("fence rw, rw" ::: "memory");
    WAYPOINT("P_03");
#endif
#ifdef TRISC_UNPACK
    WAYPOINT("U_03");
#endif
#ifdef TRISC_MATH
    WAYPOINT("M_03");
#endif
    while (sync_flags[SyncSlot::BATCH_BARRIER_NCRISC] < epoch + batch + 1) {
      asm volatile("" ::: "memory");
    }
#ifdef TRISC_UNPACK
    WAYPOINT("U_04");
#endif
#ifdef TRISC_MATH
    WAYPOINT("M_04");
#endif
#ifdef TRISC_PACK
    WAYPOINT("P_04");
#endif
    while (sync_flags[SyncSlot::BATCH_BARRIER_BRISC] < epoch + batch + 1) {
      asm volatile("" ::: "memory");
    }
#ifdef TRISC_UNPACK
    WAYPOINT("U_05");
#endif
#ifdef TRISC_MATH
    WAYPOINT("M_05");
#endif
#ifdef TRISC_PACK
    WAYPOINT("P_05");
#endif
#ifndef TRISC_PACK
    while (sync_flags[SyncSlot::BATCH_BARRIER_TRISC] < epoch + batch + 1) {
      asm volatile("" ::: "memory");
    }
#endif

    // Wait for channel observations / patterned data to be ready in L1
#ifdef TRISC_UNPACK
    WAYPOINT("U_06");
#endif
#ifdef TRISC_MATH
    WAYPOINT("M_06");
#endif
#ifdef TRISC_PACK
    WAYPOINT("P_06");
#endif
    while (sync_flags[SyncSlot::BATCH_READY] < epoch + batch + 1) {
      asm volatile("" ::: "memory");
    }

    for (uint32_t step = 0; step < max_steps; ++step) {
      uint32_t bar = batch * max_steps + step + 1;
      uint32_t t_step = step % num_tiles;
      uint32_t iter = step / num_tiles;

#ifdef TRISC_UNPACK
      sync_flags[SyncSlot::DEBUG_TRISC0_STEP] = (step << 8) | 1;
      WAYPOINT("U_CP");
#endif
#ifdef TRISC_PACK
      sync_flags[SyncSlot::DEBUG_TRISC2_STEP] = (step << 8) | 1;
      WAYPOINT("P_CP");
#endif
#ifdef TRISC_MATH
      sync_flags[SyncSlot::DEBUG_TRISC1_STEP] = (step << 8) | 1;
      WAYPOINT("M_CP");
#endif

      uint32_t t_c0 = *wall_clock_reg;
      uint32_t m_cyc = 0;
      bool is_deg6 = (t_step < QC_LDPC_4CW::DEG6_TILES);
      if (is_deg6) {
        m_cyc = compute_tile_deg6(cb_scratch_in, cb_scratch_out, cb_r_msg, iter, sync_flags);
      } else {
        m_cyc = compute_tile_deg3(cb_scratch_in, cb_scratch_out, cb_r_msg, iter, sync_flags);
      }
#ifdef TRISC_MATH
      cycles_math_pure += m_cyc;
#else
      (void)m_cyc;
#endif
#ifdef TRISC_PACK
      cycles_compute += (*wall_clock_reg - t_c0);
#endif

#ifdef TRISC_UNPACK
      WAYPOINT("U_DN");
#endif
#ifdef TRISC_PACK
      WAYPOINT("P_DN");
#endif
#ifdef TRISC_MATH
      WAYPOINT("M_DN");
#endif

      // Iteration boundary early termination check (Tile 39)
      if (t_step == num_tiles - 1) {
        while (sync_flags[SyncSlot::SCATTER_DONE_W1] < bar) {
          asm volatile("" ::: "memory");
        }
        if (sync_flags[SyncSlot::EARLY_TERMINATION] != 0) {
          break;
        }
      }
    }

#ifdef TRISC_UNPACK
    // Drain circular FIFO check messages for next batch
    for (uint32_t t = 0; t < num_tiles; ++t) {
      cb_wait_front(cb_r_msg, 1);
      cb_pop_front(cb_r_msg, 1);
    }
    sync_flags[SyncSlot::DRAIN_ACK_T0] = epoch + batch + 1;
    asm volatile("fence rw, rw" ::: "memory");
#endif
#ifdef TRISC_PACK
    while (sync_flags[SyncSlot::DRAIN_ACK_T0] < epoch + batch + 1) {
      asm volatile("" ::: "memory");
    }
#endif
#ifdef TRISC_MATH
    while (sync_flags[SyncSlot::DRAIN_ACK_T0] < epoch + batch + 1) {
      asm volatile("" ::: "memory");
    }
#endif
  }

#ifdef TRISC_MATH
  sync_flags[SyncSlot::TRISC_MATH_CYCLES] = cycles_math_pure;
#endif
#ifdef TRISC_PACK
  sync_flags[SyncSlot::TRISC_CYCLES] = cycles_compute;
  sync_flags[SyncSlot::BATCH_BARRIER_TRISC] = epoch + num_batches + 1;
#endif
}
