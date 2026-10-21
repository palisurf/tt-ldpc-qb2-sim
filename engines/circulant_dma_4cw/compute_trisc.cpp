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
  sfpi::vUInt u = sfpi::as<sfpi::vUInt>(val);
  sfpi::vUInt lsb = (u >> 16) & 1U;
  sfpi::vUInt bias = lsb + 0x7FFFU;
  u += bias;
  return sfpi::as<sfpi::vFloat>(u);
}

// ------------------------------------------------------------------------------
// Zero-Transpose 32-Lane Vector Approximate-Min* for Degree-6 Tile (32 Checks)
// ------------------------------------------------------------------------------
uint32_t __attribute__((noinline)) compute_tile_deg6(uint32_t cb_in,
                                                     uint32_t cb_out,
                                                     uint32_t cb_r_msg,
                                                     uint32_t iter) {
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

#ifdef TRISC_MATH
  volatile uint32_t *const wall_clock_reg = reinterpret_cast<volatile uint32_t *>(0xFFB121F0);
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

      // Stage 1: 1-Cycle Native Integer XOR sign parity -> stored in dst_reg[sc0]
      {
        sfpi::vInt s03 = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[r0])) ^
                         sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[r3]));
        sfpi::vInt s14 = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[r1])) ^
                         sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[r4]));
        sfpi::vInt s25 = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[r2])) ^
                         sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[r5]));
        sfpi::dst_reg[sc0] = sfpi::as<sfpi::vFloat>((s03 ^ s14) ^ s25);
      }

      // Stage 2: Dr. Christopher Jones' Approximate-Min* Formulation (MILCOM 2003)
      // Step 2a: Find minimum magnitude and its edge index (0..5) across 32 lanes
      {
        sfpi::vFloat min_mag = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r0]));
        sfpi::vInt min_idx = 0;

        const int r_all[6] = {r0, r1, r2, r3, r4, r5};
        #pragma GCC unroll 5
        for (int i = 1; i < 6; ++i) {
          sfpi::vFloat m = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r_all[i]]));
          v_if (m < min_mag) {
            min_mag = m;
            min_idx = i;
          } v_endif;
        }
        sfpi::dst_reg[sc1] = sfpi::as<sfpi::vFloat>(min_idx); // min_idx in sc1
        sfpi::dst_reg[sc2] = min_mag;                         // min_mag in sc2
      }

      // Step 2b: Form M_others by accumulating all edges k != d_min (Neutral element: 1000.0f)
      // Level 1: Pairwise accumulation of 3 edge pairs with neutral element substitution
      {
        sfpi::vInt min_idx = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc1]));
        sfpi::vFloat m0 = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r0]));
        v_if (min_idx == 0) { m0 = 1000.0f; } v_endif;
        sfpi::vFloat m3 = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r3]));
        v_if (min_idx == 3) { m3 = 1000.0f; } v_endif;
        sfpi::dst_reg[32 + sc1] = sfpi_amin_star_branchless(m0, m3); // Pair (0, 3)
      }
      {
        sfpi::vInt min_idx = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc1]));
        sfpi::vFloat m1 = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r1]));
        v_if (min_idx == 1) { m1 = 1000.0f; } v_endif;
        sfpi::vFloat m4 = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r4]));
        v_if (min_idx == 4) { m4 = 1000.0f; } v_endif;
        sfpi::dst_reg[32 + sc2] = sfpi_amin_star_branchless(m1, m4); // Pair (1, 4)
      }
      {
        sfpi::vInt min_idx = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc1]));
        sfpi::vFloat m2 = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r2]));
        v_if (min_idx == 2) { m2 = 1000.0f; } v_endif;
        sfpi::vFloat m5 = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r5]));
        v_if (min_idx == 5) { m5 = 1000.0f; } v_endif;
        sfpi::dst_reg[32 + sc3] = sfpi_amin_star_branchless(m2, m5); // Pair (2, 5)
      }

      // Level 2: Combine Pair (0,3) and Pair (1,4)
      {
        sfpi::vFloat p03 = sfpi::dst_reg[32 + sc1];
        sfpi::vFloat p14 = sfpi::dst_reg[32 + sc2];
        sfpi::dst_reg[sc3] = sfpi_amin_star_branchless(p03, p14);
      }

      // Level 3: Combine with Pair (2,5) to complete M_others
      {
        sfpi::vFloat p0134 = sfpi::dst_reg[sc3];
        sfpi::vFloat p25   = sfpi::dst_reg[32 + sc3];
        sfpi::dst_reg[sc3] = sfpi_amin_star_branchless(p0134, p25); // M_others in sc3
      }

      // Step 2c: Final single binary AMin* to form M_all = AMin*(M_others, min_mag)
      {
        sfpi::vFloat m_others = sfpi::dst_reg[sc3];
        sfpi::vFloat min_mag  = sfpi::dst_reg[sc2];
        sfpi::dst_reg[32 + sc1] = sfpi_amin_star_branchless(m_others, min_mag); // M_all in 32+sc1
      }

      // Stage 3: Outgoing check messages with Intrinsic Damping (beta = 0.9375 = 15/16)
      const sfpi::vFloat v_damping = sfpi::vFloat(sfpi::sFloat16b(0.9375f));
      sfpi::vInt s_tot             = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc0]));
      sfpi::vInt min_idx_saved     = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc1]));
      sfpi::vFloat m_others_damped = sfpi::vFloat(sfpi::dst_reg[sc3]) * v_damping;
      sfpi::vFloat m_all_damped    = sfpi::vFloat(sfpi::dst_reg[32 + sc1]) * v_damping;

      const int r_all[6] = {r0, r1, r2, r3, r4, r5};
      #pragma GCC unroll 6
      for (int i = 0; i < 6; ++i) {
        int r = r_all[i];
        sfpi::vFloat q = sfpi::dst_reg[r];
        sfpi::vFloat m_out = m_all_damped;
        v_if (min_idx_saved == i) {
          m_out = m_others_damped;
        } v_endif;

        sfpi::vFloat r_out = round_bf16(sfpi::copysgn(m_out, s_tot ^ sfpi::as<sfpi::vInt>(q)));
        sfpi::dst_reg[32 + r] = round_bf16_fast(q + r_out); // L_new in Tile 1
        sfpi::dst_reg[r]      = r_out;                     // R_new in Tile 0
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
                                                     uint32_t iter) {
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

#ifdef TRISC_MATH
  volatile uint32_t *const wall_clock_reg = reinterpret_cast<volatile uint32_t *>(0xFFB121F0);
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

      if (iter != 0) {
        sfpi::dst_reg[r0] = round_bf16_fast(sfpi::vFloat(sfpi::dst_reg[r0]) - sfpi::vFloat(sfpi::dst_reg[32 + r0]));
        sfpi::dst_reg[r1] = round_bf16_fast(sfpi::vFloat(sfpi::dst_reg[r1]) - sfpi::vFloat(sfpi::dst_reg[32 + r1]));
        sfpi::dst_reg[r2] = round_bf16_fast(sfpi::vFloat(sfpi::dst_reg[r2]) - sfpi::vFloat(sfpi::dst_reg[32 + r2]));
      }

      // Stage 1: Sign parity stored into scratch
      {
        sfpi::vInt s0 = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[r0]));
        sfpi::vInt s1 = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[r1]));
        sfpi::vInt s2 = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[r2]));
        sfpi::dst_reg[sc0] = sfpi::as<sfpi::vFloat>((s0 ^ s1) ^ s2);
      }

      const sfpi::vFloat v_damping = sfpi::vFloat(sfpi::sFloat16b(0.9375f));

      // Edge 0
      {
        sfpi::vFloat m1 = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r1]));
        sfpi::vFloat m2 = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r2]));
        sfpi::vFloat r_mag0 = sfpi_amin_star_branchless(m1, m2) * v_damping;

        sfpi::vInt s_tot = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc0]));
        sfpi::vFloat q0     = sfpi::dst_reg[r0];
        sfpi::vFloat r_out0 = round_bf16(sfpi::copysgn(r_mag0, s_tot ^ sfpi::as<sfpi::vInt>(q0)));
        sfpi::dst_reg[32 + r0] = round_bf16_fast(q0 + r_out0);
        sfpi::dst_reg[r0]      = r_out0;
      }

      // Edge 1
      {
        sfpi::vFloat m0 = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r0]));
        sfpi::vFloat m2 = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r2]));
        sfpi::vFloat r_mag1 = sfpi_amin_star_branchless(m0, m2) * v_damping;

        sfpi::vInt s_tot = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc0]));
        sfpi::vFloat q1     = sfpi::dst_reg[r1];
        sfpi::vFloat r_out1 = round_bf16(sfpi::copysgn(r_mag1, s_tot ^ sfpi::as<sfpi::vInt>(q1)));
        sfpi::dst_reg[32 + r1] = round_bf16_fast(q1 + r_out1);
        sfpi::dst_reg[r1]      = r_out1;
      }

      // Edge 2
      {
        sfpi::vFloat m0 = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r0]));
        sfpi::vFloat m1 = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[r1]));
        sfpi::vFloat r_mag2 = sfpi_amin_star_branchless(m0, m1) * v_damping;

        sfpi::vInt s_tot = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc0]));
        sfpi::vFloat q2     = sfpi::dst_reg[r2];
        sfpi::vFloat r_out2 = round_bf16(sfpi::copysgn(r_mag2, s_tot ^ sfpi::as<sfpi::vInt>(q2)));
        sfpi::dst_reg[32 + r2] = round_bf16_fast(q2 + r_out2);
        sfpi::dst_reg[r2]      = r_out2;
      }
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
#ifdef TRISC_MATH
  return math_cyc;
#else
  return 0;
#endif
}

// ------------------------------------------------------------------------------
// Vector PRNG for Box-Muller AWGN Channel Noise Generation
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

void __attribute__((noinline))
generate_channel_tiles(uint32_t cb_out, float mu_val, float sigma_val,
                       bool benchmark_mode, uint32_t invocation_id,
                       uint32_t batch, uint32_t core_id) {
  pack_init(cb_out);

  for (uint32_t t = 0; t < 10; ++t) {
    tile_regs_acquire();

#ifdef TRISC_MATH
    _llk_math_eltwise_sfpu_start_(0);

#if LDPC_TEST_PASSTHROUGH_MODE
    for (int r = 0; r < 32; ++r) {
      sfpi::dst_reg[r] = 0.0f;
    }
#else
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
                               (core_id * 1000003U) +
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

          // Remez Minimax Polynomial Evaluation (212x lower truncation error over [-pi, pi])
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

          sfpi::vFloat llr0 = v_mu + v_sigma * (r * cos_val);
          sfpi::vFloat llr1 = v_mu + v_sigma * (r * (sin_val * theta));

          // 50/50 Codeword Interleaving: Even p assigns (llr0, llr1); Odd p assigns (llr1, llr0).
          // Guarantees all 4 codewords receive an identical 50/50 blend of sine & cosine variates.
          if ((p % 2) == 1) {
            sfpi::dst_reg[2 * p + 0] = round_bf16(llr1);
            sfpi::dst_reg[2 * p + 1] = round_bf16(llr0);
          } else {
            sfpi::dst_reg[2 * p + 0] = round_bf16(llr0);
            sfpi::dst_reg[2 * p + 1] = round_bf16(llr1);
          }
        }
      }
    } else {
      // Punctured variable nodes: neutral zero LLR
      for (int r = 0; r < 32; ++r) {
        sfpi::dst_reg[r] = 0.0f;
      }
    }
#endif
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
// Main Kernel Entry Point
// ------------------------------------------------------------------------------
void kernel_main() {
  uint32_t num_batch_pairs = get_arg_val<uint32_t>(0);
  uint32_t max_iter        = get_arg_val<uint32_t>(1);
  uint32_t num_tiles       = get_arg_val<uint32_t>(2);
  uint32_t mu_bits         = get_arg_val<uint32_t>(3);
  uint32_t sigma_bits      = get_arg_val<uint32_t>(4);
  uint32_t invocation_id   = get_arg_val<uint32_t>(5);
  uint32_t diff_test_mode  = get_arg_val<uint32_t>(6);
  (void)diff_test_mode;
  uint32_t start_batch     = get_arg_val<uint32_t>(7);
  (void)start_batch;
  uint32_t core_id         = get_arg_val<uint32_t>(8);
  (void)core_id;

  if (invocation_id == 0) invocation_id = 1;
  uint32_t epoch = invocation_id << 16;

  float mu_val    = uint_as_float(mu_bits);
  float sigma_val = uint_as_float(sigma_bits);
  bool benchmark_mode = (sigma_val <= 0.0f);

  constexpr uint32_t cb_vnode_a       = TileMatrixCB::CB_VNODE_A;
  constexpr uint32_t cb_vnode_b       = TileMatrixCB::CB_VNODE_B;
  constexpr uint32_t cb_scratch_in_a  = TileMatrixCB::CB_SCRATCH_GATHER_0;
  constexpr uint32_t cb_scratch_in_b  = TileMatrixCB::CB_SCRATCH_GATHER_1;
  constexpr uint32_t cb_scratch_out_a = TileMatrixCB::CB_SCRATCH_SCATTER_0;
  constexpr uint32_t cb_scratch_out_b = TileMatrixCB::CB_SCRATCH_SCATTER_1;
  constexpr uint32_t cb_r_msg_a       = TileMatrixCB::CB_R_MSG_A;
  constexpr uint32_t cb_r_msg_b       = TileMatrixCB::CB_R_MSG_B;
  constexpr uint32_t cb_telem         = TileMatrixCB::CB_TELEMETRY;

  compute_kernel_hw_startup(cb_scratch_in_a, cb_scratch_out_a);
  copy_init(cb_scratch_in_a);

  uint32_t telem_l1 = ckernel::get_tile_address(cb_telem, 0);
  volatile uint32_t* sync_flags = reinterpret_cast<volatile uint32_t*>(telem_l1);
  volatile uint32_t* stats = sync_flags;

  uint32_t max_steps = max_iter * num_tiles;
  uint32_t total_steps = 2 * max_steps + 2;

  volatile uint32_t *const wall_clock_reg = reinterpret_cast<volatile uint32_t *>(0xFFB121F0);
  uint32_t cycles_compute = 0;
  uint32_t cycles_math_pure = 0;

  for (uint32_t pair = 0; pair < num_batch_pairs; ++pair) {
    // Generate Box-Muller AWGN channel observations for Batch A & Batch B
#ifdef TRISC_MATH
    WAYPOINT("M_CH");
#endif
    generate_channel_tiles(cb_vnode_a, mu_val, sigma_val, benchmark_mode, invocation_id, 2 * pair + 0, core_id);
    generate_channel_tiles(cb_vnode_b, mu_val, sigma_val, benchmark_mode, invocation_id, 2 * pair + 1, core_id);

    // Tri-core batch rendezvous barrier
#ifdef TRISC_PACK
    sync_flags[SyncSlot::BATCH_BARRIER_TRISC] = epoch + pair + 1;
    asm volatile("fence rw, rw" ::: "memory");
    WAYPOINT("P_02");
#endif
#ifdef TRISC_UNPACK
    WAYPOINT("U_03");
#endif
#ifdef TRISC_MATH
    WAYPOINT("M_03");
#endif
    while (sync_flags[SyncSlot::BATCH_BARRIER_NCRISC] < epoch + pair + 1) {
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
    while (sync_flags[SyncSlot::BATCH_BARRIER_BRISC] < epoch + pair + 1) {
      asm volatile("" ::: "memory");
    }
#ifdef TRISC_UNPACK
    WAYPOINT("U_05");
#endif
#ifdef TRISC_MATH
    WAYPOINT("M_05");
#endif
    while (sync_flags[SyncSlot::BATCH_BARRIER_TRISC] < epoch + pair + 1) {
      asm volatile("" ::: "memory");
    }

    // Wait for channel observations to be 100% ready in L1
#ifdef TRISC_UNPACK
    WAYPOINT("U_06");
#endif
#ifdef TRISC_MATH
    WAYPOINT("M_06");
#endif
#ifdef TRISC_PACK
    WAYPOINT("P_06");
#endif
    while (sync_flags[SyncSlot::BATCH_READY] < epoch + pair + 1) {
      asm volatile("" ::: "memory");
    }

#ifdef TRISC_MATH
    uint8_t active_mask_a = 0x0F;
    uint8_t active_mask_b = 0x0F;
    sync_flags[SyncSlot::ACTIVE_MASK_A]     = 0x0F;
    sync_flags[SyncSlot::ACTIVE_MASK_B]     = 0x0F;
    sync_flags[SyncSlot::PARITY_REPORT_W0]  = 0x0F;
    sync_flags[SyncSlot::PARITY_REPORT_W1]  = 0x0F;
    sync_flags[SyncSlot::EARLY_TERMINATION] = 0;
    sync_flags[SyncSlot::DRAIN_ACK_T0]      = pair;
    sync_flags[SyncSlot::DRAIN_ACK_W1]      = pair;
    sync_flags[SyncSlot::TRISC1_GRANT_STEP] = pair * total_steps + 1; // Grant Step 0
    sync_flags[SyncSlot::TRISC1_CMD]        = static_cast<uint32_t>(MaestroCmd::CMD_STEP_EXEC);
    asm volatile("fence rw, rw" ::: "memory");
    WAYPOINT("M_07");
#endif

#if defined(TRISC_UNPACK) || defined(TRISC_PACK)
    // Satellite Pre-Step Gate: Wait for TRISC1 Maestro to launch Step 0
#ifdef TRISC_UNPACK
    WAYPOINT("U_08");
#endif
#ifdef TRISC_PACK
    WAYPOINT("P_08");
#endif
    while (sync_flags[SyncSlot::TRISC1_CMD] != static_cast<uint32_t>(MaestroCmd::CMD_STEP_EXEC) ||
           sync_flags[SyncSlot::TRISC1_GRANT_STEP] < pair * total_steps + 1) {
      asm volatile("" ::: "memory");
    }
#endif

    uint32_t last_completed_k = 0;
    (void)last_completed_k;

    for (uint32_t k = 0; k < total_steps; ++k) {
      uint32_t bar = pair * total_steps + k + 1;
      uint32_t b_dae = k & 1;
      uint32_t s_dae = k >> 1;

#ifdef TRISC_UNPACK
      sync_flags[SyncSlot::DEBUG_TRISC0_STEP] = (k << 8) | 1; // Wait grant
#endif
#ifdef TRISC_PACK
      sync_flags[SyncSlot::DEBUG_TRISC2_STEP] = (k << 8) | 1; // Wait grant
#endif
#ifdef TRISC_MATH
      sync_flags[SyncSlot::DEBUG_TRISC1_STEP] = (k << 8) | 1; // Start step
#endif

#if defined(TRISC_UNPACK) || defined(TRISC_PACK)
#ifdef TRISC_UNPACK
      WAYPOINT("U_WT");
#endif
#ifdef TRISC_PACK
      WAYPOINT("P_WT");
#endif
      while (sync_flags[SyncSlot::TRISC1_GRANT_STEP] < bar) {
        asm volatile("" ::: "memory");
        if (sync_flags[SyncSlot::TRISC1_CMD] >= static_cast<uint32_t>(MaestroCmd::CMD_DRAIN_RESIDUAL)) {
          break;
        }
      }
      if (sync_flags[SyncSlot::TRISC1_CMD] >= static_cast<uint32_t>(MaestroCmd::CMD_DRAIN_RESIDUAL)) {
#ifdef TRISC_UNPACK
        sync_flags[SyncSlot::DEBUG_TRISC0_STEP] = (k << 8) | 0xEE;
        WAYPOINT("U_DR");
#endif
#ifdef TRISC_PACK
        sync_flags[SyncSlot::DEBUG_TRISC2_STEP] = (k << 8) | 0xEE;
        WAYPOINT("P_DR");
#endif
        break;
      }
#ifdef TRISC_UNPACK
      sync_flags[SyncSlot::DEBUG_TRISC0_STEP] = (k << 8) | 2; // Grant passed
      WAYPOINT("U_GR");
#endif
#ifdef TRISC_PACK
      sync_flags[SyncSlot::DEBUG_TRISC2_STEP] = (k << 8) | 2; // Grant passed
      WAYPOINT("P_GR");
#endif
#endif

      // Compute Stage: Compute check nodes for tile (k - 1)
      if (k >= 1 && k <= 2 * max_steps) {
        uint32_t b_comp = (k - 1) % 2;
        uint32_t s_comp = (k - 1) / 2;
        uint32_t t_comp = s_comp % num_tiles;
        uint32_t iter_comp = s_comp / num_tiles;

        uint32_t cb_in  = (b_comp == 0) ? cb_scratch_in_a  : cb_scratch_in_b;
        uint32_t cb_out = (b_comp == 0) ? cb_scratch_out_a : cb_scratch_out_b;
        uint32_t cb_r   = (b_comp == 0) ? cb_r_msg_a       : cb_r_msg_b;

#ifdef TRISC_UNPACK
        sync_flags[SyncSlot::DEBUG_TRISC0_STEP] = (k << 8) | 3; // In compute
        WAYPOINT("U_CP");
#endif
#ifdef TRISC_MATH
        sync_flags[SyncSlot::DEBUG_TRISC1_STEP] = (k << 8) | 2; // In compute
        WAYPOINT("M_CP");
#endif
#ifdef TRISC_PACK
        sync_flags[SyncSlot::DEBUG_TRISC2_STEP] = (k << 8) | 3; // In compute
        WAYPOINT("P_CP");
#endif

        uint32_t t_c0 = *wall_clock_reg;
        uint32_t m_cyc = 0;
        bool is_deg6 = (t_comp < QC_LDPC_4CW::DEG6_TILES);
        if (is_deg6) {
          m_cyc = compute_tile_deg6(cb_in, cb_out, cb_r, iter_comp);
        } else {
          m_cyc = compute_tile_deg3(cb_in, cb_out, cb_r, iter_comp);
        }
#ifdef TRISC_MATH
        cycles_math_pure += m_cyc;
#endif
#ifdef TRISC_PACK
        cycles_compute += (*wall_clock_reg - t_c0);
#endif
      }

#ifdef TRISC_UNPACK
      sync_flags[SyncSlot::DEBUG_TRISC0_STEP] = (k << 8) | 9; // Step done
      WAYPOINT("U_DN");
      last_completed_k = k;
#endif
#ifdef TRISC_PACK
      sync_flags[SyncSlot::DEBUG_TRISC2_STEP] = (k << 8) | 9; // Step done
      WAYPOINT("P_DN");
#endif

#ifdef TRISC_MATH
      // Poll satellites for step k
      if (s_dae >= 1 && (s_dae - 1) < max_steps) {
        sync_flags[SyncSlot::DEBUG_TRISC1_STEP] = (k << 8) | 3; // Wait SCATTER_DONE
        WAYPOINT("M_WS");
        while (sync_flags[SyncSlot::SCATTER_DONE_W0] < bar) {
          asm volatile("" ::: "memory");
        }
        while (sync_flags[SyncSlot::SCATTER_DONE_W1] < bar) {
          asm volatile("" ::: "memory");
        }
      }
      if (s_dae < max_steps) {
        sync_flags[SyncSlot::DEBUG_TRISC1_STEP] = (k << 8) | 4; // Wait GATHER_DONE
        WAYPOINT("M_WG");
        while (sync_flags[SyncSlot::GATHER_DONE_W0] < bar) {
          asm volatile("" ::: "memory");
        }
        while (sync_flags[SyncSlot::GATHER_DONE_W1] < bar) {
          asm volatile("" ::: "memory");
        }
      }

      // Check Iteration Boundary & Parity Evaluation (Tile 39)
      bool early_exit = false;
      if (s_dae >= 1 && (s_dae - 1) < max_steps) {
        uint32_t s_scat = s_dae - 1;
        uint32_t t_scat = s_scat % num_tiles;
        if (t_scat == num_tiles - 1) {
          uint32_t iter_scat = s_scat / num_tiles;
          uint32_t info_errs   = sync_flags[SyncSlot::PARITY_REPORT_W0];
          uint32_t parity_errs = sync_flags[SyncSlot::PARITY_REPORT_W1];
          uint32_t total_cw_errs = info_errs | parity_errs;

          // Mandatory Invariant: Immediately re-arm mailboxes to non-converged fail-safe values
          sync_flags[SyncSlot::PARITY_REPORT_W0] = 0x0F;
          sync_flags[SyncSlot::PARITY_REPORT_W1] = 0x0F;
          asm volatile("fence rw, rw" ::: "memory");

          if (b_dae == 0) {
            for (uint32_t c = 0; c < 4; ++c) {
              if (((total_cw_errs & (1 << c)) == 0) && (active_mask_a & (1 << c))) {
                active_mask_a &= ~(1 << c);
                stats[TelemSlot::LAST_BATCH_ITER_CW0 + c] = iter_scat + 1;
              }
            }
            sync_flags[SyncSlot::ACTIVE_MASK_A] = active_mask_a;
          } else {
            for (uint32_t c = 0; c < 4; ++c) {
              if (((total_cw_errs & (1 << c)) == 0) && (active_mask_b & (1 << c))) {
                active_mask_b &= ~(1 << c);
                stats[TelemSlot::LAST_BATCH_ITER_CW0 + 4 + c] = iter_scat + 1;
              }
            }
            sync_flags[SyncSlot::ACTIVE_MASK_B] = active_mask_b;
          }
          asm volatile("fence rw, rw" ::: "memory");

          if (active_mask_a == 0 && active_mask_b == 0) {
            sync_flags[SyncSlot::EARLY_TERMINATION] = 1;
            sync_flags[SyncSlot::TRISC1_CMD] = static_cast<uint32_t>(MaestroCmd::CMD_DRAIN_RESIDUAL);
            asm volatile("fence rw, rw" ::: "memory");
            early_exit = true;
          }
        }
      }
      if (early_exit) {
        sync_flags[SyncSlot::DEBUG_TRISC1_STEP] = (k << 8) | 0xEE; // Drain exit
        WAYPOINT("M_DR");
        while (sync_flags[SyncSlot::DRAIN_ACK_T0] < pair + 1) {
          asm volatile("" ::: "memory");
        }
        while (sync_flags[SyncSlot::DRAIN_ACK_W1] < pair + 1) {
          asm volatile("" ::: "memory");
        }
        sync_flags[SyncSlot::TRISC1_CMD] = static_cast<uint32_t>(MaestroCmd::CMD_BATCH_DONE);
        asm volatile("fence rw, rw" ::: "memory");
        break;
      }

      sync_flags[SyncSlot::DEBUG_TRISC1_STEP] = (k << 8) | 5; // Granting next step
      if (k + 1 < total_steps) {
        sync_flags[SyncSlot::TRISC1_GRANT_STEP] = pair * total_steps + (k + 1) + 1;
        asm volatile("fence rw, rw" ::: "memory");
        WAYPOINT("M_GN");
      } else {
        // Normal schedule complete: broadcast CMD_BATCH_DONE first so satellites break out of step loop
        sync_flags[SyncSlot::TRISC1_CMD] = static_cast<uint32_t>(MaestroCmd::CMD_BATCH_DONE);
        asm volatile("fence rw, rw" ::: "memory");
        while (sync_flags[SyncSlot::DRAIN_ACK_T0] < pair + 1) {
          asm volatile("" ::: "memory");
        }
        while (sync_flags[SyncSlot::DRAIN_ACK_W1] < pair + 1) {
          asm volatile("" ::: "memory");
        }
      }
      sync_flags[SyncSlot::DEBUG_TRISC1_STEP] = (k << 8) | 9; // Step done
#endif
    }

    // End of Dual-Batch Pair: Drain In-Flight FIFOs and Synchronize
#ifdef TRISC_UNPACK
    // Drain speculative residual input tile if early termination interrupted pipeline
    if (sync_flags[SyncSlot::TRISC1_CMD] == static_cast<uint32_t>(MaestroCmd::CMD_DRAIN_RESIDUAL)) {
      if (last_completed_k >= 0 && last_completed_k < 2 * max_steps) {
        uint32_t b_res = last_completed_k % 2;
        uint32_t res_in = (b_res == 0) ? cb_scratch_in_a : cb_scratch_in_b;
        cb_wait_front(res_in, 1);
        cb_pop_front(res_in, 1);
      }
    }
    // Always drain circular FIFO check messages so queues are clean for next batch
    for (uint32_t t = 0; t < num_tiles; ++t) {
      cb_wait_front(cb_r_msg_a, 1);
      cb_pop_front(cb_r_msg_a, 1);
      cb_wait_front(cb_r_msg_b, 1);
      cb_pop_front(cb_r_msg_b, 1);
    }
    sync_flags[SyncSlot::DRAIN_ACK_T0] = pair + 1;
    asm volatile("fence rw, rw" ::: "memory");
#endif

#ifdef TRISC_PACK
    while (sync_flags[SyncSlot::TRISC1_CMD] < static_cast<uint32_t>(MaestroCmd::CMD_BATCH_DONE)) {
      asm volatile("" ::: "memory");
    }
#endif
  }

#ifdef TRISC_MATH
  sync_flags[SyncSlot::TRISC_MATH_CYCLES] = cycles_math_pure;
#endif
#ifdef TRISC_PACK
  sync_flags[SyncSlot::TRISC_CYCLES] = cycles_compute;
  sync_flags[SyncSlot::BATCH_BARRIER_TRISC] = epoch + num_batch_pairs + 1;
#endif
}
