#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "ldpc_tile_matrix_core.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "sfpi.h"
#include <cstdint>

// ==============================================================================
// Working 440 code 8CW pipelined simulation, still needs orchestration optimization and FER performance analysis
//
// Tenstorrent QuietBox 2: Tile Matrix 4-Codeword LDPC Decoder Engine
// Component: Tri-TRISC Compute & Centralized Macro-Orchestrator Kernel
// Architecture: Centralized TRISC1 Maestro with Bounded Satellite Autonomy (DAE)
// Target: Tenstorrent Blackhole (440 Tensix Cores, 2,200 RISC-V Processors)
// Specification: docs/MD_files/CENTRALIZED_TRISC1_ORCHESTRATOR_AND_BOUNDED_AUTONOMY_SPECIFICATION.md
//
// Key Principles:
// 1. Hub-and-Spoke Orchestration: TRISC1 (MATH) is the fastest compute engine (~780 cyc)
//    and acts as the sole Maestro, governing macro-step grants (TRISC1_GRANT_STEP) and
//    command dispatch (TRISC1_CMD: CMD_STEP_EXEC, CMD_DRAIN_RESIDUAL, CMD_BATCH_DONE).
// 2. Bounded Satellite Autonomy: TRISC0 (UNPACK) & TRISC2 (PACK) execute intra-tile
//    gather/scatter and unpacking/packing 100% autonomously, but have zero authority
//    to advance steps or early-terminate without TRISC1 grant.
// 3. Inter-Pair Quiescence & Pre-Step Gate: TRISC1_CMD is reset to CMD_IDLE before
//    the tri-core batch rendezvous barrier. Satellites wait for CMD_STEP_EXEC and
//    GRANT_STEP >= pair*total_steps+1 before launching Step 0, preventing premature
//    termination on stale command residual flags.
// ==============================================================================

#ifndef LDPC_ENABLE_CH_CLAMP
#define LDPC_ENABLE_CH_CLAMP 0
#endif

#ifndef LDPC_L_CH_MAX
#define LDPC_L_CH_MAX 7.0f
#endif

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
// Computes AMin*(a, b) = min(a, b) - g(|a - b|) without branch instructions.
// Supports both:
//   1. Certified 2-Piece Baseline: Dr. Jones (MILCOM 2003) 2-region piecewise linear.
//   2. Optimal 3-Piece Convex Hull (LDPC_USE_3PIECE_AMIN_STAR):
//      Approximates the continuous Jacobi logarithm g(Δ) = ln(1 + e^-Δ) across 3 linear
//      tangents with 6.68x lower max error and 14.5x lower check message error:
//        Piece 1 [0.00, 0.85): y = 0.67862 - 0.39622 * Δ  (bf16: 0x3F2D, 0xBECA)
//        Piece 2 [0.85, 2.20): y = 0.48710 - 0.18297 * Δ  (bf16: 0x3EF9, 0xBE3B)
//        Piece 3 [2.20, 4.35): y = 0.17614 - 0.04029 * Δ  (bf16: 0x3E34, 0xBD25)
//        Tail    [4.35,  inf): y = 0.0
//
// MICROARCHITECTURAL REGISTER INVARIANT:
//   Fold each linear segment sequentially into `corr` (`corr = sfpi::max(corr, line)`).
//   This keeps concurrent active LRegs <= 4, preventing GCC compiler spilling and
//   eliminating 'too few lregs' errors during TRISC LTO code generation.
// ------------------------------------------------------------------------------
sfpi_inline sfpi::vFloat sfpi_amin_star_branchless(sfpi::vFloat a,
                                                   sfpi::vFloat b) {
#if defined(LDPC_USE_SFPU_LUT)
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
#elif defined(LDPC_USE_3PIECE_AMIN_STAR)
  auto [min_v, max_v] = sfpi::min_max(a, b);
  sfpi::vFloat diff = max_v - min_v;
  // Fold sequentially to keep active LReg pressure <= 4 registers
  sfpi::vFloat corr = sfpi::vFloat(sfpi::sFloat16b(0.67862f)) +
                      sfpi::vFloat(sfpi::sFloat16b(-0.39622f)) * diff;
  corr = sfpi::max(corr, sfpi::vFloat(sfpi::sFloat16b(0.48710f)) +
                         sfpi::vFloat(sfpi::sFloat16b(-0.18297f)) * diff);
  corr = sfpi::max(corr, sfpi::vFloat(sfpi::sFloat16b(0.17614f)) +
                         sfpi::vFloat(sfpi::sFloat16b(-0.04029f)) * diff);
  corr = sfpi::max(corr, 0.0f);
  return sfpi::max(min_v - corr, 0.0f);
#else
  auto [min_v, max_v] = sfpi::min_max(a, b);
  sfpi::vFloat diff = max_v - min_v;
  sfpi::vFloat line1 = sfpi::vFloat(sfpi::sFloat16b(0.69315f)) +
                       sfpi::vFloat(sfpi::sFloat16b(-0.4000f)) * diff;
  sfpi::vFloat line2 = sfpi::vFloat(sfpi::sFloat16b(0.65715f)) +
                       sfpi::vFloat(sfpi::sFloat16b(-0.3280f)) * diff;
  sfpi::vFloat corr = sfpi::max(sfpi::max(line1, line2), 0.0f);
  return sfpi::max(min_v - corr, 0.0f);
#endif
}

// ------------------------------------------------------------------------------
// Bit-Exact IEEE Round-to-Nearest-Even (RNE) on Tensix SFPU (bfloat16)
// Bypasses SFPSTOCHRND hardware bug bias; matches host stepper identically.
// ------------------------------------------------------------------------------
// Full RNE with bitmask: required when rounded value is reused in an FP32 LReg expression
sfpi_inline sfpi::vFloat round_bf16_rne(sfpi::vFloat val) {
  sfpi::vUInt u = sfpi::as<sfpi::vUInt>(val);
  sfpi::vUInt lsb = (u >> 16) & 1U;
  sfpi::vUInt bias = lsb + 0x7FFFU;
  u += bias;
  u &= 0xFFFF0000U;
  return sfpi::as<sfpi::vFloat>(u);
}
#define round_bf16(v) round_bf16_rne(v)

// Fast RNE for values written directly to 16-bit DEST (sfpstore drops bits [15:0] in hardware).
// Saves 2 instructions per round (sfploadi and sfpand).
sfpi_inline sfpi::vFloat round_bf16_fast(sfpi::vFloat val) {
  sfpi::vUInt u = sfpi::as<sfpi::vUInt>(val);
  sfpi::vUInt lsb = (u >> 16) & 1U;
  sfpi::vUInt bias = lsb + 0x7FFFU;
  u += bias;
  return sfpi::as<sfpi::vFloat>(u);
}

// ------------------------------------------------------------------------------
// CRITICAL HARDWARE INVARIANT: SFPU Subvector Transpose Register Clobber
// Warning
// ------------------------------------------------------------------------------
// SFPTRANSP (subvec_transp) implicitly hard-wires L0..L3 and clobbers L4..L7.
// The GCC compiler (riscv-tt-elf-gcc) DOES NOT model this clobber. Therefore:
// 1. NEVER pass an existing live local variable into swap_subvec16 if its value
// is
//    needed downstream (it will be silently corrupted in-place).
// 2. ALWAYS invoke swap_subvec16 as the first line of a block directly on
// dst_reg.
// 3. ALWAYS freshly reload unswapped operands from dst_reg AFTER swap_subvec16.
// 4. Staging across multiple transpositions MUST use DEST registers
// (dst_reg[sc0]). See
// docs/MD_files/SFPU_LREG_HARDWARE_CLOBBER_AND_COMPILER_DIRECTIVE_SOLUTIONS.md
// ------------------------------------------------------------------------------
sfpi_inline sfpi::vInt swap_subvec16(sfpi::vInt vi) {
  sfpi::vInt a = vi, b = vi, c = vi, d = vi;
  sfpi::subvec_transp(a, b, c, d);
  sfpi::subvec_transp(c, d, a, b);
  return c;
}

sfpi_inline sfpi::vFloat swap_subvec16(sfpi::vFloat vf) {
  sfpi::vFloat a = vf, b = vf, c = vf, d = vf;
  sfpi::subvec_transp(a, b, c, d);
  sfpi::subvec_transp(c, d, a, b);
  return c;
}

// ------------------------------------------------------------------------------
// Isolated Scope: 32-Lane Vector Approximate-Min* for Degree-6 Tile (5 Blocks)
// Directly outputs r_out to eliminate subtraction cancellation!
// ------------------------------------------------------------------------------
void __attribute__((noinline)) compute_tile_deg6(uint32_t cb_in,
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
  _llk_math_eltwise_sfpu_start_(0);

#if LDPC_TEST_PASSTHROUGH_MODE
  // In pass-through mode, Tile 0 is already in dst_reg[0..31] from
  // copy_tile(cb_in, 0, 0).
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
      int p0 = base_r + 0 + sub;
      int p1 = base_r + 2 + sub;
      int p2 = base_r + 4 + sub;

      int sc0 = base_r + 6;      // Tile 0 scratch 0: holds s_tot
      int sc1 = base_r + 7;      // Tile 0 scratch 1: holds M01
      int sc2 = 32 + base_r + 6; // Tile 1 scratch 0: holds M23
      int sc3 = 32 + base_r + 7; // Tile 1 scratch 1: holds M45

      // Stage 0: Q = L - R_old in-place in Tile 0 (iter > 0)
      if (iter != 0) {
        sfpi::dst_reg[p0] = round_bf16_fast(sfpi::vFloat(sfpi::dst_reg[p0]) -
                                             sfpi::vFloat(sfpi::dst_reg[32 + p0]));
        sfpi::dst_reg[p1] = round_bf16_fast(sfpi::vFloat(sfpi::dst_reg[p1]) -
                                             sfpi::vFloat(sfpi::dst_reg[32 + p1]));
        sfpi::dst_reg[p2] = round_bf16_fast(sfpi::vFloat(sfpi::dst_reg[p2]) -
                                             sfpi::vFloat(sfpi::dst_reg[32 + p2]));
      }

      // Stage 1: Transpose raw Q vectors directly into dst_reg[32 + p*]
      // (reusing Tile 1 as staging). Exactly 3 swaps per sub-quartet (down from
      // 9)! SFPTRANSP operates on L0..L3 and clobbers L4..L7, but ZERO C++
      // variables are live in LRegs here.
      sfpi::dst_reg[32 + p0] = swap_subvec16(sfpi::vFloat(sfpi::dst_reg[p0]));
      sfpi::dst_reg[32 + p1] = swap_subvec16(sfpi::vFloat(sfpi::dst_reg[p1]));
      sfpi::dst_reg[32 + p2] = swap_subvec16(sfpi::vFloat(sfpi::dst_reg[p2]));

      // Stage 2: 1-Cycle Native Integer XOR sign parity -> stored in
      // dst_reg[sc0]
      {
        sfpi::vInt s0 =
            sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[p0])) ^
            sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[32 + p0]));
        sfpi::vInt s1 =
            sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[p1])) ^
            sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[32 + p1]));
        sfpi::vInt s2 =
            sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[p2])) ^
            sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[32 + p2]));
        sfpi::dst_reg[sc0] =
            sfpi::as<sfpi::vFloat>((s0 ^ s1) ^ s2); // s_tot in sc0
      }

      // Stage 3: Pairwise AMin* within edge pairs -> stored in sc1 (M01), sc2
      // (M23), sc3 (M45)
      {
        sfpi::vFloat m0 = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[p0]));
        sfpi::vFloat m0_sw = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[32 + p0]));
        sfpi::dst_reg[sc1] =
            round_bf16_fast(sfpi_amin_star_branchless(m0, m0_sw)); // M01
      }
      {
        sfpi::vFloat m1 = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[p1]));
        sfpi::vFloat m1_sw = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[32 + p1]));
        sfpi::dst_reg[sc2] =
            round_bf16_fast(sfpi_amin_star_branchless(m1, m1_sw)); // M23
      }
      {
        sfpi::vFloat m2 = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[p2]));
        sfpi::vFloat m2_sw = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[32 + p2]));
        sfpi::dst_reg[sc3] =
            round_bf16_fast(sfpi_amin_star_branchless(m2, m2_sw)); // M45
      }

      // Stage 4: Outgoing check messages with signs applied via SFPSETSGN (1 cycle)
      // Single-load Q reuse: q* is loaded once from DEST into an LReg and shared between
      // the 1-cycle sign XOR parity and posterior accumulation (L_new = q* + r_out*).
      // Eliminates redundant SFPLOAD instructions and RV32 instrn_buffer writes, enabling
      // Blackhole hardware instruction replay loops (ttreplay) to engage.
      // Edge pair 0 (p0)
      {
        sfpi::vFloat m_oth0 = sfpi_amin_star_branchless(
            sfpi::vFloat(sfpi::dst_reg[sc2]), sfpi::vFloat(sfpi::dst_reg[sc3]));
        sfpi::vFloat m0_sw = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[32 + p0]));
        sfpi::vFloat r_mag0 = sfpi_amin_star_branchless(m_oth0, m0_sw);
        sfpi::vFloat q0 = sfpi::dst_reg[p0];
        sfpi::vInt s_edge0 =
            sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc0])) ^
            sfpi::as<sfpi::vInt>(q0);
        sfpi::vFloat r_out0 = round_bf16(sfpi::copysgn(r_mag0, s_edge0));
        sfpi::dst_reg[32 + p0] = round_bf16_fast(q0 + r_out0); // L_new in Tile 1
        sfpi::dst_reg[p0] = r_out0;                            // R_new in Tile 0
      }

      // Edge pair 1 (p1)
      {
        sfpi::vFloat m_oth1 = sfpi_amin_star_branchless(
            sfpi::vFloat(sfpi::dst_reg[sc1]), sfpi::vFloat(sfpi::dst_reg[sc3]));
        sfpi::vFloat m1_sw = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[32 + p1]));
        sfpi::vFloat r_mag1 = sfpi_amin_star_branchless(m_oth1, m1_sw);
        sfpi::vFloat q1 = sfpi::dst_reg[p1];
        sfpi::vInt s_edge1 =
            sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc0])) ^
            sfpi::as<sfpi::vInt>(q1);
        sfpi::vFloat r_out1 = round_bf16(sfpi::copysgn(r_mag1, s_edge1));
        sfpi::dst_reg[32 + p1] = round_bf16_fast(q1 + r_out1); // L_new in Tile 1
        sfpi::dst_reg[p1] = r_out1;                            // R_new in Tile 0
      }

      // Edge pair 2 (p2)
      {
        sfpi::vFloat m_oth2 = sfpi_amin_star_branchless(
            sfpi::vFloat(sfpi::dst_reg[sc1]), sfpi::vFloat(sfpi::dst_reg[sc2]));
        sfpi::vFloat m2_sw = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[32 + p2]));
        sfpi::vFloat r_mag2 = sfpi_amin_star_branchless(m_oth2, m2_sw);
        sfpi::vFloat q2 = sfpi::dst_reg[p2];
        sfpi::vInt s_edge2 =
            sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc0])) ^
            sfpi::as<sfpi::vInt>(q2);
        sfpi::vFloat r_out2 = round_bf16(sfpi::copysgn(r_mag2, s_edge2));
        sfpi::dst_reg[32 + p2] = round_bf16_fast(q2 + r_out2); // L_new in Tile 1
        sfpi::dst_reg[p2] = r_out2;                            // R_new in Tile 0
      }
    }

    // Zero out unused padding / scratch registers
    sfpi::dst_reg[base_r + 6] = 0.0f;
    sfpi::dst_reg[base_r + 7] = 0.0f;
    sfpi::dst_reg[32 + base_r + 6] = 0.0f;
    sfpi::dst_reg[32 + base_r + 7] = 0.0f;
  }
#endif

  _llk_math_eltwise_sfpu_done_();
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
}

sfpi_inline sfpi::vInt mux16_int(sfpi::vInt lo, sfpi::vInt hi,
                                 sfpi::vInt mask) {
  sfpi::vInt diff = lo ^ hi;
  diff &= mask;
  return lo ^ diff;
}

sfpi_inline sfpi::vFloat mux16_flt(sfpi::vFloat lo, sfpi::vFloat hi,
                                   sfpi::vInt mask) {
  sfpi::vInt ilo = sfpi::as<sfpi::vInt>(lo);
  sfpi::vInt ihi = sfpi::as<sfpi::vInt>(hi);
  sfpi::vInt diff = ilo ^ ihi;
  diff &= mask;
  return sfpi::as<sfpi::vFloat>(ilo ^ diff);
}

// ------------------------------------------------------------------------------
// Isolated Scope: 32-Lane Vector Approximate-Min* for Degree-3 Tile (8 Passes)
// Directly outputs r_out to eliminate subtraction cancellation!
// ------------------------------------------------------------------------------
void __attribute__((noinline)) compute_tile_deg3(uint32_t cb_in,
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
  _llk_math_eltwise_sfpu_start_(0);

#if LDPC_TEST_PASSTHROUGH_MODE
  // In pass-through mode, Tile 0 is already in dst_reg[0..31] from
  // copy_tile(cb_in, 0, 0).
#elif LDPC_TEST_R_DATAFLOW_MODE
  const int face_bases[4] = {0, 8, 16, 24};
#pragma GCC unroll 4
  for (int f = 0; f < 4; ++f) {
    int base = face_bases[f];
#pragma GCC unroll 3
    for (int k = 0; k < 3; ++k) {
#pragma GCC unroll 2
      for (int sub = 0; sub < 2; ++sub) {
        int reg = base + 2 * k + sub;
        sfpi::vFloat l_val = sfpi::dst_reg[reg];
        sfpi::vFloat r_old = 0.0f;
        if (iter != 0) {
          r_old = sfpi::dst_reg[32 + reg];
        }
        sfpi::vFloat q = l_val - r_old;
        sfpi::vFloat r_new = swap_subvec16(q);
        sfpi::vFloat l_new = q + r_new;
        sfpi::dst_reg[32 + reg] = l_new; // L_new in Tile 1
        sfpi::dst_reg[reg] = r_new;      // R_new in Tile 0
      }
    }
    // Zero out padding registers: regs 6, 7 of this face
    sfpi::dst_reg[base + 6] = 0.0f;
    sfpi::dst_reg[base + 7] = 0.0f;
    sfpi::dst_reg[32 + base + 6] = 0.0f;
    sfpi::dst_reg[32 + base + 7] = 0.0f;
  }
#else
  const int face_bases[4] = {0, 8, 16, 24};

#pragma GCC unroll 4
  for (int f = 0; f < 4; ++f) {
    int base_r = face_bases[f];

#pragma GCC unroll 1
    for (int sub = 0; sub < 2; ++sub) {
      int p0 = base_r + 0 + sub;
      int p1 = base_r + 2 + sub;
      int p2 = base_r + 4 + sub;

      int sc0 = base_r + 6; // Tile 0 scratch 0: holds q0_sw
      int sc1 =
          base_r + 7; // Tile 0 scratch 1: holds q1_sw -> q_e2_blk0 -> r_out_e2
      int sc2 = 32 + base_r + 6; // Tile 1 scratch 0: holds q2_sw
      int sc3 =
          32 + base_r + 7; // Tile 1 scratch 1: holds q_e0_blk1 -> r_out_e0

      // Stage 0: Q = L - R_old in-place in Tile 0 (iter > 0)
      if (iter != 0) {
        sfpi::dst_reg[p0] = round_bf16_fast(sfpi::vFloat(sfpi::dst_reg[p0]) -
                                             sfpi::vFloat(sfpi::dst_reg[32 + p0]));
        sfpi::dst_reg[p1] = round_bf16_fast(sfpi::vFloat(sfpi::dst_reg[p1]) -
                                             sfpi::vFloat(sfpi::dst_reg[32 + p1]));
        sfpi::dst_reg[p2] = round_bf16_fast(sfpi::vFloat(sfpi::dst_reg[p2]) -
                                             sfpi::vFloat(sfpi::dst_reg[32 + p2]));
      }

      // Stage 1: Transpose raw Q vectors directly into DEST scratch registers.
      // SFPTRANSP implicitly operates on L0..L3 and clobbers L4..L7.
      // Calling directly from dst_reg ensures ZERO C++ variables are live in
      // LRegs!
      sfpi::dst_reg[sc0] = swap_subvec16(sfpi::vFloat(sfpi::dst_reg[p0]));
      sfpi::dst_reg[sc1] = swap_subvec16(sfpi::vFloat(sfpi::dst_reg[p1]));
      sfpi::dst_reg[sc2] = swap_subvec16(sfpi::vFloat(sfpi::dst_reg[p2]));

      // Broadcast E2 of Block 0 into sc1, and E0 of Block 1 into sc3
      {
        sfpi::vFloat q1 = sfpi::dst_reg[p1];
        sfpi::vFloat q1_sw = sfpi::dst_reg[sc1];

        sfpi::vFloat q_e2 = q1;
        v_if(sfpi::as<sfpi::vUInt>(sfpi::vConstTileId) >= 32) { q_e2 = q1_sw; }
        v_endif;
        sfpi::dst_reg[sc1] = q_e2;

        sfpi::vFloat q_e0 = q1;
        v_if(sfpi::as<sfpi::vUInt>(sfpi::vConstTileId) < 32) { q_e0 = q1_sw; }
        v_endif;
        sfpi::dst_reg[sc3] = q_e0;
      }

      // Step 2a: Block 0 sign parity -> store s_tot in dst_reg[32 + p0]
      {
        sfpi::vInt s0    = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[p0]));
        sfpi::vInt s0_sw = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc0]));
        sfpi::vInt s_e2  = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc1]));
        sfpi::dst_reg[32 + p0] = sfpi::as<sfpi::vFloat>((s0 ^ s0_sw) ^ s_e2);
      }

      // Step 2b: Compute r_out_e2 -> stage in dst_reg[32 + p1]
      {
        sfpi::vFloat m0    = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[p0]));
        sfpi::vFloat m0_sw = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[sc0]));
        sfpi::vFloat r_mag_e2 = sfpi_amin_star_branchless(m0, m0_sw);
        sfpi::vInt s_tot = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[32 + p0]));
        sfpi::vInt s_e2  = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc1]));
        sfpi::dst_reg[32 + p1] = round_bf16_fast(sfpi::copysgn(r_mag_e2, s_tot ^ s_e2));
      }

      // Step 2c: Compute r_out0 -> write R_new to p0, L_new to 32 + p0
      {
        sfpi::vFloat m0_sw = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[sc0]));
        sfpi::vFloat m_e2  = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[sc1]));
        sfpi::vFloat r_mag0 = sfpi_amin_star_branchless(m0_sw, m_e2);
        sfpi::vFloat q0     = sfpi::dst_reg[p0];
        sfpi::vInt s_tot    = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[32 + p0]));
        sfpi::vInt s0       = sfpi::as<sfpi::vInt>(q0);
        sfpi::vFloat r_out0 = round_bf16(sfpi::copysgn(r_mag0, s_tot ^ s0));
        sfpi::dst_reg[32 + p0] = round_bf16_fast(q0 + r_out0); // L_new in Tile 1
        sfpi::dst_reg[p0]      = r_out0;                       // R_new in Tile 0
      }

      // Step 3a: Block 1 sign parity -> store s_tot in dst_reg[32 + p2]
      {
        sfpi::vInt s2    = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[p2]));
        sfpi::vInt s2_sw = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc2]));
        sfpi::vInt s_e0  = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc3]));
        sfpi::dst_reg[32 + p2] = sfpi::as<sfpi::vFloat>((s2 ^ s2_sw) ^ s_e0);
      }

      // Step 3b: Compute r_out_e0 -> stage in dst_reg[sc0]
      {
        sfpi::vFloat m2    = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[p2]));
        sfpi::vFloat m2_sw = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[sc2]));
        sfpi::vFloat r_mag_e0 = sfpi_amin_star_branchless(m2, m2_sw);
        sfpi::vInt s_tot = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[32 + p2]));
        sfpi::vInt s_e0  = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc3]));
        sfpi::dst_reg[sc0] = round_bf16_fast(sfpi::copysgn(r_mag_e0, s_tot ^ s_e0));
      }

      // Step 3c: Compute r_out2 -> write R_new to p2, L_new to 32 + p2
      {
        sfpi::vFloat m2_sw = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[sc2]));
        sfpi::vFloat m_e0  = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[sc3]));
        sfpi::vFloat r_mag2 = sfpi_amin_star_branchless(m2_sw, m_e0);
        sfpi::vFloat q2     = sfpi::dst_reg[p2];
        sfpi::vInt s_tot    = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[32 + p2]));
        sfpi::vInt s2       = sfpi::as<sfpi::vInt>(q2);
        sfpi::vFloat r_out2 = round_bf16(sfpi::copysgn(r_mag2, s_tot ^ s2));
        sfpi::dst_reg[32 + p2] = round_bf16_fast(q2 + r_out2); // L_new in Tile 1
        sfpi::dst_reg[p2]      = r_out2;                       // R_new in Tile 0
      }

      // Stage 4: Assemble p1 (lower 16 lanes is r_out_e2, upper 16 lanes is r_out_e0)
      {
        sfpi::vFloat r_out1 = sfpi::dst_reg[32 + p1];
        v_if(sfpi::as<sfpi::vUInt>(sfpi::vConstTileId) >= 32) {
          r_out1 = sfpi::vFloat(sfpi::dst_reg[sc0]);
        }
        v_endif;

        sfpi::dst_reg[32 + p1] = round_bf16_fast(sfpi::vFloat(sfpi::dst_reg[p1]) +
                                                 r_out1); // L_new in Tile 1
        sfpi::dst_reg[p1] = r_out1;                       // R_new in Tile 0
      }
    }

    // Zero padding / scratch registers
    sfpi::dst_reg[base_r + 6] = 0.0f;
    sfpi::dst_reg[base_r + 7] = 0.0f;
    sfpi::dst_reg[32 + base_r + 6] = 0.0f;
    sfpi::dst_reg[32 + base_r + 7] = 0.0f;
  }
#endif

  _llk_math_eltwise_sfpu_done_();
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

// ------------------------------------------------------------------------------
// 32-Lane Vector SFPU Channel Observation Generator
// Generates 10 tiles (8 AWGN / Benchmark tiles + 2 Punctured zero tiles)
// Directly streams into CB_VNODE_A in native Tensix face quadrant tile format
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

        // Phase 2: Transcendental Box-Muller Transformation (PRNG is dead, 0
        // LReg pressure)
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
          // Legacy: Taylor Series Evaluation (Requires
          // -DLDPC_USE_LEGACY_TAYLOR_TRIG)
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
          // Default: Remez Minimax Polynomial Evaluation (212x lower truncation
          // error over [-pi, pi])
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
      // Stage 2: Generate 2 Punctured Zero Tiles (2 x 1024 = 2,048 Zeros)
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

// Main Tri-TRISC Kernel Execution
// ==============================================================================
void kernel_main() {
  uint32_t num_batch_pairs = get_arg_val<uint32_t>(0); // number of 8-codeword pairs
  uint32_t max_iter        = get_arg_val<uint32_t>(1);
  uint32_t num_tiles       = get_arg_val<uint32_t>(2);
  uint32_t mu_bits         = get_arg_val<uint32_t>(3);
  uint32_t sigma_bits      = get_arg_val<uint32_t>(4);
  uint32_t invocation_id   = get_arg_val<uint32_t>(5);
  uint32_t start_batch     = get_arg_val<uint32_t>(6);
  uint32_t core_id         = get_arg_val<uint32_t>(7);
  uint32_t global_seed     = get_arg_val<uint32_t>(8);
  if (invocation_id == 0)
    invocation_id = 1;
  uint32_t epoch = invocation_id << 16;

  bool benchmark_mode = (sigma_bits == 0);
  union {
    uint32_t u;
    float f;
  } mu_conv, sigma_conv;
  mu_conv.u = mu_bits;
  sigma_conv.u = sigma_bits;
  float mu_val = mu_conv.f;
  float sigma_val = sigma_conv.f;

  constexpr uint32_t cb_vnode_a       = TileMatrixCB::CB_VNODE_A;
  constexpr uint32_t cb_vnode_b       = TileMatrixCB::CB_VNODE_B;
  constexpr uint32_t cb_scratch_in_a  = TileMatrixCB::CB_SCRATCH_GATHER_0;
  constexpr uint32_t cb_scratch_in_b  = TileMatrixCB::CB_SCRATCH_GATHER_1;
  constexpr uint32_t cb_scratch_out_a = TileMatrixCB::CB_SCRATCH_SCATTER_0;
  constexpr uint32_t cb_scratch_out_b = TileMatrixCB::CB_SCRATCH_SCATTER_1;
  constexpr uint32_t cb_r_msg_a       = TileMatrixCB::CB_R_MSG_A;
  constexpr uint32_t cb_r_msg_b       = TileMatrixCB::CB_R_MSG_B;
  constexpr uint32_t cb_telemetry     = TileMatrixCB::CB_TELEMETRY;

  compute_kernel_hw_startup(cb_scratch_in_a, cb_scratch_out_a);
  copy_init(cb_scratch_in_a);

  uint32_t stats_l1 = ckernel::get_tile_address(cb_telemetry, 0);
  volatile uint32_t *sync_flags =
      reinterpret_cast<volatile uint32_t *>(stats_l1);
  volatile uint32_t *stats = sync_flags;

  volatile uint32_t *const wall_clock_reg =
      reinterpret_cast<volatile uint32_t *>(0xFFB121F0);
  uint32_t t_start = *wall_clock_reg;

  // Initialize PRNG with 4-tier seed hierarchy
  Xoshiro128Plus rng_core;
  rng_core.init(global_seed != 0 ? global_seed : 0x12345678U);
  for (uint32_t c = 0; c < core_id; ++c) {
    rng_core.long_jump();
  }
  for (uint32_t p = 0; p < (start_batch / 2); ++p) {
    rng_core.jump();
    rng_core.jump();
  }

  uint32_t max_steps = max_iter * num_tiles;
  uint32_t total_steps = 2 * max_steps + 2;

  // --------------------------------------------------------------------------
  // Main Simulation Loop (Dual-Batch Pairs)
  // --------------------------------------------------------------------------
  for (uint32_t pair = 0; pair < num_batch_pairs; ++pair) {
#ifdef TRISC_MATH
    sync_flags[SyncSlot::TRISC1_CMD] = static_cast<uint32_t>(MaestroCmd::CMD_IDLE);
#endif
    // Step 1: Generate Channel Observations for Batch A (CB_VNODE_A)
    uint32_t seed_a = rng_core.next();
    generate_channel_tiles(cb_vnode_a, mu_val, sigma_val, benchmark_mode,
                           invocation_id, seed_a);
    rng_core.jump();

    // Step 2: Generate Channel Observations for Batch B (CB_VNODE_B)
    uint32_t seed_b = rng_core.next();
    generate_channel_tiles(cb_vnode_b, mu_val, sigma_val, benchmark_mode,
                           invocation_id, seed_b);
    rng_core.jump();

    // Step 3: Tri-core batch rendezvous barrier
#ifdef TRISC_PACK
    sync_flags[SyncSlot::BATCH_BARRIER_TRISC] = epoch + pair + 1;
#endif
    while (sync_flags[SyncSlot::BATCH_BARRIER_NCRISC] < epoch + pair + 1) {
    }
    while (sync_flags[SyncSlot::BATCH_BARRIER_BRISC] < epoch + pair + 1) {
    }
    while (sync_flags[SyncSlot::BATCH_BARRIER_TRISC] < epoch + pair + 1) {
    }

    // Wait for channel observations to be 100% ready in L1
    while (sync_flags[SyncSlot::BATCH_READY] < epoch + pair + 1) {
    }

    // Step 4: Interleaved Dual-Batch 4-Worker DAE & TRISC Compute Pipeline
    const char* vnode_bytes_a = reinterpret_cast<const char*>(sync_flags[SyncSlot::VNODE_L1_ADDR_A]);
    const char* vnode_bytes_b = reinterpret_cast<const char*>(sync_flags[SyncSlot::VNODE_L1_ADDR_B]);
    char* vnode_out_a = reinterpret_cast<char*>(sync_flags[SyncSlot::VNODE_L1_ADDR_A]);
    char* vnode_out_b = reinterpret_cast<char*>(sync_flags[SyncSlot::VNODE_L1_ADDR_B]);
    const char* vnodes_bytes[2] = {vnode_bytes_a, vnode_bytes_b};
    char* vnodes_out[2] = {vnode_out_a, vnode_out_b};
    const uint32_t* p_hoff_base = reinterpret_cast<const uint32_t*>(sync_flags[SyncSlot::HOFF_L1_ADDR]);

#ifdef TRISC_MATH
    uint8_t active_mask_a = 0x0F;
    uint8_t active_mask_b = 0x0F;
    sync_flags[SyncSlot::ACTIVE_MASK_A]     = 0x0F;
    sync_flags[SyncSlot::ACTIVE_MASK_B]     = 0x0F;
    sync_flags[SyncSlot::EARLY_TERMINATION] = 0;
    sync_flags[SyncSlot::DRAIN_ACK_T0]      = pair;
    sync_flags[SyncSlot::DRAIN_ACK_W1]      = pair;
    sync_flags[SyncSlot::TRISC1_GRANT_STEP] = pair * total_steps + 1; // Grant Step 0
    sync_flags[SyncSlot::TRISC1_CMD]        = static_cast<uint32_t>(MaestroCmd::CMD_STEP_EXEC);
#endif

#if defined(TRISC_UNPACK) || defined(TRISC_PACK)
    // Satellite Pre-Step Gate: Wait for TRISC1 Maestro to launch Step 0
    while (sync_flags[SyncSlot::TRISC1_CMD] != static_cast<uint32_t>(MaestroCmd::CMD_STEP_EXEC) ||
           sync_flags[SyncSlot::TRISC1_GRANT_STEP] < pair * total_steps + 1) { }
#endif

#ifdef TRISC_UNPACK
    uint32_t last_completed_k = 0;
#endif

    for (uint32_t k = 0; k < total_steps; ++k) {
      uint32_t bar = pair * total_steps + k + 1;
      uint32_t b_dae = k % 2;
      uint32_t s_dae = k / 2;

      // 1. Wait for TRISC1 Macro Step Grant (Satellites: TRISC0 & TRISC2)
#if defined(TRISC_UNPACK) || defined(TRISC_PACK)
      while (sync_flags[SyncSlot::TRISC1_GRANT_STEP] < bar) {
        if (sync_flags[SyncSlot::TRISC1_CMD] >= static_cast<uint32_t>(MaestroCmd::CMD_DRAIN_RESIDUAL)) {
          break;
        }
      }
      if (sync_flags[SyncSlot::TRISC1_CMD] >= static_cast<uint32_t>(MaestroCmd::CMD_DRAIN_RESIDUAL)) {
        break;
      }
#endif

      // 2. Scatter Stage: Scatter previous tile (s_dae - 1) of Batch b_dae
#if defined(TRISC_UNPACK) || defined(TRISC_PACK)
      uint8_t active_mask = static_cast<uint8_t>(
          (b_dae == 0) ? sync_flags[SyncSlot::ACTIVE_MASK_A] : sync_flags[SyncSlot::ACTIVE_MASK_B]
      );
      if (s_dae >= 1 && (s_dae - 1) < max_steps) {
        while (sync_flags[SyncSlot::SCATTER_START] < bar) { }
        uint32_t scratch_out_addr = sync_flags[SyncSlot::SCRATCH_SCATTER_PTR];
        uint32_t s_scat = s_dae - 1;
        uint32_t t_scat = s_scat % num_tiles;
        char* vnode_out_ptr = vnodes_out[b_dae];
#if defined(TRISC_UNPACK)
        if (active_mask != 0) {
          scatter_face_24pairs(scratch_out_addr + 1024, p_hoff_base + t_scat * 96 + 24, vnode_out_ptr);
        }
        sync_flags[SyncSlot::SCATTER_DONE_T0] = bar;
#elif defined(TRISC_PACK)
        if (active_mask != 0) {
          scatter_face_24pairs(scratch_out_addr + 1536, p_hoff_base + t_scat * 96 + 72, vnode_out_ptr);
        }
        sync_flags[SyncSlot::SCATTER_DONE_T2] = bar;
#endif
      }
#endif

      // 3. Gather Stage: Gather current tile s_dae of Batch b_dae
#if defined(TRISC_UNPACK) || defined(TRISC_PACK)
      if (s_dae < max_steps) {
        while (sync_flags[SyncSlot::GATHER_START] < bar) { }
        uint32_t scratch_in_addr = sync_flags[SyncSlot::SCRATCH_GATHER_PTR];
        uint32_t t_gath = s_dae % num_tiles;
        const char* vnode_bytes_ptr = vnodes_bytes[b_dae];
#if defined(TRISC_UNPACK)
        if (active_mask != 0) {
          gather_face_24pairs(scratch_in_addr + 1024, p_hoff_base + t_gath * 96 + 24, vnode_bytes_ptr);
        }
        sync_flags[SyncSlot::GATHER_DONE_T0] = bar;
#elif defined(TRISC_PACK)
        if (active_mask != 0) {
          gather_face_24pairs(scratch_in_addr + 1536, p_hoff_base + t_gath * 96 + 72, vnode_bytes_ptr);
        }
        sync_flags[SyncSlot::GATHER_DONE_T2] = bar;
#endif
      }
#endif

      // 4. Compute Stage: Compute check nodes for tile (k - 1)
      if (k >= 1 && k <= 2 * max_steps) {
        uint32_t b_comp = (k - 1) % 2;
        uint32_t s_comp = (k - 1) / 2;
        uint32_t t_comp = s_comp % num_tiles;
        uint32_t iter_comp = s_comp / num_tiles;

        uint32_t cb_in  = (b_comp == 0) ? cb_scratch_in_a  : cb_scratch_in_b;
        uint32_t cb_out = (b_comp == 0) ? cb_scratch_out_a : cb_scratch_out_b;
        uint32_t cb_r   = (b_comp == 0) ? cb_r_msg_a       : cb_r_msg_b;

        uint16_t deg = sync_flags[64 + t_comp];
        if (deg == 6) {
          compute_tile_deg6(cb_in, cb_out, cb_r, iter_comp);
        } else {
          compute_tile_deg3(cb_in, cb_out, cb_r, iter_comp);
        }
      }

#ifdef TRISC_UNPACK
      last_completed_k = k;
#endif

#ifdef TRISC_MATH
      // 5. TRISC1 Conductor: Poll all satellites for step k
      if (s_dae >= 1 && (s_dae - 1) < max_steps) {
        while (sync_flags[SyncSlot::SCATTER_DONE_W0] < bar) { }
        while (sync_flags[SyncSlot::SCATTER_DONE_W1] < bar) { }
        while (sync_flags[SyncSlot::SCATTER_DONE_T0] < bar) { }
        while (sync_flags[SyncSlot::SCATTER_DONE_T2] < bar) { }
      }
      if (s_dae < max_steps) {
        while (sync_flags[SyncSlot::GATHER_DONE_W0] < bar) { }
        while (sync_flags[SyncSlot::GATHER_DONE_W1] < bar) { }
        while (sync_flags[SyncSlot::GATHER_DONE_T0] < bar) { }
        while (sync_flags[SyncSlot::GATHER_DONE_T2] < bar) { }
      }

      // 6. TRISC1 Iteration Boundary & Parity Evaluation (Tile 39)
      bool early_exit = false;
      if (s_dae >= 1 && (s_dae - 1) < max_steps) {
        uint32_t s_scat = s_dae - 1;
        uint32_t t_scat = s_scat % num_tiles;
        if (t_scat == num_tiles - 1) {
          uint32_t iter_scat = s_scat / num_tiles;
          uint32_t info_errs   = sync_flags[SyncSlot::PARITY_REPORT_W0];
          uint32_t parity_errs = sync_flags[SyncSlot::PARITY_REPORT_W1];
          uint32_t total_cw_errs = info_errs | parity_errs;

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

          if (active_mask_a == 0 && active_mask_b == 0) {
            sync_flags[SyncSlot::EARLY_TERMINATION] = 1;
            sync_flags[SyncSlot::TRISC1_CMD] = static_cast<uint32_t>(MaestroCmd::CMD_DRAIN_RESIDUAL);
            early_exit = true;
          }
        }
      }

      // --------------------------------------------------------------------
      // Synchronized Dual-Drain Barrier for Early Termination
      // --------------------------------------------------------------------
      // Maestro waits for both consumer threads to acknowledge draining:
      // 1. Worker 2 (TRISC0) drains cb_scratch_in and check message FIFOs (DRAIN_ACK_T0).
      // 2. Worker 1 (NCRISC) drains speculative cb_scratch_out (DRAIN_ACK_W1).
      // This strictly prevents cross-batch circular buffer pointer drift in TT-Metal.
      if (early_exit) {
        while (sync_flags[SyncSlot::DRAIN_ACK_T0] < pair + 1) { }
        while (sync_flags[SyncSlot::DRAIN_ACK_W1] < pair + 1) { }
        sync_flags[SyncSlot::TRISC1_CMD] = static_cast<uint32_t>(MaestroCmd::CMD_BATCH_DONE);
        break;
      }

      if (k + 1 < total_steps) {
        sync_flags[SyncSlot::TRISC1_GRANT_STEP] = pair * total_steps + (k + 1) + 1;
      } else {
        sync_flags[SyncSlot::TRISC1_CMD] = static_cast<uint32_t>(MaestroCmd::CMD_BATCH_DONE);
        while (sync_flags[SyncSlot::DRAIN_ACK_T0] < pair + 1) { }
        while (sync_flags[SyncSlot::DRAIN_ACK_W1] < pair + 1) { }
      }
#endif
    }

#ifdef TRISC_UNPACK
    // ----------------------------------------------------------------------
    // Drain In-Flight Input Pages & Check Messages (Consumer: TRISC_UNPACK)
    // ----------------------------------------------------------------------
    // Note: cb_scratch_out is consumed by NCRISC, so it is drained in writer_ncrisc.cpp.
    // TRISC_UNPACK only drains the queues it consumes: cb_scratch_in and cb_r_msg_a/b.
    if (sync_flags[SyncSlot::TRISC1_CMD] == static_cast<uint32_t>(MaestroCmd::CMD_DRAIN_RESIDUAL)) {
      uint32_t b_dae = last_completed_k % 2;
      uint32_t s_dae = last_completed_k / 2;
      uint32_t res_in  = (b_dae == 0) ? cb_scratch_in_a  : cb_scratch_in_b;

      if (s_dae < max_steps) {
        cb_wait_front(res_in, 1);
        cb_pop_front(res_in, 1);
      }
    }

    // Drain circular FIFO check messages
    for (uint32_t t = 0; t < num_tiles; ++t) {
      cb_wait_front(cb_r_msg_a, 1);
      cb_pop_front(cb_r_msg_a, 1);
      cb_wait_front(cb_r_msg_b, 1);
      cb_pop_front(cb_r_msg_b, 1);
    }
    sync_flags[SyncSlot::DRAIN_ACK_T0] = pair + 1;
#endif
  }

  uint32_t t_end = *wall_clock_reg;
#ifdef TRISC_PACK
  sync_flags[SyncSlot::TRISC_CYCLES] = t_end - t_start;
  sync_flags[SyncSlot::BATCH_BARRIER_TRISC] = epoch + num_batch_pairs + 1;
#endif
}
