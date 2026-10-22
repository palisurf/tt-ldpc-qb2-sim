namespace ckernel { extern volatile unsigned int *instrn_buffer; }
#include "sfpi.h"

sfpi_inline sfpi::vFloat round_bf16_fast(sfpi::vFloat val) {
  sfpi::vUInt u = sfpi::as<sfpi::vUInt>(val);
  sfpi::vUInt lsb = (u >> 16) & 1U;
  sfpi::vUInt bias = lsb + 0x7FFFU;
  u += bias;
  return sfpi::as<sfpi::vFloat>(u);
}

sfpi_inline sfpi::vFloat sfpi_amin_star_branchless(sfpi::vFloat a, sfpi::vFloat b) {
  auto [min_v, max_v] = sfpi::min_max(a, b);
  sfpi::vFloat diff = max_v - min_v;
  sfpi::vFloat line1 = sfpi::vFloat(sfpi::sFloat16b(0.69315f)) +
                       sfpi::vFloat(sfpi::sFloat16b(-0.4000f)) * diff;
  sfpi::vFloat line2 = sfpi::vFloat(sfpi::sFloat16b(0.65715f)) +
                       sfpi::vFloat(sfpi::sFloat16b(-0.3280f)) * diff;
  sfpi::vFloat corr = sfpi::max(sfpi::max(line1, line2), 0.0f);
  return sfpi::max(min_v - corr, 0.0f);
}

void test_deg6_subquartet(int p0, int p1, int p2) {
  // Transposed operands already in 32 + p*
  sfpi::vFloat q0 = sfpi::vFloat(sfpi::dst_reg[p0]);
  sfpi::vFloat q0_sw = sfpi::vFloat(sfpi::dst_reg[32 + p0]);
  sfpi::vFloat q1 = sfpi::vFloat(sfpi::dst_reg[p1]);
  sfpi::vFloat q1_sw = sfpi::vFloat(sfpi::dst_reg[32 + p1]);
  sfpi::vFloat q2 = sfpi::vFloat(sfpi::dst_reg[p2]);
  sfpi::vFloat q2_sw = sfpi::vFloat(sfpi::dst_reg[32 + p2]);

  sfpi::vInt s0 = sfpi::as<sfpi::vInt>(q0) ^ sfpi::as<sfpi::vInt>(q0_sw);
  sfpi::vInt s1 = sfpi::as<sfpi::vInt>(q1) ^ sfpi::as<sfpi::vInt>(q1_sw);
  sfpi::vInt s2 = sfpi::as<sfpi::vInt>(q2) ^ sfpi::as<sfpi::vInt>(q2_sw);
  sfpi::vInt s_tot = (s0 ^ s1) ^ s2;

  sfpi::vFloat m0 = sfpi::abs(q0);
  sfpi::vFloat m0_sw = sfpi::abs(q0_sw);
  sfpi::vFloat m01 = sfpi_amin_star_branchless(m0, m0_sw);

  sfpi::vFloat m1 = sfpi::abs(q1);
  sfpi::vFloat m1_sw = sfpi::abs(q1_sw);
  sfpi::vFloat m23 = sfpi_amin_star_branchless(m1, m1_sw);

  sfpi::vFloat m2 = sfpi::abs(q2);
  sfpi::vFloat m2_sw = sfpi::abs(q2_sw);
  sfpi::vFloat m45 = sfpi_amin_star_branchless(m2, m2_sw);

  // Edge pair 0
  {
    sfpi::vFloat m_oth0 = sfpi_amin_star_branchless(m23, m45);
    sfpi::vFloat r_mag0 = sfpi_amin_star_branchless(m_oth0, m0_sw);
    sfpi::vInt s_edge0 = s_tot ^ sfpi::as<sfpi::vInt>(q0);
    sfpi::vFloat r_out0 = round_bf16_fast(sfpi::copysgn(r_mag0, s_edge0));
    sfpi::dst_reg[32 + p0] = round_bf16_fast(q0 + r_out0);
    sfpi::dst_reg[p0] = r_out0;
  }

  // Edge pair 1
  {
    sfpi::vFloat m_oth1 = sfpi_amin_star_branchless(m01, m45);
    sfpi::vFloat r_mag1 = sfpi_amin_star_branchless(m_oth1, m1_sw);
    sfpi::vInt s_edge1 = s_tot ^ sfpi::as<sfpi::vInt>(q1);
    sfpi::vFloat r_out1 = round_bf16_fast(sfpi::copysgn(r_mag1, s_edge1));
    sfpi::dst_reg[32 + p1] = round_bf16_fast(q1 + r_out1);
    sfpi::dst_reg[p1] = r_out1;
  }

  // Edge pair 2
  {
    sfpi::vFloat m_oth2 = sfpi_amin_star_branchless(m01, m23);
    sfpi::vFloat r_mag2 = sfpi_amin_star_branchless(m_oth2, m2_sw);
    sfpi::vInt s_edge2 = s_tot ^ sfpi::as<sfpi::vInt>(q2);
    sfpi::vFloat r_out2 = round_bf16_fast(sfpi::copysgn(r_mag2, s_edge2));
    sfpi::dst_reg[32 + p2] = round_bf16_fast(q2 + r_out2);
    sfpi::dst_reg[p2] = r_out2;
  }
}
