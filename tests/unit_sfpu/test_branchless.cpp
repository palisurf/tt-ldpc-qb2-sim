namespace ckernel { extern volatile unsigned int *instrn_buffer; }
#include "sfpi.h"

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

void test_call() {
  sfpi::vFloat a = sfpi::dst_reg[0];
  sfpi::vFloat b = sfpi::dst_reg[1];
  sfpi::dst_reg[0] = sfpi_amin_star_branchless(a, b);
}
