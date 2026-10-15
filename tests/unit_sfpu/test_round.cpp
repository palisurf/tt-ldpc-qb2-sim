namespace ckernel { extern volatile unsigned int *instrn_buffer; }
#include "sfpi.h"

void test_load_dest_low_bits(int r) {
  sfpi::vFloat f = sfpi::dst_reg[r];
  sfpi::vUInt u = sfpi::as<sfpi::vUInt>(f);
  sfpi::dst_reg[0] = sfpi::as<sfpi::vFloat>(u & 0xFFFFU);
}
