namespace ckernel { extern volatile unsigned int *instrn_buffer; }
#include "sfpi.h"

void test_eval(sfpi::vLut16ss s01, sfpi::vLut16ii i01,
               sfpi::vLut16ss s23, sfpi::vLut16ii i23,
               sfpi::vLut16ss s45, sfpi::vLut16ii i45) {
    sfpi::vFloat a = sfpi::dst_reg[0];
    sfpi::vFloat b = sfpi::dst_reg[1];
    auto [min_v, max_v] = sfpi::min_max(a, b);
    sfpi::vFloat diff = max_v - min_v;
    sfpi::vFloat corr = sfpi::lut<sfpi::LutMode::Fp16x6_HWM3>(
        diff, s01, i01, s23, i23, s45, i45, sfpi::LutSign::Update);
    sfpi::dst_reg[2] = sfpi::max(min_v - corr, 0.0f);
}
