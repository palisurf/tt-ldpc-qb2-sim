namespace ckernel { extern volatile unsigned int *instrn_buffer; }
#include "sfpi.h"

sfpi_inline sfpi::vFloat sfpi_amin_star_lut(sfpi::vFloat a, sfpi::vFloat b) {
    auto [min_v, max_v] = sfpi::min_max(a, b);
    sfpi::vFloat diff   = max_v - min_v;
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

void test_two_calls() {
    sfpi::vFloat a = sfpi::dst_reg[0];
    sfpi::vFloat b = sfpi::dst_reg[1];
    sfpi::dst_reg[2] = sfpi_amin_star_lut(a, b);

    sfpi::vFloat c = sfpi::dst_reg[2];
    sfpi::vFloat d = sfpi::dst_reg[3];
    sfpi::dst_reg[4] = sfpi_amin_star_lut(c, d);
}
