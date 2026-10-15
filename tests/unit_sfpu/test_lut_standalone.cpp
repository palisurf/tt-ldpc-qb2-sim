namespace ckernel {
    extern volatile unsigned int *instrn_buffer;
}

#include "sfpi.h"

void test_lut() {
    sfpi::vLut16ss s01(-0.4000f, -0.3280f);
    sfpi::vLut16ii i01(0.69315f, 0.65715f);
    sfpi::vLut16ss s23(-0.3280f, -0.3280f);
    sfpi::vLut16ii i23(0.65715f, 0.65715f);
    sfpi::vLut16ss s45(0.0f, 0.0f);
    sfpi::vLut16ii i45(0.0f, 0.0f);

    sfpi::vFloat diff = sfpi::dst_reg[0];
    sfpi::vFloat corr = sfpi::lut<sfpi::LutMode::Fp16x6_HWM3>(
        diff, s01, i01, s23, i23, s45, i45, sfpi::LutSign::Update);
    sfpi::dst_reg[1] = corr;
}
