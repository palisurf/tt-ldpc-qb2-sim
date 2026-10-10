import subprocess

src = """
#include <cstdint>
namespace ckernel { extern uint32_t* instrn_buffer; }
#include "sfpi.h"

using namespace sfpi;

sfpi_inline sfpi::vFloat sfpi_amin_star_branchless(sfpi::vFloat a, sfpi::vFloat b) {
    auto [min_v, max_v] = sfpi::min_max(a, b);
    sfpi::vFloat diff   = max_v - min_v;
    sfpi::vFloat line1  = sfpi::vFloat(sfpi::sFloat16b(0.69315f)) + sfpi::vFloat(sfpi::sFloat16b(-0.4000f)) * diff;
    sfpi::vFloat line2  = sfpi::vFloat(sfpi::sFloat16b(0.65715f)) + sfpi::vFloat(sfpi::sFloat16b(-0.3280f)) * diff;
    sfpi::vFloat corr   = sfpi::max(sfpi::max(line1, line2), 0.0f);
    return sfpi::max(min_v - corr, 0.0f);
}

void test_kernel(vFloat q0, vFloat q1, vFloat q2, vFloat q3, vFloat q4, vFloat q5, vFloat& out0, vFloat& out5) {
    vFloat min_mag = abs(q0);
    vFloat min_idx = 0.0f;

    vFloat m1 = abs(q1);
    v_if (m1 < min_mag) { min_mag = m1; min_idx = 1.0f; } v_endif;
    vFloat m2 = abs(q2);
    v_if (m2 < min_mag) { min_mag = m2; min_idx = 2.0f; } v_endif;
    vFloat m3 = abs(q3);
    v_if (m3 < min_mag) { min_mag = m3; min_idx = 3.0f; } v_endif;
    vFloat m4 = abs(q4);
    v_if (m4 < min_mag) { min_mag = m4; min_idx = 4.0f; } v_endif;
    vFloat m5 = abs(q5);
    v_if (m5 < min_mag) { min_mag = m5; min_idx = 5.0f; } v_endif;

    vFloat m_others = abs(q0);
    v_if (min_idx == 0.0f) { m_others = 1000.0f; } v_endif;

    v_if (min_idx == 1.0f) { m1 = 1000.0f; } v_endif;
    m_others = sfpi_amin_star_branchless(m_others, m1);

    v_if (min_idx == 2.0f) { m2 = 1000.0f; } v_endif;
    m_others = sfpi_amin_star_branchless(m_others, m2);

    v_if (min_idx == 3.0f) { m3 = 1000.0f; } v_endif;
    m_others = sfpi_amin_star_branchless(m_others, m3);

    v_if (min_idx == 4.0f) { m4 = 1000.0f; } v_endif;
    m_others = sfpi_amin_star_branchless(m_others, m4);

    v_if (min_idx == 5.0f) { m5 = 1000.0f; } v_endif;
    m_others = sfpi_amin_star_branchless(m_others, m5);

    vFloat m_all = sfpi_amin_star_branchless(m_others, min_mag);

    vFloat m_d0 = m_all;
    v_if (min_idx == 0.0f) { m_d0 = m_others; } v_endif;

    vFloat m_d5 = m_all;
    v_if (min_idx == 5.0f) { m_d5 = m_others; } v_endif;

    out0 = m_d0;
    out5 = m_d5;
}
"""

with open("/tmp/test_min_idx.cpp", "w") as f:
    f.write(src)

gxx = "/home/ttuser/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-g++"
incs = (
    "-I/home/ttuser/tt-metal/runtime/sfpi/include "
    "-I/home/ttuser/tt-metal/tt_metal/hw/inc "
    "-I/home/ttuser/tt-metal/tt_metal/hw/inc/internal/tt-1xx "
    "-I/home/ttuser/tt-metal/tt_metal/hw/inc/internal/tt-1xx/blackhole "
    "-I/home/ttuser/tt-metal/tt_metal/hw/ckernels/blackhole/metal/llk_api "
    "-I/home/ttuser/tt-metal/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu "
)
defs = "-DARCH_BLACKHOLE -DTENSIX_FIRMWARE -DKERNEL_BUILD -DTRISC_MATH"
cmd = f"{gxx} -O3 -mcpu=tt-bh-tensix {defs} {incs} -c /tmp/test_min_idx.cpp -o /tmp/test_min_idx.o"
res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print("Return code:", res.returncode)
if res.returncode != 0:
    print(res.stderr[:800])
else:
    print("SUCCESS! Disassembling...")
    objdump = "/home/ttuser/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump"
    dis = subprocess.run(f"{objdump} -d /tmp/test_min_idx.o", shell=True, capture_output=True, text=True)
    print(dis.stdout[:2000])
