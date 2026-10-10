import subprocess

def main():
    test_src = """
#include "api/compute/common_globals.h"
#include "api/compute/cb_api.h"
#include "sfpi.h"
#include "sfpi_lib.h"

using namespace sfpi;

void test_amin(vFloat a, vFloat b, vFloat& out) {
    vFloat min_v = min(a, b);
    vFloat diff  = abs(a - b);
    
    vFloat corr = 0.0f;
    v_if (diff < 0.5f) {
        corr = 0.69315f - 0.40f * diff;
    } v_elseif (diff < 2.0f) {
        corr = 0.49315f - 0.328f * (diff - 0.5f);
    } v_endif;
    
    vFloat res = min_v - corr;
    out = max(res, 0.0f);
}
"""
    with open("/tmp/test_sfpi_tt.cpp", "w") as f:
        f.write(test_src)

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
    cmd = f"{gxx} -O3 -mcpu=tt-bh-tensix {defs} {incs} -c /tmp/test_sfpi_tt.cpp -o /tmp/test_sfpi_tt.o"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("Compilation return code:", res.returncode)
    if res.returncode != 0:
        print("STDERR:\n", res.stderr[:1000])
    else:
        print("SUCCESS! Compiled cleanly to object file.")
        objdump = "/home/ttuser/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump"
        dis = subprocess.run(f"{objdump} -d /tmp/test_sfpi_tt.o", shell=True, capture_output=True, text=True)
        print("Disassembly:")
        print(dis.stdout[:1500])

if __name__ == "__main__":
    main()
