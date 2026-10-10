#!/bin/bash
echo "=== LLK UNPACK A API ==="
grep -rn "copy" /home/ttuser/tt-metal/build_Release/libexec/tt-metalium/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_unpack_A_api.h -A 10
echo "=== Search for dst_reg in hw ==="
grep -rn "dst_reg" /home/ttuser/tt-metal/build_Release/libexec/tt-metalium/tt_metal/hw/ | head -n 30
echo "=== Search for face in ckernels blackhole ==="
find /home/ttuser/tt-metal/build_Release/libexec/tt-metalium/tt_metal/hw/ckernels/blackhole/ -type f | xargs grep -in "face" | head -n 30
