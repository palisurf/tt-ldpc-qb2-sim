#!/bin/bash
echo "=== Searching TT-Metal source tree (excluding cpmcache) ==="
grep -rn "VectorMode" /home/ttuser/tt-metal/tt_metal/ /home/ttuser/tt-metal/docs/ 2>/dev/null | head -n 30
echo "=== Searching for unpacker tile descriptor and Dst mapping ==="
grep -rn "llk_unpack_A" /home/ttuser/tt-metal/tt_metal/hw/ckernels/blackhole/ 2>/dev/null | head -n 30
echo "=== Searching for SFPI documentation ==="
find /home/ttuser/tt-metal/runtime/sfpi/ -name "*.md" -o -name "*.h" | xargs grep -in "dst_reg" 2>/dev/null | head -n 30
