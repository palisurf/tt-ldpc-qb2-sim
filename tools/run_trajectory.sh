#!/bin/bash
source /home/ttuser/.tenstorrent-venv/bin/activate
export TT_METAL_HOME=/home/ttuser/tt-metal
export TT_METAL_RUNTIME_ROOT=/home/ttuser/tt-metal
export ARCH_NAME=blackhole
cd /home/ttuser/LDPC_QB2_GA
for it in 1 2 3 4 5 6 7 8 9 10 12 14 16; do
    printf "Iter %2d: " "$it"
    ./engines/tile_matrix_4cw/ldpc_sim_tile_matrix --diff-test-real 2.0 "$it" 13 2>&1 | grep 'Final Batch Errs'
done
