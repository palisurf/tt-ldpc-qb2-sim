#!/bin/bash
set -e
source /home/ttuser/.tenstorrent-venv/bin/activate
export TT_METAL_HOME=/home/ttuser/tt-metal
export TT_METAL_RUNTIME_ROOT=/home/ttuser/tt-metal
export TT_METAL_LOGGER_LEVEL=WARN
cd /home/ttuser/LDPC_QB2_GA

for it in 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25; do
    echo "============================================="
    echo "=== TESTING ITERATION ${it} ==="
    echo "============================================="
    ./engines/tile_matrix_4cw/ldpc_sim_tile_matrix --diff-test-real 2.0 ${it} 6541 19 > /tmp/sim_${it}.log 2>&1
    ./test_40tile_stepper_sfpi ${it} dump/tensix_initial_vnodes.bin dump/tensix_final_vnodes.bin > /tmp/stepper_${it}.log 2>&1
    python3 tools/trace_divergence_cw2.py dump/golden_lockstep_final_vnodes.bin dump/tensix_final_vnodes.bin
    
    # Also grep the BE from both logs for CW2
    echo "Tensix CW2 status:"
    grep -E "Final Batch Errs|Trapped Failing" /tmp/sim_${it}.log | tail -n 2
    echo "Stepper CW2 status:"
    grep -E "Iter *${it}\]" /tmp/stepper_${it}.log || true
done
