#!/bin/bash
# ==============================================================================
# Isolation Study: Equalized vs Unequalized with Remez Minimax Polynomials
# 8 threads pinned to Cores 0-7:   --tensix-noise --equalize-noise --minimax
# 8 threads pinned to Cores 8-15:  --tensix-noise --minimax
# ==============================================================================

set -e
cd /home/ttuser/LDPC_QB2_GA

mkdir -p logs Results

echo "================================================================="
echo " Launching Dual 8-Thread Minimax Isolation Simulations"
echo "================================================================="

LOG_EQ="logs/sim_equalized_minimax_8t.log"
LOG_UNEQ="logs/sim_unequalized_minimax_8t.log"

CSV_EQ="Results/results_40tile_sfpi_equalized_minimax_8t.csv"
CSV_UNEQ="Results/results_40tile_sfpi_unequalized_minimax_8t.csv"

# Remove any previous result CSV for this specific test
rm -f "$CSV_EQ" "$CSV_UNEQ"

echo "Launching Job 1 (Equalized + Minimax) on cores 0-7 -> $LOG_EQ"
taskset -c 0-7 ./tests/test_40tile_stepper_sfpi \
    --monte-carlo 2.0 10000000 25 200 8 0.0 "$CSV_EQ" \
    --tensix-noise --equalize-noise --minimax > "$LOG_EQ" 2>&1 &
PID_EQ=$!

echo "Launching Job 2 (Unequalized + Minimax) on cores 8-15 -> $LOG_UNEQ"
taskset -c 8-15 ./tests/test_40tile_stepper_sfpi \
    --monte-carlo 2.0 10000000 25 200 8 0.0 "$CSV_UNEQ" \
    --tensix-noise --minimax > "$LOG_UNEQ" 2>&1 &
PID_UNEQ=$!

echo "Job 1 PID: $PID_EQ"
echo "Job 2 PID: $PID_UNEQ"
echo "Both jobs running concurrently across all 16 host cores."
