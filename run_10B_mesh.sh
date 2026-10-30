#!/bin/bash
set -e

source /home/ttuser/.tenstorrent-venv/bin/activate
export TT_METAL_HOME=/home/ttuser/tt-metal
export TT_METAL_RUNTIME_ROOT=/home/ttuser/tt-metal
export TT_METAL_LOGGER_LEVEL=WARN
export ARCH_NAME=blackhole

cd /home/ttuser/LDPC_QB2_GA

# Target AICLK frequency in MHz (default: 1400 MHz max firmware ceiling; set BH_AICLK=0 to use standard 1350 MHz DVFS)
BH_AICLK="${BH_AICLK:-1400}"

if [ "$BH_AICLK" -gt 0 ]; then
    echo "[STARTUP] Locking all 4 Blackhole ASICs to ${BH_AICLK} MHz..."
    python3 tools/set_blackhole_clk.py "$BH_AICLK"
    trap 'echo "[CLEANUP] Releasing forced AICLK back to default DVFS..."; python3 tools/set_blackhole_clk.py 0' EXIT
fi

EBN0="${1:-2.50}"
MAX_BLOCKS="${2:-10000000000}"
MAX_ITER="${3:-200}"
MIN_FE="${4:-1000000}"
SEED_OPT="${5:---random-seed}"
LOG_FILE="sim_10B_ebn0_${EBN0}_$( [ "$BH_AICLK" -gt 0 ] && echo "${BH_AICLK}MHz" || echo "1350MHz" ).log"

stdbuf -oL -eL ./engines/circulant_dma_4cw/ldpc_sim_circulant_dma "$EBN0" "$MAX_BLOCKS" "$MAX_ITER" --min-fe "$MIN_FE" $SEED_OPT 2>&1 | tee "$LOG_FILE"
