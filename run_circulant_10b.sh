#!/usr/bin/env bash
set -e

# Activate python virtual environment
if [ -f "/home/ttuser/.tenstorrent-venv/bin/activate" ]; then
    source "/home/ttuser/.tenstorrent-venv/bin/activate"
fi

export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-/home/ttuser/tt-metal}"
export TT_METAL_LOGGER_LEVEL="${TT_METAL_LOGGER_LEVEL:-WARN}"

cd /home/ttuser/LDPC_QB2_GA
mkdir -p Results

echo "=========================================================="
echo "Starting 10 Billion Block Circulant DMA Mesh Sweep"
echo "Eb/N0: 2.50 dB | Target FE: 25 | Max Iter: 200 | Cores: 440"
echo "=========================================================="

./engines/circulant_dma_4cw/ldpc_sim_circulant_dma 2.5 10000000000 200 --mesh --min-fe 25 --chunk-batches 2500 2>&1 | tee Results/run_circulant_dma_ebn0_2.50_10B.log
