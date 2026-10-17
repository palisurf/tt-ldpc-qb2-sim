#!/usr/bin/env bash
source ~/.tenstorrent-venv/bin/activate
export TT_METAL_RUNTIME_ROOT=/home/ttuser/tt-metal
cd /home/ttuser/LDPC_QB2_GA/engines/tile_matrix_4cw
mkdir -p ../../Results
./ldpc_sim_tile_matrix 2.0 2500000 200 --min-fe 25 --chunk-size 10000 --csv ../../Results/results_tile_matrix_4cw_ar4ja_r12_ebn0_2.00_10M_1core_3piece.csv 2>&1 | tee ../../Results/results_tile_matrix_4cw_ar4ja_r12_ebn0_2.00_10M_1core_3piece.log
