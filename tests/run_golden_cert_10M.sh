#!/bin/bash
cd /home/ttuser/LDPC_QB2_GA/tests
./test_40tile_stepper --monte-carlo 2.0 10000000 5000 200 16 0.0 ../Results/results_40tile_lockstep_ebn0_2.00_10M.csv 2>&1 | tee ../Results/results_40tile_lockstep_ebn0_2.00_10M.log
