#!/usr/bin/env python3
import subprocess, re, struct, sys, os
import numpy as np

def u16_to_f(u):
    return struct.unpack('>f', struct.pack('>I', int(u) << 16))[0]

batch_idx = 777594
ebn0 = 2.0

iters_to_test = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 22, 25, 30, 40, 50, 75, 100, 150, 200]

print("==========================================================================================")
print(f" PROBING TRAJECTORY FOR BLOCK 3,110,379 (Batch {batch_idx}, CW3, Eb/N0 {ebn0} dB)")
print("==========================================================================================")
print(f"{'Iter':<5} | {'Stepper Info BE [CW0, 1, 2, 3]':<32} | {'Tensix Silicon Info BE':<30} | {'Mismatches (CW3)':<16} | {'MaxDiff CW3':<12}")
print("-" * 105)

for it in iters_to_test:
    # 1. Run Stepper
    cmd_step = f"./test_40tile_stepper_sfpi --tensix-noise 1 {batch_idx} {ebn0} {it}"
    res_step = subprocess.run(cmd_step, shell=True, capture_output=True, text=True)
    
    # Parse Stepper BE
    m_step = re.search(r"\[Iter\s+(\d+)\]\s+Info BE:\s+\[(\d+),\s+(\d+),\s+(\d+),\s+(\d+)\]", res_step.stdout)
    step_be_str = "N/A"
    step_be_cw3 = 0
    if m_step:
        step_be = [int(m_step.group(2)), int(m_step.group(3)), int(m_step.group(4)), int(m_step.group(5))]
        step_be_str = f"[{step_be[0]:3d}, {step_be[1]:3d}, {step_be[2]:3d}, {step_be[3]:3d}]"
        step_be_cw3 = step_be[3]
    
    # 2. Run Tensix Hardware
    cmd_ten = f"source /home/ttuser/.tenstorrent-venv/bin/activate && export TT_METAL_HOME=/home/ttuser/tt-metal && export TT_METAL_RUNTIME_ROOT=/home/ttuser/tt-metal && export TT_METAL_LOGGER_LEVEL=WARN && ./engines/tile_matrix_4cw/ldpc_sim_tile_matrix --diff-test-real {ebn0} {it} {batch_idx} 1"
    res_ten = subprocess.run(cmd_ten, shell=True, capture_output=True, text=True, executable='/bin/bash')
    
    # Parse Tensix BE
    m_ten = re.search(r"Final Batch Errs\s+:\s+CW0=(\d+),\s+CW1=(\d+),\s+CW2=(\d+),\s+CW3=(\d+)", res_ten.stdout)
    ten_be_str = "N/A"
    ten_be_cw3 = 0
    if m_ten:
        ten_be = [int(m_ten.group(1)), int(m_ten.group(2)), int(m_ten.group(3)), int(m_ten.group(4))]
        ten_be_str = f"[{ten_be[0]:3d}, {ten_be[1]:3d}, {ten_be[2]:3d}, {ten_be[3]:3d}]"
        ten_be_cw3 = ten_be[3]

    # Compare dumps
    mismatches = 0
    max_d = 0.0
    if os.path.exists('dump/golden_lockstep_final_vnodes.bin') and os.path.exists('dump/tensix_final_vnodes.bin'):
        with open('dump/golden_lockstep_final_vnodes.bin', 'rb') as f: gb = f.read()
        with open('dump/tensix_final_vnodes.bin', 'rb') as f: tb = f.read()
        wg = np.frombuffer(gb, dtype=np.uint16)
        wt = np.frombuffer(tb, dtype=np.uint16)
        for idx in range(len(wg)):
            cw = (idx * 2 % 8) // 2
            if cw == 3 and wg[idx] != wt[idx]:
                mismatches += 1
                d = abs(u16_to_f(wg[idx]) - u16_to_f(wt[idx]))
                if d > max_d: max_d = d

    print(f"{it:<5d} | {step_be_str:<32} | {ten_be_str:<30} | {mismatches:<4d} / 2560      | {max_d:.6f}")

    # If both converged or diverged, check if we should break early
    if step_be_cw3 == 0 and ten_be_cw3 == 0:
        print(f"\n[CONVERGED] Both Stepper and Tensix converged at or before iteration {it}!")
        break
