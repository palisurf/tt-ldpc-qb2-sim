#!/usr/bin/env python3
"""
Tenstorrent QuietBox 2: LLR Evolution Tracer
Compares iteration-by-iteration LLR evolution between CPU Golden Stepper and Tensix Silicon.
"""

import os
import sys
import struct
import subprocess
import numpy as np

def bf16_to_fp32(val_u16):
    u32 = int(val_u16) << 16
    return struct.unpack('>f', struct.pack('>I', u32))[0]

def extract_codeword_llrs(raw_bytes, cw_idx, num_vnodes=2560):
    words = np.frombuffer(raw_bytes[:20480], dtype=np.uint16)
    llrs = np.zeros(num_vnodes, dtype=np.float32)
    
    for v in range(num_vnodes):
        t = v // 256
        v_loc = v % 256
        row = v_loc // 8
        k = v_loc % 8
        face = (0 if k < 4 else 1) if row < 16 else (2 if k < 4 else 3)
        col_grp = k if k < 4 else (k - 4)
        
        byte_off = t * 2048 + face * 512 + ((row & 15) << 5) + (col_grp << 3) + cw_idx * 2
        word_idx = byte_off // 2
        llrs[v] = bf16_to_fp32(words[word_idx])
    return llrs

def main():
    start_batch = 13
    focus_cw = 0
    invocation_id = 1
    max_iter_limit = 25

    if len(sys.argv) == 1 and os.path.exists("dump/first_failing_frame_info.txt"):
        with open("dump/first_failing_frame_info.txt") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    if k == "batch": start_batch = int(v)
                    elif k == "cw": focus_cw = int(v)
                    elif k == "invocation_id": invocation_id = int(v)
        print(f"[AUTO-LOAD] Loaded from dump/first_failing_frame_info.txt: batch={start_batch}, cw={focus_cw}, invocation_id={invocation_id}")
    else:
        if len(sys.argv) > 1:
            start_batch = int(sys.argv[1])
        if len(sys.argv) > 2:
            focus_cw = int(sys.argv[2])
        if len(sys.argv) > 3:
            invocation_id = int(sys.argv[3])
        if len(sys.argv) > 4:
            max_iter_limit = int(sys.argv[4])

    max_iters = list(range(1, max_iter_limit + 1))

    print("=" * 115)
    print(f" LLR EVOLUTION TRACER: Batch {start_batch}, Invocation {invocation_id} (Focus: Codeword {focus_cw})")
    print("=" * 115)
    print(f"{'Iter':<5} | {'Gold BE':<8} | {'Tensix BE':<10} | {'Sign Mismatch':<14} | {'Gold Mean|L|':<13} | {'Ten Mean|L|':<12} | {'Gold Max|L|':<12} | {'Ten Max|L|':<11} | {'Max |Diff|':<10} | CW Bit Errors (G vs T)")
    print("-" * 115)

    os.makedirs("dump", exist_ok=True)

    for it in max_iters:
        # Run Tensix for 'it' iterations
        cmd_ten = (
            "export TT_METAL_HOME=/home/ttuser/tt-metal && "
            "export TT_METAL_RUNTIME_ROOT=/home/ttuser/tt-metal && "
            "export ARCH_NAME=blackhole && "
            "cd /home/ttuser/LDPC_QB2_GA && "
            f"./engines/tile_matrix_4cw/ldpc_sim_tile_matrix --diff-test-real 2.0 {it} {start_batch} {invocation_id} >/dev/null 2>&1"
        )
        subprocess.run(cmd_ten, shell=True, check=True)
        with open("dump/tensix_final_vnodes.bin", "rb") as f:
            ten_bytes = f.read()

        # Run Golden Stepper for 'it' iterations
        cmd_gold = f"cd /home/ttuser/LDPC_QB2_GA && ./test_40tile_stepper {it} dump/tensix_initial_vnodes.bin dump/tensix_final_vnodes.bin >/dev/null 2>&1"
        subprocess.run(cmd_gold, shell=True, check=True)
        with open("dump/golden_lockstep_final_vnodes.bin", "rb") as f:
            gold_bytes = f.read()

        # Extract focus codeword
        g_cw = extract_codeword_llrs(gold_bytes, focus_cw)
        t_cw = extract_codeword_llrs(ten_bytes, focus_cw)

        # Codeword bit errors (v < 1024)
        g_bes = [np.sum(extract_codeword_llrs(gold_bytes, c)[:1024] < 0.0) for c in range(4)]
        t_bes = [np.sum(extract_codeword_llrs(ten_bytes, c)[:1024] < 0.0) for c in range(4)]

        # Transmitted variables sign mismatch (v < 2048) for focus codeword
        sign_mismatch = np.sum((g_cw[:2048] < 0.0) != (t_cw[:2048] < 0.0))

        # Magnitudes across transmitted nodes (v < 2048)
        g_mean_mag = np.mean(np.abs(g_cw[:2048]))
        t_mean_mag = np.mean(np.abs(t_cw[:2048]))

        g_max_mag = np.max(np.abs(g_cw[:2048]))
        t_max_mag = np.max(np.abs(t_cw[:2048]))

        max_diff = np.max(np.abs(g_cw[:2048] - t_cw[:2048]))

        print(f"{it:<5} | {g_bes[focus_cw]:<8} | {t_bes[focus_cw]:<10} | {sign_mismatch:<14} | {g_mean_mag:<13.3f} | {t_mean_mag:<12.3f} | {g_max_mag:<12.2f} | {t_max_mag:<11.2f} | {max_diff:<10.3f} | G:{g_bes} T:{t_bes}")

        # At Iteration 5, 10, 15, inspect top divergent nodes
        if it in [5, 10, 15]:
            diffs = np.abs(g_cw[:2048] - t_cw[:2048])
            top_divergent = np.argsort(diffs)[::-1][:5]
            print(f"   [Iter {it} Top 5 Divergent Nodes for CW{focus_cw}]:")
            for node in top_divergent:
                print(f"      Node v={node:4d}: Gold={g_cw[node]:+8.3f}, Tensix={t_cw[node]:+8.3f}, Diff={diffs[node]:8.3f}, SignMatch={((g_cw[node]<0)==(t_cw[node]<0))}")


    print("=" * 100)

if __name__ == "__main__":
    main()
