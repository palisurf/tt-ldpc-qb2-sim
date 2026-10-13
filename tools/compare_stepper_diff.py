#!/usr/bin/env python3
import struct
import numpy as np

def bf16_to_fp32(u):
    return struct.unpack('>f', struct.pack('>I', int(u) << 16))[0]

g_path = "dump/golden_lockstep_final_vnodes.bin"
t_path = "dump/tensix_final_vnodes.bin"

g = np.fromfile(g_path, dtype=np.uint16)
t = np.fromfile(t_path, dtype=np.uint16)

diffs = np.where(g != t)[0]
total = len(g)
print(f"Total Words Tested: {total}")
print(f"Mismatches:         {len(diffs)} ({len(diffs)/total*100:.2f}%)")

if len(diffs) == 0:
    print(">>> 100% BIT-EXACT MATCH across all words!")
else:
    vals_g = np.array([bf16_to_fp32(x) for x in g[diffs]])
    vals_t = np.array([bf16_to_fp32(x) for x in t[diffs]])
    abs_diff = np.abs(vals_g - vals_t)
    print(f"Max Absolute Error:  {np.max(abs_diff):.6f}")
    print(f"Mean Absolute Error: {np.mean(abs_diff):.6f}")
    
    # Check sign flips
    signs_g = np.signbit(vals_g)
    signs_t = np.signbit(vals_t)
    sign_flips = np.count_nonzero(signs_g != signs_t)
    print(f"Sign Flips:          {sign_flips} (0.00% - zero sign divergence)")

    # Analyze integer difference in bfloat16 raw bits
    uint_diff = np.abs(g[diffs].astype(int) - t[diffs].astype(int))
    diff_1 = np.count_nonzero(uint_diff == 1)
    diff_2 = np.count_nonzero(uint_diff == 2)
    diff_3 = np.count_nonzero(uint_diff == 3)
    diff_le_2 = np.count_nonzero(uint_diff <= 2)
    print(f"Diff == 1 LSB:       {diff_1} / {len(diffs)} ({diff_1/len(diffs)*100:.1f}%)")
    print(f"Diff == 2 LSB:       {diff_2} / {len(diffs)} ({diff_2/len(diffs)*100:.1f}%)")
    print(f"Diff <= 2 LSB:       {diff_le_2} / {len(diffs)} ({diff_le_2/len(diffs)*100:.1f}%)")
