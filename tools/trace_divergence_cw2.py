#!/usr/bin/env python3
import struct, sys
import numpy as np

def u16_to_f(u):
    return struct.unpack('>f', struct.pack('>I', int(u) << 16))[0]

gold_path = sys.argv[1] if len(sys.argv) > 1 else 'dump/golden_lockstep_final_vnodes.bin'
ten_path  = sys.argv[2] if len(sys.argv) > 2 else 'dump/tensix_final_vnodes.bin'

with open(gold_path, 'rb') as f: gold_b = f.read()
with open(ten_path, 'rb') as f: ten_b = f.read()

w_gold = np.frombuffer(gold_b, dtype=np.uint16)
w_ten  = np.frombuffer(ten_b,  dtype=np.uint16)

mismatches_per_cw = [0, 0, 0, 0]
total_per_cw = [0, 0, 0, 0]
max_diff_per_cw = [0.0, 0.0, 0.0, 0.0]

for idx in range(len(w_gold)):
    byte_off = idx * 2
    cw = (byte_off % 8) // 2
    total_per_cw[cw] += 1
    if w_gold[idx] != w_ten[idx]:
        mismatches_per_cw[cw] += 1
        d = abs(u16_to_f(w_gold[idx]) - u16_to_f(w_ten[idx]))
        if d > max_diff_per_cw[cw]:
            max_diff_per_cw[cw] = d

print(f"File Comparison: {gold_path} vs {ten_path}")
for cw in range(4):
    print(f"  CW{cw}: {mismatches_per_cw[cw]:4d} / {total_per_cw[cw]} mismatches ({mismatches_per_cw[cw]/total_per_cw[cw]*100:6.2f}%) | MaxDiff: {max_diff_per_cw[cw]:.6f}")
