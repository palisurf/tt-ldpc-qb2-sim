#!/usr/bin/env python3
import numpy as np, struct

def bf16_to_fp32(val_u16):
    return struct.unpack('>f', struct.pack('>I', int(val_u16) << 16))[0]

def extract_cw2(fpath):
    with open(fpath, 'rb') as f:
        words = np.frombuffer(f.read()[:20480], dtype=np.uint16)
    llrs = np.zeros(2560, dtype=np.float32)
    for v in range(2560):
        t, v_loc = v // 256, v % 256
        row, k = v_loc // 8, v_loc % 8
        face = (0 if k < 4 else 1) if row < 16 else (2 if k < 4 else 3)
        col_grp = k if k < 4 else (k - 4)
        byte_off = t * 2048 + face * 512 + ((row & 15) << 5) + (col_grp << 3) + 2 * 2
        llrs[v] = bf16_to_fp32(words[byte_off // 2])
    return llrs

g = extract_cw2("/home/ttuser/LDPC_QB2_GA/dump/golden_lockstep_final_vnodes.bin")
t = extract_cw2("/home/ttuser/LDPC_QB2_GA/dump/tensix_final_vnodes.bin")

t_errs = np.where(t[:1024] < 0)[0]
print(f"Tensix CW2 failing nodes count: {len(t_errs)}")
print("First 25 failing nodes in Tensix CW2:")
for idx in t_errs[:25]:
    print(f"  v={idx:4d}: Ten={t[idx]:+8.3f}, Gold={g[idx]:+8.3f}")
