import numpy as np
import struct

# Let's read matrix and simulate add_tile_to_schedule
with open('/home/ttuser/LDPC_QB2_GA/matrices/AR4JA_r45_4c_128c_r12.chinn.out') as f:
    lines = [l.strip().split() for l in f if l.strip()]
dim = lines[-1]
M, N = int(dim[0]), int(dim[1])
c_to_v = [[] for _ in range(M)]
for l in lines[:-1]:
    if len(l) == 3 and float(l[2]) > 0.5:
        r, c = int(l[0]) - 1, int(l[1]) - 1
        c_to_v[r].append(c)

deg6 = [m for m in range(M) if len(c_to_v[m]) == 6]
tile0 = deg6[:32] # first 32 checks

print("Tile 0 checks:", tile0[:8])

# Let's inspect p=0..23 for Tile 0:
for p in range(24):
    d = p // 4
    kh = p % 4
    c0_even = 0 + kh
    c0_odd = 4 + kh
    v_even = c_to_v[tile0[c0_even]][d]
    v_odd = c_to_v[tile0[c0_odd]][d]
    row_even = 2 * d
    row_odd = 2 * d + 1
    col = kh * 4
    print(f"p={p:2d}: d={d}, c_even={tile0[c0_even]}(v={v_even}) -> Row {row_even}, c_odd={tile0[c0_odd]}(v={v_odd}) -> Row {row_odd}")
