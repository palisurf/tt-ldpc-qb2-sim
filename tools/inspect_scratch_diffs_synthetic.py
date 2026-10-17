#!/usr/bin/env python3
import struct
import numpy as np

def bf16(u):
    return struct.unpack('>f', struct.pack('>I', int(u) << 16))[0]

g_in = np.fromfile('dump/golden_scratch_in.bin', dtype=np.uint16).reshape((32, 32))
t_in = np.fromfile('dump/tensix_scratch_in.bin', dtype=np.uint16).reshape((32, 32))
g_out = np.fromfile('dump/golden_scratch_out.bin', dtype=np.uint16).reshape((32, 32))
t_out = np.fromfile('dump/tensix_scratch_out.bin', dtype=np.uint16).reshape((32, 32))

print('In diffs:', np.count_nonzero(g_in != t_in))
print('Out diffs:', np.count_nonzero(g_out != t_out))
for r, c in zip(*np.where(g_out != t_out)):
    print(f'Row {r:2d}, Col {c:2d}: G=0x{g_out[r,c]:04X} ({bf16(g_out[r,c]):+.5f}) vs T=0x{t_out[r,c]:04X} ({bf16(t_out[r,c]):+.5f}) | In: G=0x{g_in[r,c]:04X} ({bf16(g_in[r,c]):+.5f})')
