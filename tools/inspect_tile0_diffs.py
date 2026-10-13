#!/usr/bin/env python3
import numpy as np, struct

def bf16_to_fp32(val_u16):
    return struct.unpack('>f', struct.pack('>I', int(val_u16) << 16))[0]

def fp32_to_bf16(f):
    u = struct.unpack('>I', struct.pack('>f', f))[0]
    lsb = (u >> 16) & 1
    bias = 0x7FFF + lsb
    return (u + bias) >> 16

# Read golden_scratch_out and tensix_scratch_out
with open("/home/ttuser/LDPC_QB2_GA/dump/golden_scratch_out.bin", "rb") as f:
    g_out = np.frombuffer(f.read(), dtype=np.uint16)
with open("/home/ttuser/LDPC_QB2_GA/dump/tensix_scratch_out.bin", "rb") as f:
    t_out = np.frombuffer(f.read(), dtype=np.uint16)

with open("/home/ttuser/LDPC_QB2_GA/dump/golden_scratch_in.bin", "rb") as f:
    s_in = np.frombuffer(f.read(), dtype=np.uint16)

# Look at differing words
diffs = np.where(g_out != t_out)[0]
print(f"Total differing words in Tile 0 scratch out: {len(diffs)}")

for idx in diffs:
    # Decode row and col
    # byte_off = idx * 2
    # In scratchpad: face = row // 8, ...
    g_val = bf16_to_fp32(g_out[idx])
    t_val = bf16_to_fp32(t_out[idx])
    print(f"Index {idx:4d} (0x{idx*2:04x}): Golden=0x{g_out[idx]:04x} ({g_val:+8.5f}), Tensix=0x{t_out[idx]:04x} ({t_val:+8.5f})")
