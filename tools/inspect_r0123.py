import numpy as np
import struct

def bf16_to_fp32(val_u16):
    u32 = int(val_u16) << 16
    return struct.unpack('>f', struct.pack('>I', u32))[0]

g_in = np.fromfile('/home/ttuser/LDPC_QB2_GA/dump/golden_scratch_in.bin', dtype=np.uint16).reshape((32, 32))
t_in = np.fromfile('/home/ttuser/LDPC_QB2_GA/dump/tensix_scratch_in.bin', dtype=np.uint16).reshape((32, 32))
g_out = np.fromfile('/home/ttuser/LDPC_QB2_GA/dump/golden_scratch_out.bin', dtype=np.uint16).reshape((32, 32))
t_out = np.fromfile('/home/ttuser/LDPC_QB2_GA/dump/tensix_scratch_out.bin', dtype=np.uint16).reshape((32, 32))

print("=== Inputs at Row 0, 1, 2, 3 Col 0 ===")
for r in [0, 1, 2, 3]:
    print(f"Row {r} in: golden={bf16_to_fp32(g_in[r,0]):.4f}, tensix={bf16_to_fp32(t_in[r,0]):.4f} (match={g_in[r,0]==t_in[r,0]})")
    print(f"Row {r} out: golden={bf16_to_fp32(g_out[r,0]):.4f}, tensix={bf16_to_fp32(t_out[r,0]):.4f} (match={g_out[r,0]==t_out[r,0]})")
