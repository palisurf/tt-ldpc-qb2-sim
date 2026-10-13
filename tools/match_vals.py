import numpy as np
import struct

def u2f(u):
    return struct.unpack('>f', struct.pack('>I', int(u) << 16))[0]

g_in = np.fromfile('/home/ttuser/LDPC_QB2_GA/dump/golden_scratch_in.bin', dtype=np.uint16)
face0 = g_in[:256].reshape((16, 16))

print("=== Face 0 Inputs (first 4 cols of all rows) ===")
for r in range(16):
    vals = [f"{u2f(face0[r, c]):+.3f}" for c in range(4)]
    print(f"Row {r:2d}: {vals}")
