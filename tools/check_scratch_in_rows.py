import numpy as np
import struct

def u2f(u):
    return struct.unpack('>f', struct.pack('>I', int(u) << 16))[0]

# Let's inspect tensix_scratch_in (which is what unpacker unpacks from L1 into dst_reg)
t_in = np.fromfile('/home/ttuser/LDPC_QB2_GA/dump/tensix_scratch_in.bin', dtype=np.uint16)
f0 = t_in[:256].reshape((16, 16))

print("=== Face 0 Scratch In Rows 0..5 (First 4 words) ===")
for r in range(6):
    vals = [f"{u2f(f0[r, c]):+.2f}" for c in range(4)]
    print(f"Row {r}: {vals}")
