import numpy as np

# Inputs in check 0
# Let's see what values exist in golden_scratch_in for the 6 edges
g_in = np.fromfile('/home/ttuser/LDPC_QB2_GA/dump/golden_scratch_in.bin', dtype=np.uint16)
# Let's inspect all 16 values of column 0 in Face 0
face0 = g_in[:256].reshape((16, 16))
col0 = [face0[r, 0] for r in range(16)]

import struct
def u2f(u):
    return struct.unpack('>f', struct.pack('>I', int(u) << 16))[0]

print("Face 0 Col 0 values:")
for r, u in enumerate(col0):
    print(f"Row {r:2d}: {u2f(u):+.4f} (0x{u:04x})")
