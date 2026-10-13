import numpy as np
import struct

def u2f(u):
    return struct.unpack('>f', struct.pack('>I', int(u) << 16))[0]

gin = np.fromfile('dump/golden_scratch_in.bin', dtype=np.uint16)[:256].reshape((16, 16))
print("=== Face 0 Input Rows (first 8 cols) ===")
for r in range(12):
    row_vals = [round(u2f(x), 3) for x in gin[r, :8]]
    print(f"Row {r:2d}: {row_vals}")
