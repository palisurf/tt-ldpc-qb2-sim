import numpy as np
import struct

def u2f(u):
    return struct.unpack('>f', struct.pack('>I', int(u) << 16))[0]

tout = np.fromfile('dump/tensix_scratch_out.bin', dtype=np.uint16)
f0 = tout[:256].reshape((16, 16))

print("=== Face 0 Probe Registers (Rows 12..15) ===")
# Reg 6 (base_r + 6): Row 12 (lanes 0..7), Row 13 (lanes 8..15), Row 14 (lanes 16..23), Row 15 (lanes 24..31)
for r in [12, 13, 14, 15]:
    vals = [u2f(x) for x in f0[r, :]]
    print(f"Row {r:2d}: {[round(v, 4) for v in vals]}")
