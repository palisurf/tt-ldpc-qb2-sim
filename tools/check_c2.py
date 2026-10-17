import struct, numpy as np

def bf(u): return struct.unpack('>f', struct.pack('>I', int(u) << 16))[0]
gin = np.fromfile('Z:/LDPC_QB2_GA/dump/golden_scratch_in.bin', dtype=np.uint16)
f_gin = gin[:256].reshape((16, 16))

print("=== Check 2 Inputs (Cols 8..11 of Face 0) ===")
# Even checks are rows 0, 2, 4, 6, 8, 10
# For Check 2: col = 8
for edge in range(6):
    r = 2 * edge
    val = bf(f_gin[r, 8])
    print(f"Edge {edge} (Row {r:2d}): {val:+8.4f}")
