import struct
import numpy as np

with open("dump/tensix_scratch_out.bin", "rb") as f:
    data = f.read()

words = struct.unpack("<1024H", data[:2048])
def bf(w):
    return np.frombuffer(struct.pack("<HH", 0, w), dtype=np.float32)[0]

print("Tensix Scratch Out Face 0 rows 0..3 (16 elements each):")
for r in range(4):
    row_vals = [f"{bf(words[r*16 + c]):+.2f}" for c in range(16)]
    print(f"Row {r:2d}: {' '.join(row_vals)}")
