import numpy as np

gin = np.frombuffer(open("dump/golden_scratch_in.bin", "rb").read(), dtype=np.uint16)
f0 = gin[0:256].reshape(16, 16)
for r in range(16):
    print(f"Row {r:2d}: {list(f0[r][:4])}")
