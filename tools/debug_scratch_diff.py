import numpy as np
import struct
import os

def b2f(u):
    u32 = int(u) << 16
    return struct.unpack('>f', struct.pack('>I', u32))[0]

dump_dir = "dump"
if not os.path.exists(f"{dump_dir}/golden_scratch_out.bin"):
    dump_dir = "/home/ttuser/LDPC_QB2_GA/dump"

g = np.fromfile(f"{dump_dir}/golden_scratch_out.bin", dtype=np.uint16)
t = np.fromfile(f"{dump_dir}/tensix_scratch_out.bin", dtype=np.uint16)

diff = np.where(g != t)[0]
print(f"Total differing words: {len(diff)}")

for f in range(4):
    f_diff = [i for i in diff if 256*f <= i < 256*(f+1)]
    print(f"Face {f} diff count: {len(f_diff)}")

gi = np.fromfile(f"{dump_dir}/golden_scratch_in.bin", dtype=np.uint16)
ti = np.fromfile(f"{dump_dir}/tensix_scratch_in.bin", dtype=np.uint16)

print("OutTens - InTens across the 4 faces:")
for f in range(4):
    print(f"\n--- FACE {f} ---")
    for r in range(4):
        diffs = []
        for c in range(8):
            idx = f * 256 + r * 16 + c
            diffs.append(f"{b2f(ti[idx]) - b2f(gi[idx]):+.2f}")
        print(f"Row {r:2d}: {' '.join(diffs)}")
