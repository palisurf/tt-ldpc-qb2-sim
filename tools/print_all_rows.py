# Let's inspect the actual values in scratch_in for Face 0
import struct
import numpy as np

def to_f(u):
    return struct.unpack('>f', struct.pack('>HH', u, 0))[0]

sin = np.frombuffer(open("dump/tensix_scratch_in.bin", "rb").read(), dtype=np.uint16)
for face in range(4):
    f_arr = sin[face*256:(face+1)*256].reshape(16, 16)
    print(f"\nFace {face} all 16 rows (first 4 elements of each row):")
    for r in range(16):
        vals = [f"{to_f(f_arr[r, c]):+.2f}" for c in range(4)]
        print(f"  Row {r:2d}: {vals}")
