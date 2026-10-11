import struct
import numpy as np

def to_f(u):
    return struct.unpack('>f', struct.pack('>HH', u, 0))[0]

sin = np.frombuffer(open("dump/tensix_scratch_in.bin", "rb").read(), dtype=np.uint16)
sout = np.frombuffer(open("dump/tensix_scratch_out.bin", "rb").read(), dtype=np.uint16)

print("Face 0 row 0 elements (in vs out):")
for c in range(16):
    print(f"c={c:2d}: in={to_f(sin[c]):+.4f} (0x{sin[c]:04X}) out={to_f(sout[c]):+.4f} (0x{sout[c]:04X})")
