import struct
import numpy as np

# Let's inspect the tensix scratch out that was computed by Tensix hardware!
# We want to know:
# When Tensix computed on Face 2:
# What check node edges did it actually compute?

tensix_out = np.frombuffer(open("dump/tensix_scratch_out.bin", "rb").read(), dtype=np.uint16)
tensix_in  = np.frombuffer(open("dump/tensix_scratch_in.bin", "rb").read(), dtype=np.uint16)

def to_f(u):
    return struct.unpack('>f', struct.pack('>HH', u, 0))[0]

def to_u16(f):
    u = struct.unpack('>I', struct.pack('>f', f))[0]
    return u >> 16

# Look at Face 2 in tensix_out:
f2_out = tensix_out[512:768].reshape(16, 16)
f2_in  = tensix_in[512:768].reshape(16, 16)

print("Face 2 Tensix in vs out for row 0:")
for c in range(16):
    print(f"c={c:2d}: in={to_f(f2_in[0,c]):+.4f} out={to_f(f2_out[0,c]):+.4f}")
