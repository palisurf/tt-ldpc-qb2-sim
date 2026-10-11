import struct
import numpy as np

def to_f(u):
    return struct.unpack('>f', struct.pack('>HH', u, 0))[0]

init_vn = open("dump/tensix_initial_vnodes.bin", "rb").read()
print(f"Length of initial vnodes: {len(init_vn)} bytes")

# Check byte 13400 (v=1683)
u16_vals = np.frombuffer(init_vn[13400:13400+8], dtype=np.uint16)
print(f"v=1683 (byte 13400): {u16_vals} -> {[to_f(x) for x in u16_vals]}")

# Also check how many non-zero bytes are in init_vn
arr = np.frombuffer(init_vn, dtype=np.uint16)
print(f"Non-zero elements in initial vnodes: {np.count_nonzero(arr)} / {len(arr)}")

# Check punctured nodes: variable 2048 starts at what byte?
# Let's check where v >= 2048 is:
# In AR4JA, v in [2048, 2559] are punctured nodes.
# They are tiles 8 and 9 (since 256 per tile, 8*256=2048).
# Byte offset for tile 8 is 8 * 2048 = 16384.
print(f"Non-zero in transmitted (bytes 0..16384): {np.count_nonzero(arr[:8192])} / 8192")
print(f"Non-zero in punctured (bytes 16384..20480): {np.count_nonzero(arr[8192:])} / 2048")
