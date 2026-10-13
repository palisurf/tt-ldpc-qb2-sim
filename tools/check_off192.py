import numpy as np
import struct

def u2f(u):
    return struct.unpack('>f', struct.pack('>I', int(u) << 16))[0]

with open('/home/ttuser/LDPC_QB2_GA/dump/golden_scratch_in.bin', 'rb') as f:
    g_raw = f.read()

with open('/home/ttuser/LDPC_QB2_GA/dump/tensix_scratch_in.bin', 'rb') as f:
    t_raw = f.read()

print("Offset 192 (Row 6, Col 0..3):")
g_words = struct.unpack('<4H', g_raw[192:192+8])
t_words = struct.unpack('<4H', t_raw[192:192+8])
print("Golden:", [f"{u2f(w):+.3f}" for w in g_words], [hex(w) for w in g_words])
print("Tensix:", [f"{u2f(w):+.3f}" for w in t_words], [hex(w) for w in t_words])
