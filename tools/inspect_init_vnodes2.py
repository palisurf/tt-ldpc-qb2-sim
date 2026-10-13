import numpy as np
import struct

def u2f(u):
    return struct.unpack('>f', struct.pack('>I', int(u) << 16))[0]

def get_tile_vnode_byte_offset(v):
    t = v // 256
    v_loc = v % 256
    row = v_loc // 8
    k = v_loc % 8
    face = (0 if k < 4 else 1) if row < 16 else (2 if k < 4 else 3)
    col_grp = k if k < 4 else (k - 4)
    return t * 2048 + (face * 512) + ((row & 15) << 5) + (col_grp << 3)

with open('/home/ttuser/LDPC_QB2_GA/dump/tensix_initial_vnodes.bin', 'rb') as f:
    raw = f.read()

v_indices = [
    (803, "d=0, c0"),
    (1173, "d=1, c0"),
    (1237, "d=2, c0"),
    (1683, "d=3, c0"),
    (1889, "d=4, c0"),
    (2347, "d=5, c0 (punctured)")
]

for v, name in v_indices:
    off = get_tile_vnode_byte_offset(v)
    words = struct.unpack('<4H', raw[off:off+8])
    vals = [f"{u2f(w):+.3f}" for w in words]
    print(f"Var {v:4d} ({name}) at off 0x{off:04x}: {vals}")
