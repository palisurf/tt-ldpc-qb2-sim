import struct
import numpy as np

def u2f(u):
    return struct.unpack('>f', struct.pack('>I', int(u) << 16))[0]

with open('/home/ttuser/LDPC_QB2_GA/matrices/AR4JA_r45_4c_128c_r12.chinn.out') as f:
    lines = [l.strip().split() for l in f if l.strip()]
dim = lines[-1]
M, N = int(dim[0]), int(dim[1])
c_to_v = [[] for _ in range(M)]
for l in lines[:-1]:
    if len(l) == 3 and float(l[2]) > 0.5:
        r, c = int(l[0]) - 1, int(l[1]) - 1
        c_to_v[r].append(c)

deg6 = [m for m in range(M) if len(c_to_v[m]) == 6]

def partition_checks_balanced(checks, target_tile_count, max_tile_size):
    out_tiles = [[] for _ in range(target_tile_count)]
    tile_vars = [set() for _ in range(target_tile_count)]
    for c in checks:
        c_vars = c_to_v[c]
        best_tile = -1
        min_size = max_tile_size + 1
        for t in range(target_tile_count):
            if len(out_tiles[t]) < max_tile_size:
                conflict = any(v in tile_vars[t] for v in c_vars)
                if not conflict and len(out_tiles[t]) < min_size:
                    min_size = len(out_tiles[t])
                    best_tile = t
        assert best_tile != -1
        out_tiles[best_tile].append(c)
        for v in c_vars:
            tile_vars[best_tile].add(v)
    return out_tiles

deg6_tiles = partition_checks_balanced(deg6, 32, 32)
tile0 = deg6_tiles[0]

def get_tile_vnode_byte_offset(v):
    t = v // 256
    v_loc = v % 256
    row = v_loc // 8
    k = v_loc % 8
    face = (0 if k < 4 else 1) if row < 16 else (2 if k < 4 else 3)
    col_grp = k if k < 4 else (k - 4)
    return t * 2048 + (face * 512) + ((row & 15) << 5) + (col_grp << 3)

with open('/home/ttuser/LDPC_QB2_GA/dump/tensix_initial_vnodes.bin', 'rb') as f:
    vnode_l1 = f.read()

for p in range(24):
    d = p // 4
    kh = p % 4
    c0_even = 0 + kh
    c0_odd  = 4 + kh
    v0_even = c_to_v[tile0[c0_even]][d]
    v0_odd  = c_to_v[tile0[c0_odd]][d]
    off0_even = get_tile_vnode_byte_offset(v0_even)
    off0_odd  = get_tile_vnode_byte_offset(v0_odd)
    
    s0_even = (2 * d) * 32 + kh * 8
    s0_odd  = (2 * d + 1) * 32 + kh * 8
    
    val_even = struct.unpack('<4H', vnode_l1[off0_even : off0_even + 8])
    val_odd  = struct.unpack('<4H', vnode_l1[off0_odd  : off0_odd  + 8])
    
    nz_e = sum(1 for x in val_even if x != 0)
    nz_o = sum(1 for x in val_odd if x != 0)
    print(f"p={p:2d}: d={d}, kh={kh} | Row {s0_even//32:2d} c_even={tile0[c0_even]:3d} v_even={v0_even:4d} nz={nz_e} (off 0x{off0_even:04x}) | Row {s0_odd//32:2d} c_odd={tile0[c0_odd]:3d} v_odd={v0_odd:4d} nz={nz_o} (off 0x{off0_odd:04x})")
