# Let's inspect pcm.h_offsets_pairs for Tile 0 in ldpc_sim_tile_matrix.cpp
# We can read the first 96 uint32 from a binary dump or check how they are generated.
import struct

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

def get_tile_vnode_byte_offset(v):
    t = v // 256
    v_loc = v % 256
    row = v_loc // 8
    k = v_loc % 8
    face = (0 if k < 4 else 1) if row < 16 else (2 if k < 4 else 3)
    col_grp = k if k < 4 else (k - 4)
    return t * 2048 + (face * 512) + ((row & 15) << 5) + (col_grp << 3)

# Let's load tensix_initial_vnodes.bin (20480 bytes)
with open('/home/ttuser/LDPC_QB2_GA/dump/tensix_initial_vnodes.bin', 'rb') as f:
    vnode_l1 = f.read()

# Let's compute s0_even, s0_odd, off0_even, off0_odd for p=0..23:
for p in range(24):
    d = p // 4
    kh = p % 4
    c0_even = 0 + kh
    c0_odd = 4 + kh
    v0_even = c_to_v[deg6[c0_even]][d]
    v0_odd  = c_to_v[deg6[c0_odd]][d]
    off0_even = get_tile_vnode_byte_offset(v0_even)
    off0_odd  = get_tile_vnode_byte_offset(v0_odd)
    
    s0_even = (2 * d) * 32 + kh * 8
    s0_odd  = (2 * d + 1) * 32 + kh * 8
    
    val_even = struct.unpack('<4H', vnode_l1[off0_even : off0_even + 8])
    val_odd  = struct.unpack('<4H', vnode_l1[off0_odd  : off0_odd  + 8])
    
    print(f"p={p:2d}: d={d}, kh={kh} | s0_even={s0_even:3d} (Row {s0_even//32:2d}) v_even={v0_even:4d} nonzeros={sum(1 for x in val_even if x != 0)} | s0_odd={s0_odd:3d} (Row {s0_odd//32:2d}) v_odd={v0_odd:4d} nonzeros={sum(1 for x in val_odd if x != 0)}")
