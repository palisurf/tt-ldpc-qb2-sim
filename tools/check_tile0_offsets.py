import numpy as np

def parse_pcm(path):
    lines = open(path).readlines()
    edges = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            r, c, val = int(parts[0]), int(parts[1]), float(parts[2])
            if abs(val) > 1e-6:
                edges.append((r-1, c-1))
    dim_line = lines[-1].strip().split()
    M, N = int(dim_line[0]), int(dim_line[1])
    return M, N, edges

M, N, edges = parse_pcm("matrices/AR4JA_r45_4c_128c_r12.chinn.out")
check_to_var = {}
for r, c in edges:
    if r not in check_to_var: check_to_var[r] = []
    check_to_var[r].append(c)

def get_tile_vnode_byte_offset(v):
    t = v // 256
    v_loc = v % 256
    row = v_loc // 8
    k = v_loc % 8
    face = (0 if k < 4 else 1) if row < 16 else (2 if k < 4 else 3)
    col_grp = k if k < 4 else (k - 4)
    return t * 2048 + face * 512 + (row & 15) * 32 + col_grp * 8

offset_to_v = {get_tile_vnode_byte_offset(v): v for v in range(N)}

# Check checks in Tile 0
deg6_checks = [m for m in range(M) if len(check_to_var[m]) == 6]
def partition_checks_balanced(checks, num_tiles, checks_per_tile):
    out_tiles = [[] for _ in range(num_tiles)]
    tile_vars = [set() for _ in range(num_tiles)]
    for c in checks:
        c_vars = check_to_var[c]
        best_tile = -1
        min_size = 999999
        for t in range(num_tiles):
            if len(out_tiles[t]) >= checks_per_tile:
                continue
            conflict = False
            for v in c_vars:
                if v in tile_vars[t]:
                    conflict = True
                    break
            if not conflict and len(out_tiles[t]) < min_size:
                min_size = len(out_tiles[t])
                best_tile = t
        assert best_tile != -1
        out_tiles[best_tile].append(c)
        for v in c_vars:
            tile_vars[best_tile].add(v)
    return out_tiles

deg6_tiles = partition_checks_balanced(deg6_checks, 32, 32)
tile0 = deg6_tiles[0]

# Now let's generate the 48 pairs for w0 and 48 pairs for w1 as in build_circulant_aware_tile_schedule
print("Building pairs for Tile 0...")
w0_pairs = []
w1_pairs = []
for p in range(48):
    if p < 24:
        d = p // 4
        kh = p % 4
        c0_even = 0 + kh
        c0_odd  = 4 + kh
        c1_even = 8 + kh
        c1_odd  = 12 + kh
    else:
        d = (p - 24) // 4
        kh = (p - 24) % 4
        c0_even = 16 + kh
        c0_odd  = 20 + kh
        c1_even = 24 + kh
        c1_odd  = 28 + kh
    
    v0_e = check_to_var[tile0[c0_even]][d]
    v0_o = check_to_var[tile0[c0_odd]][d]
    v1_e = check_to_var[tile0[c1_even]][d]
    v1_o = check_to_var[tile0[c1_odd]][d]
    
    off0_e = get_tile_vnode_byte_offset(v0_e)
    off0_o = get_tile_vnode_byte_offset(v0_o)
    off1_e = get_tile_vnode_byte_offset(v1_e)
    off1_o = get_tile_vnode_byte_offset(v1_o)
    
    w0_pairs.append((off0_e, off0_o, v0_e, v0_o))
    w1_pairs.append((off1_e, off1_o, v1_e, v1_o))

# Check if Var 33 is in w0_pairs or w1_pairs!
found_33 = False
for p in range(48):
    if 33 in [w0_pairs[p][2], w0_pairs[p][3], w1_pairs[p][2], w1_pairs[p][3]]:
        found_33 = True
        print(f"Var 33 found in p={p}!")
if not found_33:
    print("Var 33 is NOT in w0_pairs or w1_pairs!")

# Now check: What byte offset is Var 33?
print(f"Var 33 byte offset: {get_tile_vnode_byte_offset(33)}")
