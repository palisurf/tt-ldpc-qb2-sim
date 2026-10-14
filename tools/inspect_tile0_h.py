import struct

with open('matrices/AR4JA_r45_4c_128c_r12.chinn.out') as f:
    lines = [l.strip().split() for l in f if l.strip()]
dim = lines[-1]
M, N = int(dim[0]), int(dim[1])
c_to_v = [[] for _ in range(M)]
for l in lines[:-1]:
    if len(l) == 3 and float(l[2]) > 0.5:
        r, c = int(l[0]) - 1, int(l[1]) - 1
        c_to_v[r].append(c)

deg6 = [m for m in range(M) if len(c_to_v[m]) == 6]

def partition_checks_balanced(check_list, target_tile_count, max_tile_size):
    var_to_c = [[] for _ in range(N)]
    for c in check_list:
        for v in c_to_v[c]:
            var_to_c[v].append(c)
    check_conflicts = []
    for c in check_list:
        conflicts = set()
        for v in c_to_v[c]:
            for other in var_to_c[v]:
                if other != c:
                    conflicts.add(other)
        check_conflicts.append((c, len(conflicts)))
    check_conflicts.sort(key=lambda x: (-x[1], x[0]))

    out_tiles = [[] for _ in range(target_tile_count)]
    tile_vars = [set() for _ in range(target_tile_count)]
    for c, _ in check_conflicts:
        c_vars = set(c_to_v[c])
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

def make_pair_word(ve, vo):
    oe = get_tile_vnode_byte_offset(ve) & 0xFFFF
    oo = get_tile_vnode_byte_offset(vo) & 0xFFFF
    return oe | (oo << 16)

# Simulate build_circulant_aware_tile_schedule for Tile 0
w0_pairs = [0]*48
w1_pairs = [0]*48
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
    v0_even = c_to_v[tile0[c0_even]][d]
    v0_odd  = c_to_v[tile0[c0_odd]][d]
    w0_pairs[p] = make_pair_word(v0_even, v0_odd)

print(f"Tile 0 checks: {tile0[:8]}")
print(f"Check {tile0[0]} vars: {c_to_v[tile0[0]]}")
v_deg0 = c_to_v[tile0[0]][0]
off_v_deg0 = get_tile_vnode_byte_offset(v_deg0)
print(f"p=0: c0_even={tile0[0]}, d=0 => v={v_deg0}, byte_offset={off_v_deg0} (0x{off_v_deg0:04x})")
print(f"w0_pairs[0] & 0xFFFF = {w0_pairs[0] & 0xFFFF} (0x{w0_pairs[0] & 0xFFFF:04x})")
