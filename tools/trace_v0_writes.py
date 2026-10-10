import struct, numpy as np

with open('matrices/AR4JA_r45_4c_128c_r12.chinn.out') as f:
    lines = f.readlines()
check_to_vars = {}
for line in lines[:-1]:
    parts = line.split()
    if len(parts) >= 3 and float(parts[2]) > 0.5:
        m, v = int(parts[0]) - 1, int(parts[1]) - 1
        if m not in check_to_vars: check_to_vars[m] = []
        check_to_vars[m].append(v)

deg6_checks = [m for m in range(1536) if len(check_to_vars.get(m, [])) == 6]
deg3_checks = [m for m in range(1536) if len(check_to_vars.get(m, [])) == 3]

def partition(check_list, target_tile_count, max_tile_size):
    var_to_c = {v: [] for v in range(2560)}
    for c in check_list:
        for v in check_to_vars[c]:
            var_to_c[v].append(c)
    check_conflicts = []
    for c in check_list:
        conflicts = set()
        for v in check_to_vars[c]:
            for other in var_to_c[v]:
                if other != c: conflicts.add(other)
        check_conflicts.append((c, len(conflicts)))
    check_conflicts.sort(key=lambda x: x[1], reverse=True)
    out_tiles = [[] for _ in range(target_tile_count)]
    tile_vars = [set() for _ in range(target_tile_count)]
    for c, _ in check_conflicts:
        c_vars = set(check_to_vars[c])
        best_tile = -1
        min_size = max_tile_size + 1
        for t in range(target_tile_count):
            if len(out_tiles[t]) < max_tile_size:
                if not (tile_vars[t] & c_vars) and len(out_tiles[t]) < min_size:
                    min_size = len(out_tiles[t])
                    best_tile = t
        assert best_tile != -1
        out_tiles[best_tile].append(c)
        tile_vars[best_tile].update(c_vars)
    return out_tiles

deg6_tiles = partition(deg6_checks, 26, 40)
deg3_tiles = partition(deg3_checks, 7, 80)

schedule = []
i6, i3 = 0, 0
while i6 < len(deg6_tiles) or i3 < len(deg3_tiles):
    quota_i6 = ((i3 + 1) * len(deg6_tiles) // len(deg3_tiles)) if (i3 + 1 < len(deg3_tiles)) else len(deg6_tiles)
    while i6 < quota_i6 and i6 < len(deg6_tiles):
        schedule.append((deg6_tiles[i6], 6))
        i6 += 1
    if i3 < len(deg3_tiles):
        schedule.append((deg3_tiles[i3], 3))
        i3 += 1

def get_tile_vnode_byte_offset(v):
    t = v // 256
    v_loc = v % 256
    row = v_loc // 8
    k = v_loc % 8
    face = (0 if k < 4 else 1) if row < 16 else (2 if k < 4 else 3)
    col_grp = k if k < 4 else (k - 4)
    return t * 2048 + (face * 512) + ((row & 15) << 5) + (col_grp << 3)

writes_to_v0 = []
for t, (tile, deg) in enumerate(schedule):
    for p in range(60):
        r_pair = p // 4
        kh     = p % 4
        r_even = 2 * r_pair
        r_odd  = 2 * r_pair + 1
        b_even = r_even // deg
        d_even = r_even % deg
        b_odd  = r_odd // deg
        d_odd  = r_odd % deg
        
        # W0
        c0_even = 8 * b_even + kh
        c0_odd  = 8 * b_odd + kh
        v0_even = check_to_vars[tile[c0_even]][d_even] if c0_even < len(tile) else 2559
        v0_odd  = check_to_vars[tile[c0_odd]][d_odd] if c0_odd < len(tile) else 2559
        
        off0_even = get_tile_vnode_byte_offset(v0_even)
        off0_odd  = get_tile_vnode_byte_offset(v0_odd)
        
        if off0_even == 0: writes_to_v0.append((t, 'W0', 'even', p, v0_even, c0_even))
        if off0_odd  == 0: writes_to_v0.append((t, 'W0', 'odd',  p, v0_odd,  c0_odd))
        
        # W1
        c1_even = 8 * b_even + 4 + kh
        c1_odd  = 8 * b_odd + 4 + kh
        v1_even = check_to_vars[tile[c1_even]][d_even] if c1_even < len(tile) else 2559
        v1_odd  = check_to_vars[tile[c1_odd]][d_odd] if c1_odd < len(tile) else 2559
        
        off1_even = get_tile_vnode_byte_offset(v1_even)
        off1_odd  = get_tile_vnode_byte_offset(v1_odd)
        
        if off1_even == 0: writes_to_v0.append((t, 'W1', 'even', p, v1_even, c1_even))
        if off1_odd  == 0: writes_to_v0.append((t, 'W1', 'odd',  p, v1_odd,  c1_odd))

print(f"Total scatter writes to offset 0 (Var 0): {len(writes_to_v0)}")
for w in writes_to_v0:
    print(" ", w)
