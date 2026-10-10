with open('matrices/AR4JA_r45_4c_128c_r12.chinn.out') as f:
    lines = f.readlines()
check_to_vars = {}
for line in lines[:-1]:
    parts = line.split()
    if len(parts) >= 3 and float(parts[2]) > 0.5:
        m, v = int(parts[0]) - 1, int(parts[1]) - 1
        if m not in check_to_vars: check_to_vars[m] = []
        check_to_vars[m].append(v)

# Re-create partition_checks_balanced
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

# Interleave
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

print(f"Total schedule tiles: {len(schedule)}")
for t, (tile, deg) in enumerate(schedule):
    if 212 in tile:
        idx_in_tile = tile.index(212)
        print(f"Check 212 is in Tile {t} (deg {deg}), index in tile: {idx_in_tile}")
    if 1515 in tile:
        idx_in_tile = tile.index(1515)
        print(f"Check 1515 is in Tile {t} (deg {deg}), index in tile: {idx_in_tile}")
