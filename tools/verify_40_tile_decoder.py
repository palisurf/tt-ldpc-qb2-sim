import numpy as np

# Load matrix
pcm_M = 1536
pcm_N = 2560
edges = {}
with open('matrices/AR4JA_r45_4c_128c_r12.chinn.out', 'r') as f:
    for line in f:
        p = line.strip().split()
        if len(p) >= 3:
            r, c, v = int(p[0]) - 1, int(p[1]) - 1, float(p[2])
            if v != 0.0:
                edges.setdefault(r, []).append(c)

deg6_checks = [m for m in range(pcm_M) if len(edges[m]) == 6]
deg3_checks = [m for m in range(pcm_M) if len(edges[m]) == 3]

def partition(check_list, target_count, max_size):
    var_to_c = {}
    for c in check_list:
        for v in edges[c]:
            var_to_c.setdefault(v, []).append(c)
    
    conflicts = []
    for c in check_list:
        conf = set()
        for v in edges[c]:
            for other in var_to_c[v]:
                if other != c: conf.add(other)
        conflicts.append((c, len(conf)))
    conflicts.sort(key=lambda x: x[1], reverse=True)
    
    out_tiles = [[] for _ in range(target_count)]
    tile_vars = [set() for _ in range(target_count)]
    
    for c, _ in conflicts:
        c_vars = set(edges[c])
        best = -1
        min_s = max_size + 1
        for t in range(target_count):
            if len(out_tiles[t]) < max_size:
                if not any(v in tile_vars[t] for v in c_vars):
                    if len(out_tiles[t]) < min_s:
                        min_s = len(out_tiles[t])
                        best = t
        if best == -1: return None
        out_tiles[best].append(c)
        for v in c_vars: tile_vars[best].add(v)
    return out_tiles

deg6_tiles = partition(deg6_checks, 32, 32)
deg3_tiles = partition(deg3_checks, 8, 64)

# Interleave 32 deg-6 and 8 deg-3 tiles: 4 deg-6 per 1 deg-3
schedule = []
for i in range(8):
    for j in range(4):
        schedule.append((deg6_tiles[i * 4 + j], 6))
    schedule.append((deg3_tiles[i], 3))

print(f"Total tiles in schedule: {len(schedule)} (32 deg6 + 8 deg3)")

# Verify that every single check m in 0..1535 is executed exactly once
executed_checks = []
for t, deg in schedule:
    executed_checks.extend(t)

assert len(executed_checks) == 1536
assert len(set(executed_checks)) == 1536
print("[PASS] All 1536 checks are scheduled exactly once!")
