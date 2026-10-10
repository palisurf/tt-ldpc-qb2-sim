import numpy as np

# We can find which tile contains check 212 by checking the schedule in test_tile_matrix_lockstep_stepper or ldpc_sim_tile_matrix
# Let's inspect the h_offset table in ldpc_sim_tile_matrix.cpp

# In ldpc_sim_tile_matrix.cpp:
# tile schedule:
# deg6_checks has check 212?
# Let's write a script that replicates partition_checks_balanced exactly
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
        assert best != -1
        out_tiles[best].append(c)
        for v in c_vars: tile_vars[best].add(v)
    return out_tiles

deg6_tiles = partition(deg6_checks, 26, 40)
deg3_tiles = partition(deg3_checks, 7, 80)

# Build schedule
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

for t_idx, (t_checks, deg) in enumerate(schedule):
    if 212 in t_checks:
        c_pos = t_checks.index(212)
        print(f"Check 212 is in Tile {t_idx} (deg={deg}), position in tile: {c_pos}")
        # Find which worker and which check index in worker:
        b = c_pos // 8
        kh = c_pos % 8
        worker = 0 if kh < 4 else 1
        worker_k = kh if kh < 4 else kh - 4
        print(f"  Block b={b}, kh={kh} -> Worker {worker}, worker_kh={worker_k}")
        print(f"  Check 212 variables: {edges[212]}")
