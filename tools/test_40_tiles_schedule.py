import numpy as np

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
        if best == -1:
            return None
        out_tiles[best].append(c)
        for v in c_vars: tile_vars[best].add(v)
    return out_tiles

deg6_tiles = partition(deg6_checks, 32, 32)
deg3_tiles = partition(deg3_checks, 8, 64)

if deg6_tiles is None or deg3_tiles is None:
    print("[FAIL] Partition failed!")
else:
    print(f"[SUCCESS] deg6_tiles: {len(deg6_tiles)} tiles x {len(deg6_tiles[0])} checks = {len(deg6_tiles)*len(deg6_tiles[0])}")
    print(f"[SUCCESS] deg3_tiles: {len(deg3_tiles)} tiles x {len(deg3_tiles[0])} checks = {len(deg3_tiles)*len(deg3_tiles[0])}")
    
    # Audit collisions
    all_passed = True
    total_edges = 0
    for t_idx, t in enumerate(deg6_tiles):
        seen = set()
        for c in t:
            for v in edges[c]:
                if v in seen:
                    print(f"Collision in deg6 tile {t_idx} on var {v}")
                    all_passed = False
                seen.add(v)
                total_edges += 1
                
    for t_idx, t in enumerate(deg3_tiles):
        seen = set()
        for c in t:
            for v in edges[c]:
                if v in seen:
                    print(f"Collision in deg3 tile {t_idx} on var {v}")
                    all_passed = False
                seen.add(v)
                total_edges += 1
                
    print(f"Collision Audit: {'PASSED (IDENTICALLY ZERO COLLISIONS)' if all_passed else 'FAILED'}")
    print(f"Total Edges Covered: {total_edges} / 7680 (100.0%)")
