from collections import defaultdict
import numpy as np

c2v = defaultdict(list)
v2c = defaultdict(list)
with open("matrices/AR4JA_r45_4c_128c_r12.chinn.out") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 3 and float(parts[2]) > 0.5:
            r, c = int(parts[0]), int(parts[1])
            c2v[r].append(c)
            v2c[c].append(r)

deg6_checks = [c for c in range(1536) if len(c2v[c]) == 6]
deg3_checks = [c for c in range(1536) if len(c2v[c]) == 3]

print(f"Deg-6 checks: {len(deg6_checks)}, Deg-3 checks: {len(deg3_checks)}")

def test_partition(check_list, max_tile_size, num_tiles):
    # Greedy graph coloring
    check_conflicts = []
    for c in check_list:
        conflicts = set()
        for v in c2v[c]:
            for other in v2c[v]:
                if other != c and other in check_list:
                    conflicts.add(other)
        check_conflicts.append((c, len(conflicts)))
    
    check_conflicts.sort(key=lambda x: x[1], reverse=True)
    
    tiles = [[] for _ in range(num_tiles)]
    tile_vars = [set() for _ in range(num_tiles)]
    unplaced = 0
    
    for c, _ in check_conflicts:
        c_vars = set(c2v[c])
        best_t = -1
        min_size = max_tile_size + 1
        for t in range(num_tiles):
            if len(tiles[t]) < max_tile_size:
                if not (c_vars & tile_vars[t]):
                    if len(tiles[t]) < min_size:
                        min_size = len(tiles[t])
                        best_t = t
        if best_t == -1:
            unplaced += 1
        else:
            tiles[best_t].append(c)
            tile_vars[best_t].update(c_vars)
            
    print(f"Max size {max_tile_size}, Num tiles {num_tiles}: unplaced = {unplaced}, tile sizes = {[len(t) for t in tiles]}")
    return unplaced

print("Testing 160 checks per tile for Deg-6 (1024 checks total):")
for num_t in [7, 8, 9, 10, 12, 14, 16]:
    test_partition(deg6_checks, 160, num_t)

print("\nTesting 320 checks per tile for Deg-3 (512 checks total):")
for num_t in [2, 3, 4, 5, 6]:
    test_partition(deg3_checks, 320, num_t)
