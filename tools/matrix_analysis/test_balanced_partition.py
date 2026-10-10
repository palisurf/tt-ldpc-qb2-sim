from collections import defaultdict

edges = defaultdict(list)
with open('matrices/AR4JA_r45_4c_128c_r12.chinn.out') as f:
    for line in f:
        parts = line.split()
        if len(parts) == 3 and float(parts[2]) != 0.0:
            edges[int(parts[0]) - 1].append(int(parts[1]) - 1)

deg6_checks = [c for c in range(1536) if len(edges[c]) == 6]
deg3_checks = [c for c in range(1536) if len(edges[c]) == 3]

def balanced_partition(check_list, target_tile_count, max_tile_size):
    # Sort checks by conflict degree
    conflict_count = {}
    var_to_c = defaultdict(list)
    for c in check_list:
        for v in edges[c]:
            var_to_c[v].append(c)
            
    for c in check_list:
        conflicts = set()
        for v in edges[c]:
            for other in var_to_c[v]:
                if other != c:
                    conflicts.add(other)
        conflict_count[c] = len(conflicts)
        
    sorted_checks = sorted(check_list, key=lambda c: -conflict_count[c])
    
    tiles = [[] for _ in range(target_tile_count)]
    tile_vars = [set() for _ in range(target_tile_count)]
    
    unplaced = []
    for c in sorted_checks:
        c_vars = set(edges[c])
        placed = False
        # Best fit: place in tile with fewest elements that doesn't conflict
        valid_tiles = [i for i in range(target_tile_count) 
                       if len(tiles[i]) < max_tile_size and not (c_vars & tile_vars[i])]
        if valid_tiles:
            # Pick valid tile with smallest current size
            valid_tiles.sort(key=lambda i: len(tiles[i]))
            best_t = valid_tiles[0]
            tiles[best_t].append(c)
            tile_vars[best_t] |= c_vars
            placed = True
        if not placed:
            unplaced.append(c)
            
    return tiles, unplaced

tiles6, unplaced6 = balanced_partition(deg6_checks, 26, 40)
print(f"Deg-6 checks into 26 tiles: unplaced={len(unplaced6)}, sizes={[len(t) for t in tiles6]}")

tiles3, unplaced3 = balanced_partition(deg3_checks, 7, 80)
print(f"Deg-3 checks into 7 tiles: unplaced={len(unplaced3)}, sizes={[len(t) for t in tiles3]}")
