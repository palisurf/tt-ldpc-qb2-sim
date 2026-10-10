from collections import defaultdict

edges = defaultdict(list)
with open('matrices/AR4JA_r45_4c_128c_r12.chinn.out') as f:
    for line in f:
        parts = line.split()
        if len(parts) == 3 and float(parts[2]) != 0.0:
            edges[int(parts[0]) - 1].append(int(parts[1]) - 1)

# Let's build a Greedy Independent Set partition of checks:
# Two checks c1 and c2 have an edge in the interference graph if set(edges[c1]) & set(edges[c2]) != empty.

deg6_checks = [c for c in range(1536) if len(edges[c]) == 6]
deg3_checks = [c for c in range(1536) if len(edges[c]) == 3]

print(f"Total deg-6 checks: {len(deg6_checks)}, deg-3 checks: {len(deg3_checks)}")

def partition_into_independent_tiles(check_list, max_tile_size):
    tiles = []
    unassigned = list(check_list)
    
    while unassigned:
        current_tile = []
        occupied_vars = set()
        remaining = []
        
        for c in unassigned:
            c_vars = set(edges[c])
            if len(current_tile) < max_tile_size and not (c_vars & occupied_vars):
                current_tile.append(c)
                occupied_vars |= c_vars
            else:
                remaining.append(c)
                
        tiles.append(current_tile)
        unassigned = remaining
        
    return tiles

tiles_deg6 = partition_into_independent_tiles(deg6_checks, 40)
tiles_deg3 = partition_into_independent_tiles(deg3_checks, 80)

print(f"Deg-6 checks partitioned into {len(tiles_deg6)} collision-free tiles:")
deg6_sizes = [len(t) for t in tiles_deg6]
print("Tile sizes:", deg6_sizes)

print(f"\nDeg-3 checks partitioned into {len(tiles_deg3)} collision-free tiles:")
deg3_sizes = [len(t) for t in tiles_deg3]
print("Tile sizes:", deg3_sizes)

print(f"\nTotal collision-free tiles across entire matrix: {len(tiles_deg6) + len(tiles_deg3)}")
