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

deg6_checks = [m for m in range(M) if len(check_to_var[m]) == 6]
deg3_checks = [m for m in range(M) if len(check_to_var[m]) == 3]

# Replicate partition_checks_balanced
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
        assert best_tile != -1, f"Failed for check {c}"
        out_tiles[best_tile].append(c)
        for v in c_vars:
            tile_vars[best_tile].add(v)
    return out_tiles

deg6_tiles = partition_checks_balanced(deg6_checks, 32, 32)
tile0_checks = deg6_tiles[0]
print(f"Tile 0 checks (32 checks): {tile0_checks}")

# Now check which variables are connected to Tile 0 checks:
tile0_vars = set()
for c in tile0_checks:
    for v in check_to_var[c]:
        tile0_vars.add(v)

print(f"Tile 0 total unique variables (6 * 32 = 192 edges, no collisions = 192 variables): {len(tile0_vars)}")

test_vars = [33, 42, 77, 171, 219, 213, 222]
for v in test_vars:
    print(f"Var {v:4d} in tile0_vars? {v in tile0_vars}")
