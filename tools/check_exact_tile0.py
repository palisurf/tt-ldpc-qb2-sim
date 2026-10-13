import struct

with open('/home/ttuser/LDPC_QB2_GA/matrices/AR4JA_r45_4c_128c_r12.chinn.out') as f:
    lines = [l.strip().split() for l in f if l.strip()]
dim = lines[-1]
M, N = int(dim[0]), int(dim[1])
c_to_v = [[] for _ in range(M)]
for l in lines[:-1]:
    if len(l) == 3 and float(l[2]) > 0.5:
        r, c = int(l[0]) - 1, int(l[1]) - 1
        c_to_v[r].append(c)

deg6 = [m for m in range(M) if len(c_to_v[m]) == 6]
deg3 = [m for m in range(M) if len(c_to_v[m]) == 3]

def partition_checks_balanced(checks, target_tile_count, max_tile_size):
    out_tiles = [[] for _ in range(target_tile_count)]
    tile_vars = [set() for _ in range(target_tile_count)]
    for c in checks:
        c_vars = c_to_v[c]
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
print(f"Tile 0 has {len(tile0)} checks: {tile0[:8]}")

# Let's inspect the check degrees in Tile 0
for idx, c in enumerate(tile0[:8]):
    print(f"Check idx {idx} (ID {c}): vars={c_to_v[c]}")
