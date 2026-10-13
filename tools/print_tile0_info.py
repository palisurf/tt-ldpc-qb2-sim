# Reproduce schedule partitioning and inspect Tile 0
import collections

def parse_chinn():
    c2v = collections.defaultdict(list)
    v2c = collections.defaultdict(list)
    with open('/home/ttuser/LDPC_QB2_GA/matrices/AR4JA_r45_4c_128c_r12.chinn.out') as f:
        for line in f:
            p = line.strip().split()
            if len(p) == 3 and float(p[2]) != 0:
                c, v = int(p[0]), int(p[1])
                c2v[c].append(v)
                v2c[v].append(c)
    return c2v, v2c

c2v, v2c = parse_chinn()
deg6 = [c for c in sorted(c2v.keys()) if len(c2v[c]) == 6]
deg3 = [c for c in sorted(c2v.keys()) if len(c2v[c]) == 3]

def partition(check_list, target_count, max_size):
    check_conflicts = []
    for c in check_list:
        conflicts = set()
        for v in c2v[c]:
            for other in v2c[v]:
                if other != c:
                    conflicts.add(other)
        check_conflicts.append((c, len(conflicts)))
    check_conflicts.sort(key=lambda x: (-x[1], x[0]))

    out_tiles = [[] for _ in range(target_count)]
    tile_vars = [set() for _ in range(target_count)]
    for c, _ in check_conflicts:
        c_vars = set(c2v[c])
        best_t = -1
        min_sz = max_size + 1
        for t in range(target_count):
            if len(out_tiles[t]) < max_size:
                if not (c_vars & tile_vars[t]) and len(out_tiles[t]) < min_sz:
                    min_sz = len(out_tiles[t])
                    best_t = t
        assert best_t != -1
        out_tiles[best_t].append(c)
        tile_vars[best_t].update(c_vars)
    return out_tiles

deg6_tiles = partition(deg6, 32, 32)
tile0 = deg6_tiles[0]
print(f"Tile 0 size: {len(tile0)}")
print("Tile 0 checks:", tile0)
for idx, c in enumerate(tile0):
    punct_count = sum(1 for v in c2v[c] if v >= 2048)
    print(f"Check idx={idx:2d} (c={c:4d}, deg={len(c2v[c])}): punct_count={punct_count}, vars={c2v[c]}")
