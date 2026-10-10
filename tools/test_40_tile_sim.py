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

# Interleave
schedule = []
for i in range(8):
    for j in range(4):
        schedule.append((deg6_tiles[i * 4 + j], 6))
    schedule.append((deg3_tiles[i], 3))

# Dr. Jones AMin*
def amin_star(a, b):
    min_v = min(a, b)
    diff = abs(a - b)
    if diff < 0.5:
        corr = 0.69315 - 0.40 * diff
    elif diff < 2.0:
        corr = 0.49315 - 0.328 * (diff - 0.5)
    else:
        corr = 0.0
    return max(min_v - corr, 0.0)

# Simulate 1 iteration on initial LLRs (+3.172 for 0..2047, 0.0 for 2048..2559)
vnodes = np.full((pcm_N, 4), 3.172, dtype=np.float32)
vnodes[2048:2560, :] = 0.0

# Store initial for comparison
vnodes_initial = vnodes.copy()

# Execute schedule
for t_idx, (tile, deg) in enumerate(schedule):
    for m in tile:
        vars_m = edges[m]
        for cw in range(4):
            q = [vnodes[v, cw] for v in vars_m] # iter 0: R_old = 0
            signs = [1.0 if x >= 0 else -1.0 for x in q]
            mags = [abs(x) for x in q]
            total_sign = 1.0
            for s in signs: total_sign *= s
            
            min_mag = min(mags)
            min_idx = mags.index(min_mag)
            
            # Recursive AMin*
            others = [mags[i] for i in range(deg) if i != min_idx]
            m_others = others[0]
            for val in others[1:]:
                m_others = amin_star(m_others, val)
            m_all = amin_star(m_others, min_mag)
            
            for d, v in enumerate(vars_m):
                m_d = m_others if d == min_idx else m_all
                r_new = total_sign * signs[d] * m_d
                vnodes[v, cw] = q[d] + r_new

print(f"Var 0 final LLR: {vnodes[0, 0]}")
print(f"Var 0 initial LLR: {vnodes_initial[0, 0]}")
print(f"Var 0 unchanged: {vnodes[0, 0] == vnodes_initial[0, 0]}")
