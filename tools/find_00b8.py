import sys, os
with open("matrices/AR4JA_r45_4c_128c_r12.chinn.out") as f:
    lines = [l.strip().split() for l in f if l.strip()]
dim = lines[-1]
M, N = int(dim[0]), int(dim[1])
c_to_v = [[] for _ in range(M)]
for l in lines[:-1]:
    if len(l) == 3 and float(l[2]) > 0.5:
        r, c = int(l[0]) - 1, int(l[1]) - 1
        c_to_v[r].append(c)

def get_tile_vnode_byte_offset(v):
    t = v // 256
    v_loc = v % 256
    row = v_loc // 8
    k = v_loc % 8
    face = (0 if k < 4 else 1) if row < 16 else (2 if k < 4 else 3)
    col_grp = k if k < 4 else (k - 4)
    return t * 2048 + (face * 512) + ((row & 15) << 5) + (col_grp << 3)

# For v = 43:
v43_off = get_tile_vnode_byte_offset(43)
print("Vnode 43 byte offset: 0x%04x (%d)" % (v43_off, v43_off))

# Which check nodes connect to vnode 43?
connected_checks = [m for m in range(M) if 43 in c_to_v[m]]
print("Checks connected to vnode 43:", connected_checks)
for c in connected_checks:
    print("  Check %d degree %d, vars: %s" % (c, len(c_to_v[c]), c_to_v[c]))
