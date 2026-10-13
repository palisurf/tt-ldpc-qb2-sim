# Let's check partition_checks_balanced
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

# In ldpc_sim_tile_matrix.cpp:
# partition_checks_balanced(deg6_checks, pcm, 32, 32, deg6_tiles);
# 1024 deg6 checks / 32 tiles = 32 checks per tile!

# But wait! Look at the checks in deg6:
for c in range(8):
    chk = deg6[c]
    print(f"Check {c} (ID {chk}) degree: {len(c_to_v[chk])}")
