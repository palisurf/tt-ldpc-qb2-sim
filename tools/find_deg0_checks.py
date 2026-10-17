with open("matrices/AR4JA_r45_4c_128c_r12.chinn.out") as f:
    lines = [l.strip().split() for l in f if l.strip()]
dim = lines[-1]
M, N = int(dim[0]), int(dim[1])
c_to_v = [[] for _ in range(M)]
for l in lines[:-1]:
    if len(l) == 3 and float(l[2]) > 0.5:
        r, c = int(l[0]) - 1, int(l[1]) - 1
        c_to_v[r].append(c)

target_vnodes = [43, 906, 556, 1541, 655, 288, 259]
for v in target_vnodes:
    deg0_checks = [m for m in range(M) if c_to_v[m][0] == v]
    print(f"Vnode {v} is degree 0 for checks: {deg0_checks}")
