# Test deterministic partitioning
with open("matrices/AR4JA_r45_4c_128c_r12.chinn.out") as f:
    lines = [line.strip().split() for line in f if line.strip()]

M = int(lines[-1][0])
N = int(lines[-1][1])
edges = [(int(p[0])-1, int(p[1])-1) for p in lines[:-1] if float(p[2]) > 0.5]

check_to_var = [[] for _ in range(M)]
var_to_check = [[] for _ in range(N)]
for m, v in edges:
    check_to_var[m].append(v)
    var_to_check[v].append(m)

deg6_checks = [m for m in range(M) if len(check_to_var[m]) == 6]
deg3_checks = [m for m in range(M) if len(check_to_var[m]) == 3]

print(f"Total deg6 checks: {len(deg6_checks)}, deg3 checks: {len(deg3_checks)}")
