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
    # dim is last line
    dim_line = lines[-1].strip().split()
    M, N = int(dim_line[0]), int(dim_line[1])
    return M, N, edges

M, N, edges = parse_pcm("matrices/AR4JA_r45_4c_128c_r12.chinn.out")
v1_checks = [r for (r, c) in edges if c == 1]
print(f"Variable 1 connects to {len(v1_checks)} checks: {v1_checks}")
v0_checks = [r for (r, c) in edges if c == 0]
print(f"Variable 0 connects to {len(v0_checks)} checks: {v0_checks}")

check_to_var = {}
for r, c in edges:
    if r not in check_to_var: check_to_var[r] = []
    check_to_var[r].append(c)

deg6_checks = [m for m in range(M) if len(check_to_var[m]) == 6]
deg3_checks = [m for m in range(M) if len(check_to_var[m]) == 3]
print(f"Total deg6 checks: {len(deg6_checks)}, deg3 checks: {len(deg3_checks)}")
print(f"Check 974 degree: {len(check_to_var[974])}, vars: {check_to_var[974]}")

