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
var_to_check = {}
for r, c in edges:
    if r not in check_to_var: check_to_var[r] = []
    check_to_var[r].append(c)
    if c not in var_to_check: var_to_check[c] = []
    var_to_check[c].append(r)

test_vars = [33, 42, 77, 171, 219, 213, 222]
for v in test_vars:
    print(f"Var {v:4d}: connected checks = {var_to_check.get(v, [])}")
