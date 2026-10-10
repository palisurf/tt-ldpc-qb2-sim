import struct

# Load initial dump
f_init = open("engines/tile_matrix_4cw/dump/tensix_initial_vnodes.bin", "rb").read()[:20480]

with open("matrices/AR4JA_r45_4c_128c_r12.chinn.out") as f:
    lines = [line.strip().split() for line in f if line.strip()]

M = int(lines[-1][0])
N = int(lines[-1][1])
edges = [(int(p[0])-1, int(p[1])-1) for p in lines[:-1] if float(p[2]) > 0.5]

check_to_var = [[] for _ in range(M)]
for m, v in edges:
    check_to_var[m].append(v)

# Find which checks touch variable 0
checks_v0 = [m for m in range(M) if 0 in check_to_var[m]]
print(f"Checks touching variable 0: {checks_v0}")
for m in checks_v0:
    vars_in_m = check_to_var[m]
    print(f"Check {m} (deg {len(vars_in_m)}): vars={vars_in_m}")
