# Locate which tile and check touches byte 24 in AR4JA Rate-1/2
import numpy as np

# Load matrix
check_to_var = {}
with open('matrices/AR4JA_r45_4c_128c_r12.chinn.out', 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 3:
            c, v, val = int(parts[0]) - 1, int(parts[1]) - 1, float(parts[2])
            if val > 0.5:
                if c not in check_to_var: check_to_var[c] = []
                check_to_var[c].append(v)

print(f"Total checks: {len(check_to_var)}")
# Checks touching var 3
c_var3 = [c for c, vs in check_to_var.items() if 3 in vs]
print(f"Checks touching Var 3: {c_var3}")
