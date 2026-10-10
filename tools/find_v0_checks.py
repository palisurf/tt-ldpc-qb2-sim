import numpy as np

# Load matrix to find checks for v=0
checks = []
with open('matrices/AR4JA_r45_4c_128c_r12.chinn.out', 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 3:
            r, c, val = int(parts[0]), int(parts[1]), float(parts[2])
            if c == 1 and val != 0.0: # variable 0 (1-indexed)
                checks.append(r - 1)

print(f"Variable 0 connects to checks: {checks}")
