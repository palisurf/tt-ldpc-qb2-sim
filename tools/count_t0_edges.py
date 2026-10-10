import struct
import numpy as np

# Load h_table from remote or check how many edges in matrix touch 0..255
# In AR4JA matrix, let's count total edges touching variables 0..255!
edges_in_t0 = 0
with open('matrices/AR4JA_r45_4c_128c_r12.chinn.out', 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 3:
            c, v, val = int(parts[0]) - 1, int(parts[1]) - 1, float(parts[2])
            if val > 0.5 and v < 256:
                edges_in_t0 += 1

print(f"Total non-zero edges in matrix connecting to variables 0..255: {edges_in_t0}")
