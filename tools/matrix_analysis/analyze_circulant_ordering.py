from collections import defaultdict
import numpy as np

edges = defaultdict(list)
with open('matrices/AR4JA_r45_4c_128c_r12.chinn.out') as f:
    for line in f:
        parts = line.split()
        if len(parts) == 3 and float(parts[2]) != 0.0:
            edges[int(parts[0]) - 1].append(int(parts[1]) - 1)

# Check degrees of each check node:
degs = [len(edges[i]) for i in range(1536)]
print("Degree counts:", {d: degs.count(d) for d in set(degs)})

# How are degree-6 and degree-3 checks distributed across rows 0..1535?
deg6_indices = [i for i, d in enumerate(degs) if d == 6]
deg3_indices = [i for i, d in enumerate(degs) if d == 3]

print(f"Degree 6 indices range: min={min(deg6_indices)}, max={max(deg6_indices)}")
print(f"Degree 3 indices range: min={min(deg3_indices)}, max={max(deg3_indices)}")

# Let's inspect differences between consecutive check connections:
# In a circulant of size p, check(c) and check(c+1) have cyclic shifts
# Let's check difference: (edges[c+1] - edges[c])
diffs = []
for c in range(10):
    print(f"Check {c}: {sorted(edges[c])}")
