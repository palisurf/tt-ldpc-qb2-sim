import numpy as np

pcm_M = 1536
pcm_N = 2560
edges = {}
with open('matrices/AR4JA_r45_4c_128c_r12.chinn.out', 'r') as f:
    for line in f:
        p = line.strip().split()
        if len(p) >= 3:
            r, c, v = int(p[0]) - 1, int(p[1]) - 1, float(p[2])
            if v != 0.0:
                edges.setdefault(r, []).append(c)

# Let's inspect Check 1515
print("Check 1515 vars:", edges[1515])
for v in edges[1515]:
    print(f"  v={v} (punctured={v >= 2048})")
