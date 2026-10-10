import struct
import numpy as np

# Let's inspect which tile in the schedule touches Var 0
with open("matrices/AR4JA_r45_4c_128c_r12.chinn.out") as f:
    lines = [line.strip().split() for line in f if len(line.strip().split()) >= 3]

edges = {}
for p in lines:
    r, c, v = int(p[0]) - 1, int(p[1]) - 1, float(p[2])
    if v != 0.0:
        edges.setdefault(r, []).append(c)

checks_for_v0 = [m for m in range(1536) if 0 in edges[m]]
print(f"All checks connected to Var 0: {checks_for_v0}")
for m in checks_for_v0:
    print(f"Check {m}: degree={len(edges[m])}, vars={edges[m]}")
