from collections import defaultdict

edges = defaultdict(list)
with open('matrices/AR4JA_r45_4c_128c_r12.chinn.out') as f:
    for line in f:
        parts = line.split()
        if len(parts) == 3 and float(parts[2]) != 0.0:
            edges[int(parts[0]) - 1].append(int(parts[1]) - 1)

print("Total checks:", len(edges))
# Let's inspect the first 40 checks:
seen = {}
for c in range(40):
    for v in edges[c]:
        if v in seen:
            print(f"Collision on Var {v}: Check {c} collides with Check {seen[v]}")
        seen[v] = c
