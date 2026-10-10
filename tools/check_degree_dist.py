with open('matrices/AR4JA_r45_4c_128c_r12.chinn.out', 'r') as f:
    lines = f.readlines()

edges = {}
for line in lines:
    p = line.strip().split()
    if len(p) >= 3:
        r, c, v = int(p[0]) - 1, int(p[1]) - 1, float(p[2])
        if v != 0.0:
            edges.setdefault(r, []).append(c)

d6 = [m for m, vs in edges.items() if len(vs) == 6]
d3 = [m for m, vs in edges.items() if len(vs) == 3]
other = [m for m, vs in edges.items() if len(vs) not in (3, 6)]

print(f"Deg 6 checks: {len(d6)}")
print(f"Deg 3 checks: {len(d3)}")
print(f"Other checks: {len(other)}")
print(f"Check 1515 degree: {len(edges.get(1515, []))}")
