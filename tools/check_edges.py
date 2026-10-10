with open('matrices/AR4JA_r45_4c_128c_r12.chinn.out', 'r') as f:
    lines = f.readlines()

edges = {}
for line in lines:
    parts = line.strip().split()
    if len(parts) >= 3:
        r, c, val = int(parts[0]) - 1, int(parts[1]) - 1, float(parts[2])
        if val != 0.0:
            edges.setdefault(r, []).append(c)

print(f"Check 212 vars: {edges[212]}")
print(f"Check 1515 vars: {edges[1515]}")
