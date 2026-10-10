# Count how many variables in 0..255 are connected to checks
var_deg = {}
with open('matrices/AR4JA_r45_4c_128c_r12.chinn.out', 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 3:
            c, v, val = int(parts[0]) - 1, int(parts[1]) - 1, float(parts[2])
            if val > 0.5:
                var_deg[v] = var_deg.get(v, 0) + 1

t0_vars = [v for v in range(256) if v in var_deg]
print(f"Total variables in 0..255 with degree > 0: {len(t0_vars)} / 256")
print(f"Total words that should be updated in Tile 0 (4 words/var): {len(t0_vars) * 4} / 1024")

# Variables with degree == 0 (punctured or unused?):
unused_vars = [v for v in range(256) if v not in var_deg]
print(f"Unused / degree-0 variables in 0..255: {len(unused_vars)}")
