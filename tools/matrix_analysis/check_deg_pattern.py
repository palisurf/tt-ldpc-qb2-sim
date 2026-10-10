from collections import defaultdict

edges = defaultdict(list)
with open('matrices/AR4JA_r45_4c_128c_r12.chinn.out') as f:
    for line in f:
        parts = line.split()
        if len(parts) == 3 and float(parts[2]) != 0.0:
            edges[int(parts[0]) - 1].append(int(parts[1]) - 1)

degs = [len(edges[i]) for i in range(1536)]
# Print first 24 degrees:
print("Degrees 0..23:", degs[:24])

# How often does degree 3 appear?
deg3_pos = [i for i, d in enumerate(degs) if d == 3]
print("First 20 positions of degree 3 checks:", deg3_pos[:20])
diffs = [deg3_pos[i+1] - deg3_pos[i] for i in range(len(deg3_pos)-1)]
print("Differences between consecutive deg-3 checks:", set(diffs[:50]))
