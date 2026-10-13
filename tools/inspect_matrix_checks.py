# Inspect Tile 0 schedule directly from the matrix
def parse_matrix():
    check_to_var = {}
    with open('/home/ttuser/LDPC_QB2_GA/matrices/AR4JA_r45_4c_128c_r12.chinn.out') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3 and float(parts[2]) != 0:
                c, v = int(parts[0]), int(parts[1])
                check_to_var.setdefault(c, []).append(v)
    return check_to_var

c2v = parse_matrix()
deg6 = [c for c, v in sorted(c2v.items()) if len(v) == 6]
deg3 = [c for c, v in sorted(c2v.items()) if len(v) == 3]

print(f"Total deg6 checks: {len(deg6)}, total deg3 checks: {len(deg3)}")
print(f"First 32 deg6 checks: {deg6[:32]}")
