lines = open('matrices/AR4JA_r45_4c_128c_r12.chinn.out').readlines()
v0_checks = []
for l in lines[:-1]:
    r, c, v = int(l.split()[0]), int(l.split()[1]), float(l.split()[2])
    if c == 1 and abs(v) > 1e-5:
        v0_checks.append(r - 1)
print("Var 0 connected checks (0-indexed):", v0_checks)
