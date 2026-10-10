with open('matrices/AR4JA_r45_4c_128c_r12.chinn.out') as f:
    lines = f.readlines()
check_to_vars = {}
for line in lines[:-1]:
    parts = line.split()
    if len(parts) >= 3 and float(parts[2]) > 0.5:
        m, v = int(parts[0]) - 1, int(parts[1]) - 1
        if m not in check_to_vars: check_to_vars[m] = []
        check_to_vars[m].append(v)

deg6_checks = [m for m in range(1536) if len(check_to_vars.get(m, [])) == 6]
deg3_checks = [m for m in range(1536) if len(check_to_vars.get(m, [])) == 3]

print(f"Total deg6 checks: {len(deg6_checks)}, total deg3 checks: {len(deg3_checks)}")
print(f"Check 212 is in deg6: {212 in deg6_checks}, index in deg6: {deg6_checks.index(212)}")
print(f"Check 1515 is in deg6: {1515 in deg6_checks}, index in deg6: {deg6_checks.index(1515)}")
