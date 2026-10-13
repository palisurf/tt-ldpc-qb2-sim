with open('/home/ttuser/LDPC_QB2_GA/matrices/AR4JA_r45_4c_128c_r12.chinn.out') as f:
    lines = [l.strip().split() for l in f if l.strip()]
edges = [(int(l[0]), int(l[1])) for l in lines if len(l) == 3 and float(l[2]) > 0.5]
c0_vars = [e[1] for e in edges if e[0] == 0]
print('Check 0 variable nodes:', c0_vars)
