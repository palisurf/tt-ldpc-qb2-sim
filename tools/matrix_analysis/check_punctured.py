from collections import defaultdict, Counter

with open("matrices/AR4JA_r45_4c_128c_r12.chinn.out") as f:
    lines = [l.strip().split() for l in f if l.strip() and not l.startswith("#")]

c2v = defaultdict(list)
for r, c, v in lines[:-1]:
    if float(v) != 0:
        c2v[int(r)-1].append(int(c)-1)

punctured = set(range(2048, 2560))
p_counts = [sum(1 for vn in v if vn in punctured) for v in c2v.values()]
print("Punctured nodes per check node (count: num_check_nodes):", sorted(Counter(p_counts).items()))
