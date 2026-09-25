from collections import defaultdict, Counter

with open("matrices/AR4JA_r45_4c_128c_r12.chinn.out") as f:
    lines = [l.strip().split() for l in f if l.strip() and not l.startswith("#")]

degs = defaultdict(int)
for r, c, v in lines[:-1]:
    if float(v) != 0:
        degs[int(c)-1] += 1

first_1024 = Counter([degs[i] for i in range(1024)])
mid_1024 = Counter([degs[i] for i in range(1024, 2048)])
last_512 = Counter([degs[i] for i in range(2048, 2560)])

print("Degrees of nodes 0..1023 (Information bits?):", sorted(first_1024.items()))
print("Degrees of nodes 1024..2047 (Parity bits?)   :", sorted(mid_1024.items()))
print("Degrees of nodes 2048..2559 (Punctured?)     :", sorted(last_512.items()))

# Total degree distribution
all_degs = Counter([degs[i] for i in range(2560)])
print("Total degree distribution across all 2560 nodes:", sorted(all_degs.items()))
