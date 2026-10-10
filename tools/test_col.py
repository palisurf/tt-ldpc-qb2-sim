from collections import defaultdict

c2v = defaultdict(list)
with open("matrices/AR4JA_r45_4c_128c_r12.chinn.out") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 3 and float(parts[2]) > 0.5:
            c2v[int(parts[0])].append(int(parts[1]))

print(f"Total check nodes loaded: {len(c2v)}")

# 1. Check sequential 32 checks
for block_size in [8, 16, 32]:
    collisions = 0
    total_blocks = 0
    for b in range(0, 1024, block_size):
        v_seen = set()
        total_blocks += 1
        for c in range(b, min(b + block_size, 1024)):
            for v in c2v[c]:
                if v in v_seen:
                    collisions += 1
                v_seen.add(v)
    print(f"Block size {block_size} (sequential): total collisions = {collisions} across {total_blocks} blocks")

# 2. Check strided 32 checks across circulants
for stride in [8, 16, 32, 64, 128]:
    collisions = 0
    for start in range(0, 512, 32 * stride):
        for offset in range(stride):
            v_seen = set()
            for k in range(32):
                c = start + offset + k * stride
                if c < 512:
                    for v in c2v[c]:
                        if v in v_seen:
                            collisions += 1
                        v_seen.add(v)
    print(f"Circulant strided 32 checks (stride {stride}): collisions = {collisions}")
