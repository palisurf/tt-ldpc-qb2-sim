# Test Deg 6 pairing
w0_pairs_d6 = []
for p in range(48):
    if p < 24:
        d = p // 4
        kh = p % 4
        c_even = 0 + kh
        c_odd  = 4 + kh
    else:
        d = (p - 24) // 4
        kh = (p - 24) % 4
        c_even = 8 + kh
        c_odd  = 12 + kh
    w0_pairs_d6.append(((c_even, d), (c_odd, d)))

# Collect all (c, d) for Worker 0 (checks 0..15, deg 6)
all_w0_d6 = []
for (c0, d0), (c1, d1) in w0_pairs_d6:
    all_w0_d6.append((c0, d0))
    all_w0_d6.append((c1, d1))

expected_w0_d6 = [(c, d) for c in range(16) for d in range(6)]
assert sorted(all_w0_d6) == sorted(expected_w0_d6)
print(f"[PASS] Deg 6 Worker 0 covered all {len(all_w0_d6)} edges perfectly!")

# Test Deg 3 pairing
w0_pairs_d3 = []
for p in range(48):
    block_pair = p // 12 # 0..3
    p_in_bp = p % 12
    d = p_in_bp // 4     # 0..2
    kh = p_in_bp % 4    # 0..3
    c_even = block_pair * 8 + kh
    c_odd  = block_pair * 8 + 4 + kh
    w0_pairs_d3.append(((c_even, d), (c_odd, d)))

all_w0_d3 = []
for (c0, d0), (c1, d1) in w0_pairs_d3:
    all_w0_d3.append((c0, d0))
    all_w0_d3.append((c1, d1))

expected_w0_d3 = [(c, d) for c in range(32) for d in range(3)]
assert sorted(all_w0_d3) == sorted(expected_w0_d3)
print(f"[PASS] Deg 3 Worker 0 covered all {len(all_w0_d3)} edges perfectly!")
