w0_even = [0]*48
w0_odd  = [0]*48
w1_even = [0]*48
w1_odd  = [0]*48

for p in range(48):
    if p < 24:
        face_w0 = 0
        face_w1 = 1
        d = p // 4
        kh = p % 4
    else:
        face_w0 = 2
        face_w1 = 3
        d = (p - 24) // 4
        kh = (p - 24) % 4
    
    w0_even[p] = face_w0 * 512 + (2 * d) * 32 + kh * 8
    w0_odd[p]  = face_w0 * 512 + (2 * d + 1) * 32 + kh * 8
    w1_even[p] = face_w1 * 512 + (2 * d) * 32 + kh * 8
    w1_odd[p]  = face_w1 * 512 + (2 * d + 1) * 32 + kh * 8

# Check alignments and uniqueness
all_offsets = []
for p in range(48):
    for name, val in [("w0_even", w0_even[p]), ("w0_odd", w0_odd[p]), ("w1_even", w1_even[p]), ("w1_odd", w1_odd[p])]:
        assert val % 8 == 0, f"{name}[{p}] = {val} not 8-byte aligned!"
        assert 0 <= val <= 2048 - 8, f"{name}[{p}] = {val} out of bounds!"
        all_offsets.append(val)

assert len(all_offsets) == len(set(all_offsets)) == 48 * 4 == 192
print(f"[PASS] All {len(all_offsets)} 8-byte offsets are 100% unique and 8-byte aligned!")
print(f"Max offset: {max(all_offsets)} (within 2048 bytes)")
