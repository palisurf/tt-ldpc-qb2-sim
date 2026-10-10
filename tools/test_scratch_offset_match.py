def get_scratch_byte_offset(reg_idx, col):
    f = reg_idx // 8
    r_in_face = reg_idx % 8
    row_in_face = (2 * r_in_face) if (col < 16) else (2 * r_in_face + 1)
    col_in_face = col % 16
    return f * 512 + row_in_face * 32 + col_in_face * 2

def get_lut():
    w0_even, w0_odd = [], []
    w1_even, w1_odd = [], []
    for p in range(48):
        face_w0 = 0 if p < 24 else 2
        face_w1 = 1 if p < 24 else 3
        p_in_half = p if p < 24 else (p - 24)
        d = p_in_half // 4
        kh = p_in_half % 4
        w0_even.append(face_w0 * 512 + (2 * d) * 32 + kh * 8)
        w0_odd.append(face_w0 * 512 + (2 * d + 1) * 32 + kh * 8)
        w1_even.append(face_w1 * 512 + (2 * d) * 32 + kh * 8)
        w1_odd.append(face_w1 * 512 + (2 * d + 1) * 32 + kh * 8)
    return w0_even, w0_odd, w1_even, w1_odd

w0_even, w0_odd, w1_even, w1_odd = get_lut()

# Collect all offsets written by Golden compute for deg==6
deg6_face_bases = [0, 8, 16, 24]
golden_offsets = set()
for pass_idx in range(4):
    base_r = deg6_face_bases[pass_idx]
    for lane in range(32):
        col = lane
        for r in range(6):
            off = get_scratch_byte_offset(base_r + r, col)
            golden_offsets.add(off)

print(f"Total golden offsets in deg-6: {len(golden_offsets)}")

# Check how many bytes of s0_even (8 bytes each) are covered by golden_offsets
s0_even_covered = 0
for p in range(48):
    base = w0_even[p]
    # Each uint64 is 8 bytes = 4 uint16 elements: base+0, base+2, base+4, base+6
    covered = [base + 2*i in golden_offsets for i in range(4)]
    if all(covered):
        s0_even_covered += 1
    else:
        print(f"p={p}, base={base}: covered={covered}")

print(f"w0_even covered: {s0_even_covered} / 48")

s0_odd_covered = sum(all(w0_odd[p] + 2*i in golden_offsets for i in range(4)) for p in range(48))
s1_even_covered = sum(all(w1_even[p] + 2*i in golden_offsets for i in range(4)) for p in range(48))
s1_odd_covered = sum(all(w1_odd[p] + 2*i in golden_offsets for i in range(4)) for p in range(48))
print(f"w0_odd covered:  {s0_odd_covered} / 48")
print(f"w1_even covered: {s1_even_covered} / 48")
print(f"w1_odd covered:  {s1_odd_covered} / 48")

# Now check deg==3
deg3_face_bases = [0, 3, 8, 11, 16, 19, 24, 27]
deg3_offsets = set()
for pass_idx in range(8):
    base_r = deg3_face_bases[pass_idx]
    for lane in range(32):
        col = lane
        for r in range(3):
            off = get_scratch_byte_offset(base_r + r, col)
            deg3_offsets.add(off)

print(f"\nTotal golden offsets in deg-3: {len(deg3_offsets)}")
s0_even_deg3 = sum(all(w0_even[p] + 2*i in deg3_offsets for i in range(4)) for p in range(48))
s0_odd_deg3  = sum(all(w0_odd[p] + 2*i in deg3_offsets for i in range(4)) for p in range(48))
s1_even_deg3 = sum(all(w1_even[p] + 2*i in deg3_offsets for i in range(4)) for p in range(48))
s1_odd_deg3  = sum(all(w1_odd[p] + 2*i in deg3_offsets for i in range(4)) for p in range(48))
print(f"Deg3 w0_even: {s0_even_deg3}/48, w0_odd: {s0_odd_deg3}/48, w1_even: {s1_even_deg3}/48, w1_odd: {s1_odd_deg3}/48")

