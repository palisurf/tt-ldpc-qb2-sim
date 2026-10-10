# Map every element in Face 0 of scratch_in
# 1. From SCRATCH_FACE_LUT:
# For p in 0..23 (Face 0):
# d = p // 4 (edge index 0..5)
# kh = p % 4 (check index in group)
# c0_even = 0 + kh (checks 0, 1, 2, 3)
# c0_odd  = 4 + kh (checks 4, 5, 6, 7)
# w0_even[p] = (2 * d) * 32 + kh * 8
# w0_odd[p]  = (2 * d + 1) * 32 + kh * 8
# Each w0_* writes 8 bytes = 4 bf16s (codeword 0, 1, 2, 3)

lut_map = {} # byte_offset -> (check_idx, edge_d, cw)

for p in range(24):
    d = p // 4
    kh = p % 4
    c_even = kh
    c_odd = 4 + kh
    off_even = (2 * d) * 32 + kh * 8
    off_odd  = (2 * d + 1) * 32 + kh * 8
    for cw in range(4):
        lut_map[off_even + cw * 2] = (c_even, d, cw)
        lut_map[off_odd  + cw * 2] = (c_odd,  d, cw)

# 2. From get_scratch_byte_offset(base_r + r, lane):
def get_scratch_byte_offset(reg_idx, lane):
    f = reg_idx // 8
    b = (reg_idx % 8) // 2
    is_odd = reg_idx % 2
    row_in_block = lane // 8
    col_in_row = lane % 8
    row_in_face = b * 4 + row_in_block
    col_in_face = 2 * col_in_row + is_odd
    return (f * 256 + row_in_face * 16 + col_in_face) * 2

print("Comparing Face 0 (pass 0: base_r=0):")
print("Lane 0 for r=0..5:")
for r in range(6):
    off = get_scratch_byte_offset(r, 0)
    info = lut_map.get(off, "NOT_FOUND")
    print(f"  r={r}: off={off} (0x{off:03x}) -> {info}")

print("\nLane 1 for r=0..5:")
for r in range(6):
    off = get_scratch_byte_offset(r, 1)
    info = lut_map.get(off, "NOT_FOUND")
    print(f"  r={r}: off={off} (0x{off:03x}) -> {info}")
