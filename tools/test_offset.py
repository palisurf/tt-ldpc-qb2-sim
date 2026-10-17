def get_scratch_byte_offset(reg_idx, lane):
    face = reg_idx // 8
    r_in_face = reg_idx % 8
    k = r_in_face // 2
    is_odd_reg = r_in_face % 2
    row_in_quartet = lane // 8
    c_pair = lane % 8
    row_in_face = 4 * k + row_in_quartet
    col_in_face = 2 * c_pair + is_odd_reg
    return face * 512 + row_in_face * 32 + col_in_face * 2

print("Reg 14 lane 0 (CW0):", get_scratch_byte_offset(14, 0))
print("Reg 15 lane 0 (CW1):", get_scratch_byte_offset(15, 0))
print("Reg 14 lane 1 (CW2):", get_scratch_byte_offset(14, 1))
print("Reg 15 lane 1 (CW3):", get_scratch_byte_offset(15, 1))
