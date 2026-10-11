def get_scratch_byte_offset(reg_idx, lane):
    f = reg_idx // 8
    r_face = reg_idx % 8
    row_in_face = (2 * r_face) if (lane < 16) else (2 * r_face + 1)
    col_in_face = lane if (lane < 16) else (lane - 16)
    return f * 512 + row_in_face * 32 + col_in_face * 2

print('Offsets generated for base_r = 16 (Face 2):')
rows_touched = set()
for r in range(6):
    for lane in range(32):
        off = get_scratch_byte_offset(16 + r, lane)
        idx = off // 2
        f = idx // 256
        row = (idx - f*256) // 16
        col = (idx - f*256) % 16
        rows_touched.add(row)

print('Rows touched in Face 2:', sorted(list(rows_touched)))
