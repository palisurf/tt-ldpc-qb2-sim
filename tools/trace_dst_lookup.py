def dst_lookup(row, col):
    # From tensix.cpp:
    # row = 4 * (r // 2) + lane // 8
    # col = 2 * (lane % 8) + (r % 2)
    #
    # We invert this mapping to find (r, lane) for any (row, col):
    r_div_2 = row // 4
    lane_div_8 = row % 4
    lane_mod_8 = col // 2
    r_mod_2 = col % 2
    
    r = 2 * r_div_2 + r_mod_2
    lane = 8 * lane_div_8 + lane_mod_8
    return r, lane

print("Mapping of 4 contiguous columns in row to (register r, lane):")
for row in range(12):
    for col_grp in range(4):
        c_base = col_grp * 4
        locs = [dst_lookup(row, c_base + c) for c in range(4)]
        regs = [loc[0] for loc in locs]
        lanes = [loc[1] for loc in locs]
        print(f"Row {row:2d}, cols {c_base:2d}..{c_base+3:2d} -> regs={regs}, lanes={lanes}")
