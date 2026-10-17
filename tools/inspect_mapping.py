def get_tile_coord(ix, lane):
    dst_row = ix * 2
    r_face = (dst_row & ~3) + lane // 8
    c_face = 2 * (lane & 7) + ((dst_row & 2) >> 1)
    
    face = r_face // 16
    row_in_face = r_face % 16
    
    if face == 0:
        tile_row = row_in_face
        tile_col = c_face
    elif face == 1:
        tile_row = row_in_face
        tile_col = 16 + c_face
    elif face == 2:
        tile_row = 16 + row_in_face
        tile_col = c_face
    elif face == 3:
        tile_row = 16 + row_in_face
        tile_col = 16 + c_face
    else:
        raise ValueError(f"Unknown face {face}")
        
    return tile_row, tile_col, face, row_in_face, c_face

# Now let's check what compute_trisc.cpp does!
# In compute_trisc.cpp:
# for f in range(4):
#     base_r = [0, 8, 16, 24][f]
#     # It loops r in range(6) (edges 0..5):
#     # For each edge r, it reads dst_reg[base_r + r]
#     # Across all 32 lanes!
# What tile coordinate does (base_r + r, lane) read?

print("=== CHECKING WHAT compute_trisc.cpp READS FROM THE 32x32 TILE ===")
for f in range(4):
    base_r = [0, 8, 16, 24][f]
    print(f"\n--- FACE {f} (base_r = {base_r}) ---")
    for r in range(6):
        reg = base_r + r
        # Let's inspect lanes 0, 1, 2, 3, 8, 16, 24
        coords = [get_tile_coord(reg, lane) for lane in range(32)]
        tile_rows = set(c[0] for c in coords)
        tile_cols = set(c[1] for c in coords)
        faces = set(c[2] for c in coords)
        print(f"  Edge r={r} (dst_reg[{reg:2d}]): face={faces} tile_rows={sorted(list(tile_rows))} tile_cols={sorted(list(tile_cols))}")
