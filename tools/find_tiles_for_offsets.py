import numpy as np
import os

from check_tile0_offsets import deg6_checks, partition_checks_balanced, check_to_var, get_tile_vnode_byte_offset, M, N

deg6_tiles = partition_checks_balanced(deg6_checks, 32, 32)
# deg3 checks
deg3_checks = [m for m in range(M) if len(check_to_var[m]) == 3]
deg3_tiles = partition_checks_balanced(deg3_checks, 8, 64)

# 40-tile schedule:
schedule_tiles = []
for i in range(8):
    for j in range(4):
        schedule_tiles.append((deg6_tiles[i * 4 + j], 6))
    schedule_tiles.append((deg3_tiles[i], 3))

print(f"Total tiles in schedule: {len(schedule_tiles)}")

# Build pairs for all 40 tiles
tile_offsets = {}
for t, (tile, deg) in enumerate(schedule_tiles):
    offs = set()
    for p in range(48):
        if deg == 6:
            if p < 24:
                d = p // 4; kh = p % 4
                c0_e = 0 + kh; c0_o = 4 + kh; c1_e = 8 + kh; c1_o = 12 + kh
            else:
                d = (p - 24) // 4; kh = (p - 24) % 4
                c0_e = 16 + kh; c0_o = 20 + kh; c1_e = 24 + kh; c1_o = 28 + kh
        else:
            if p < 24:
                blk = p // 12; d = (p % 12) // 4; kh = (p % 12) % 4
                c0_e = 0 + blk*8 + kh; c0_o = 4 + blk*8 + kh; c1_e = 16 + blk*8 + kh; c1_o = 20 + blk*8 + kh
            else:
                blk = (p - 24) // 12; d = ((p - 24) % 12) // 4; kh = ((p - 24) % 12) % 4
                c0_e = 32 + blk*8 + kh; c0_o = 36 + blk*8 + kh; c1_e = 48 + blk*8 + kh; c1_o = 52 + blk*8 + kh

        v0_e = check_to_var[tile[c0_e]][d] if c0_e < len(tile) else 2559
        v0_o = check_to_var[tile[c0_o]][d] if c0_o < len(tile) else 2559
        v1_e = check_to_var[tile[c1_e]][d] if c1_e < len(tile) else 2559
        v1_o = check_to_var[tile[c1_o]][d] if c1_o < len(tile) else 2559

        offs.add(get_tile_vnode_byte_offset(v0_e))
        offs.add(get_tile_vnode_byte_offset(v0_o))
        offs.add(get_tile_vnode_byte_offset(v1_e))
        offs.add(get_tile_vnode_byte_offset(v1_o))
    tile_offsets[t] = offs

from check_changed_vs_tile0 import changed_byte_offsets
print("\nChecking which tiles contain the 44 changed offsets:")
for off in changed_byte_offsets:
    containing_tiles = [t for t in range(40) if off in tile_offsets[t]]
    print(f"Offset {off:5d}: in tiles {containing_tiles}")
