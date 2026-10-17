import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inspect_tile0_h import deg6_tiles

target_checks = [9, 112, 268, 386, 475, 562, 663]
for t, tile in enumerate(deg6_tiles):
    match_count = sum(1 for c in target_checks if c in tile)
    if match_count > 0:
        print(f"deg6_tiles[{t}] matched {match_count} checks: {tile[:8]}")
