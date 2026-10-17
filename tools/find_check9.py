import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inspect_tile0_h import deg6_tiles

for t, tile in enumerate(deg6_tiles):
    if 9 in tile:
        idx = tile.index(9)
        print(f"Check 9 is in deg6_tiles[{t}] at index {idx}")
