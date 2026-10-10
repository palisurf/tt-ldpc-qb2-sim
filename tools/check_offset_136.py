import numpy as np

# In ldpc_sim_tile_matrix.cpp, what did pcm.h_offsets_pairs contain for Tile 0?
# Let's inspect tools/check_tile0_offsets.py output for ALL 48 w0 pairs and 48 w1 pairs:
import sys
from check_tile0_offsets import w0_pairs, w1_pairs

w0_offsets = []
for p, (oe, oo, ve, vo) in enumerate(w0_pairs):
    w0_offsets.append((oe, ve, f"w0_even p={p}"))
    w0_offsets.append((oo, vo, f"w0_odd p={p}"))

w1_offsets = []
for p, (oe, oo, ve, vo) in enumerate(w1_pairs):
    w1_offsets.append((oe, ve, f"w1_even p={p}"))
    w1_offsets.append((oo, vo, f"w1_odd p={p}"))

all_offsets = w0_offsets + w1_offsets
offsets_dict = {}
for off, v, desc in all_offsets:
    offsets_dict[off] = (v, desc)

print(f"Total offsets in Tile 0: {len(all_offsets)}")
print(f"Is offset 136 in Tile 0 offsets? {136 in offsets_dict}")
if 136 in offsets_dict:
    print(f"Offset 136 found: {offsets_dict[136]}")
else:
    print("Offset 136 is NOT in Tile 0 offsets!")
