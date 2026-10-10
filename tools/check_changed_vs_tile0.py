import numpy as np
import os
from check_tile0_offsets import w0_pairs, w1_pairs

dump_dir = "engines/tile_matrix_4cw/dump"
if not os.path.exists(f"{dump_dir}/tensix_initial_vnodes.bin"):
    dump_dir = "Z:/LDPC_QB2_GA/engines/tile_matrix_4cw/dump"

init = np.fromfile(f"{dump_dir}/tensix_initial_vnodes.bin", dtype=np.uint16)
final = np.fromfile(f"{dump_dir}/tensix_final_vnodes.bin", dtype=np.uint16)

diff_indices = np.where(init != final)[0]
changed_byte_offsets = sorted(list(set([(idx * 2) & ~7 for idx in diff_indices])))

w0_offsets = set()
for p, (oe, oo, ve, vo) in enumerate(w0_pairs):
    w0_offsets.add(oe)
    w0_offsets.add(oo)

w1_offsets = set()
for p, (oe, oo, ve, vo) in enumerate(w1_pairs):
    w1_offsets.add(oe)
    w1_offsets.add(oo)

all_tile0_offsets = w0_offsets.union(w1_offsets)

in_tile0 = [off for off in changed_byte_offsets if off in all_tile0_offsets]
not_in_tile0 = [off for off in changed_byte_offsets if off not in all_tile0_offsets]

print(f"Total changed byte offsets in Tensix: {len(changed_byte_offsets)}")
print(f"Changed offsets IN Tile 0: {len(in_tile0)}")
print(f"Changed offsets NOT in Tile 0: {len(not_in_tile0)}")
print(f"Offsets IN Tile 0: {in_tile0}")
