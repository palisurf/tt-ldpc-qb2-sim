import numpy as np
import os

dump_dir = "engines/tile_matrix_4cw/dump"
if not os.path.exists(f"{dump_dir}/c_tile0_offsets.txt"):
    dump_dir = "Z:/LDPC_QB2_GA/engines/tile_matrix_4cw/dump"

lines = open(f"{dump_dir}/c_tile0_offsets.txt").readlines()
c_tile0_offsets = {}
for line in lines:
    parts = line.strip().split()
    off, v, desc = int(parts[0]), int(parts[1]), " ".join(parts[2:])
    c_tile0_offsets[off] = (v, desc)

init = np.fromfile(f"{dump_dir}/tensix_initial_vnodes.bin", dtype=np.uint16)
final = np.fromfile(f"{dump_dir}/tensix_final_vnodes.bin", dtype=np.uint16)

diff_indices = np.where(init != final)[0]
changed_byte_offsets = sorted(list(set([(idx * 2) & ~7 for idx in diff_indices])))

in_tile0 = [off for off in changed_byte_offsets if off in c_tile0_offsets]
not_in_tile0 = [off for off in changed_byte_offsets if off not in c_tile0_offsets]

print(f"Total unique offsets changed in Tensix: {len(changed_byte_offsets)}")
print(f"Changed offsets IN C++ Tile 0: {len(in_tile0)} / {len(changed_byte_offsets)}")
print(f"Changed offsets NOT in C++ Tile 0: {len(not_in_tile0)}")
if len(not_in_tile0) > 0:
    print(f"Not in Tile 0: {not_in_tile0}")
else:
    print(">>> 100% OF THE 44 CHANGED OFFSETS IN TENSIX ARE IN TILE 0! <<<")
