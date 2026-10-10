import numpy as np
import os

dump_dir = "engines/tile_matrix_4cw/dump"
if not os.path.exists(f"{dump_dir}/tensix_initial_vnodes.bin"):
    dump_dir = "Z:/LDPC_QB2_GA/engines/tile_matrix_4cw/dump"

init = np.fromfile(f"{dump_dir}/tensix_initial_vnodes.bin", dtype=np.uint16)
final = np.fromfile(f"{dump_dir}/tensix_final_vnodes.bin", dtype=np.uint16)
gold = np.fromfile(f"{dump_dir}/golden_final_vnodes.bin", dtype=np.uint16)

diff_indices = np.where(init != final)[0]
print(f"Total words changed in Tensix: {len(diff_indices)}")

# Check what byte offsets these correspond to:
changed_byte_offsets = set()
for idx in diff_indices:
    byte_off = (idx * 2) & ~7 # align to 8-byte group (4 codewords)
    changed_byte_offsets.add(byte_off)

print(f"Unique 8-byte variable offsets changed in Tensix: {len(changed_byte_offsets)}")
print(f"Sorted byte offsets: {sorted(list(changed_byte_offsets))}")
