import numpy as np

import os
dump_dir = "engines/tile_matrix_4cw/dump"
if not os.path.exists(f"{dump_dir}/golden_final_vnodes.bin"):
    dump_dir = "Z:/LDPC_QB2_GA/engines/tile_matrix_4cw/dump"

golden_bytes = open(f"{dump_dir}/golden_final_vnodes.bin", "rb").read()[:20480]
tensix_bytes = open(f"{dump_dir}/tensix_final_vnodes.bin", "rb").read()[:20480]
init_bytes   = open(f"{dump_dir}/tensix_initial_vnodes.bin", "rb").read()[:20480]

g_words = np.frombuffer(golden_bytes, dtype=np.uint16)
t_words = np.frombuffer(tensix_bytes, dtype=np.uint16)
i_words = np.frombuffer(init_bytes, dtype=np.uint16)

mismatches = []
for idx in range(len(g_words)):
    if g_words[idx] != t_words[idx]:
        mismatches.append(idx)

print(f"Total mismatching words: {len(mismatches)} / {len(g_words)}")

# Invert get_tile_vnode_byte_offset to find variable and codeword
def word_to_vnode_cw(word_idx):
    byte_off = word_idx * 2
    tile = byte_off // 2048
    rem = byte_off % 2048
    face = rem // 512
    face_rem = rem % 512
    row = face_rem // 32
    col_byte = face_rem % 32
    col_elem = col_byte // 2
    cw = col_elem % 4
    col_grp = col_elem // 4
    
    # row in tile (0..31):
    tile_row = (row + 16) if (face in [2, 3]) else row
    # k in tile (0..7):
    k = (col_grp + 4) if (face in [1, 3]) else col_grp
    v_loc = tile_row * 8 + k
    v = tile * 256 + v_loc
    return v, cw, face, row, col_elem

print("\nFirst 20 mismatching words breakdown:")
print(f"{'Word':>6} | {'Var':>5} | {'CW':>2} | {'Face':>4} | {'Row':>3} | {'Col':>3} | {'Init (hex)':>10} | {'Golden':>10} | {'Tensix':>10}")
print("-" * 75)
for idx in mismatches[:20]:
    v, cw, face, row, col_elem = word_to_vnode_cw(idx)
    iw = f"{i_words[idx]:04x}"
    gw = f"{g_words[idx]:04x}"
    tw = f"{t_words[idx]:04x}"
    print(f"{idx:6d} | {v:5d} | {cw:2d} | {face:4d} | {row:3d} | {col_elem:3d} | {iw:>10} | {gw:>10} | {tw:>10}")
