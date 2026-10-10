import numpy as np
import struct
import os

def b2f(u):
    u32 = int(u) << 16
    return struct.unpack('>f', struct.pack('>I', u32))[0]

dump_dir = "engines/tile_matrix_4cw/dump"
if not os.path.exists(f"{dump_dir}/golden_scratch_out.bin"):
    dump_dir = "/home/ttuser/LDPC_QB2_GA/engines/tile_matrix_4cw/dump"

g = np.fromfile(f"{dump_dir}/golden_scratch_out.bin", dtype=np.uint16)
t = np.fromfile(f"{dump_dir}/tensix_scratch_out.bin", dtype=np.uint16)

diff = np.where(g != t)[0]
print(f"Total differing words: {len(diff)}")

for f in range(4):
    f_diff = [i for i in diff if 256*f <= i < 256*(f+1)]
    print(f"Face {f} diff count: {len(f_diff)}")

print("\nDetailed diff breakdown in Face 2 and 3:")
for idx in diff[:30]:
    f = idx // 256
    rem = idx % 256
    row_in_face = rem // 16
    col_in_face = rem % 16
    r_in_face = row_in_face // 2
    is_odd_row = row_in_face % 2
    lane = col_in_face if is_odd_row == 0 else col_in_face + 16
    g_val, t_val = b2f(g[idx]), b2f(t[idx])
    print(f"Idx {idx}: Face {f}, RowInFace {row_in_face} (r_in_face={r_in_face}, odd={is_odd_row}), ColInFace {col_in_face} -> Lane {lane}: Gold={g_val:+.4f} (0x{g[idx]:04x}), Tensix={t_val:+.4f} (0x{t[idx]:04x})")
