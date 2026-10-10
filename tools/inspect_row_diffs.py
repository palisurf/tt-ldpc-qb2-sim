import numpy as np
import struct
import os

def b2f(u):
    u32 = int(u) << 16
    return struct.unpack('>f', struct.pack('>I', u32))[0]

dump_dir = "engines/tile_matrix_4cw/dump"
if not os.path.exists(f"{dump_dir}/golden_scratch_out.bin"):
    dump_dir = "/home/ttuser/LDPC_QB2_GA/engines/tile_matrix_4cw/dump"

g_in = np.fromfile(f"{dump_dir}/golden_scratch_in.bin", dtype=np.uint16)
g_out = np.fromfile(f"{dump_dir}/golden_scratch_out.bin", dtype=np.uint16)
t_out = np.fromfile(f"{dump_dir}/tensix_scratch_out.bin", dtype=np.uint16)

for f in [2, 3]:
    print(f"\n================ FACE {f} DIFFERENCES (16 rows x 16 cols) ================")
    f_g_out = g_out[256*f : 256*(f+1)].reshape(16, 16)
    f_t_out = t_out[256*f : 256*(f+1)].reshape(16, 16)
    f_g_in  = g_in[256*f : 256*(f+1)].reshape(16, 16)

    for r in range(16):
        row_diffs = np.where(f_g_out[r] != f_t_out[r])[0]
        if len(row_diffs) > 0:
            print(f"Row {r:2d} has {len(row_diffs)} diffs at cols {row_diffs.tolist()}:")
            for c in row_diffs[:6]:
                in_val = b2f(f_g_in[r, c])
                g_val  = b2f(f_g_out[r, c])
                t_val  = b2f(f_t_out[r, c])
                print(f"  Col {c:2d}: In={in_val:+.4f}, Gold={g_val:+.4f} (0x{f_g_out[r, c]:04x}), Tensix={t_val:+.4f} (0x{f_t_out[r, c]:04x})")
        else:
            print(f"Row {r:2d}: [MATCH] all 16 cols match bit-exact")
