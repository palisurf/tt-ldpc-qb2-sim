import numpy as np
import struct

def u2f(u):
    return struct.unpack('>f', struct.pack('>I', int(u) << 16))[0]

g_in = np.fromfile('/home/ttuser/LDPC_QB2_GA/dump/golden_scratch_in.bin', dtype=np.uint16)
t_out = np.fromfile('/home/ttuser/LDPC_QB2_GA/dump/tensix_scratch_out.bin', dtype=np.uint16)

# Face 0 is words 0..255 (16 rows x 16 cols)
f0_in = g_in[:256].reshape((16, 16))
f0_out = t_out[:256].reshape((16, 16))

print("=== Face 0: Input vs Output of swap_subvec16 ===")
for r in range(8):
    in_c0 = f"{u2f(f0_in[r, 0]):+.3f}"
    out_c0 = f"{u2f(f0_out[r, 0]):+.3f}"
    # find where in_c0 ended up in f0_out
    matched_row = -1
    for r2 in range(16):
        if f0_out[r2, 0] == f0_in[r, 0]:
            matched_row = r2
            break
    print(f"Row {r:2d} in (Col 0 = {in_c0:>7s}) -> Out (Col 0 = {out_c0:>7s}) | In went to Out Row: {matched_row}")
