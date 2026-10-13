import numpy as np

g_in = np.fromfile('/home/ttuser/LDPC_QB2_GA/dump/golden_scratch_in.bin', dtype=np.uint16)
t_in = np.fromfile('/home/ttuser/LDPC_QB2_GA/dump/tensix_scratch_in.bin', dtype=np.uint16)
print(f"scratch_in diffs: {np.count_nonzero(g_in != t_in)} / {len(g_in)}")

g_out = np.fromfile('/home/ttuser/LDPC_QB2_GA/dump/golden_scratch_out.bin', dtype=np.uint16)
t_out = np.fromfile('/home/ttuser/LDPC_QB2_GA/dump/tensix_scratch_out.bin', dtype=np.uint16)
print(f"scratch_out diffs: {np.count_nonzero(g_out != t_out)} / {len(g_out)}")

# Per-face analysis for scratch_out
for f in range(4):
    fg = g_out[f*256 : (f+1)*256]
    ft = t_out[f*256 : (f+1)*256]
    fdiff = np.count_nonzero(fg != ft)
    print(f"  Face {f}: {fdiff} / 256 diffs")
