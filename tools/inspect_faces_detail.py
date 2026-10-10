import numpy as np
import struct
import os

def b2f(u):
    u32 = int(u) << 16
    return struct.unpack('>f', struct.pack('>I', u32))[0]

dump_dir = "engines/tile_matrix_4cw/dump"
if not os.path.exists(f"{dump_dir}/golden_scratch_in.bin"):
    dump_dir = "/home/ttuser/LDPC_QB2_GA/engines/tile_matrix_4cw/dump"

g_in = np.fromfile(f"{dump_dir}/golden_scratch_in.bin", dtype=np.uint16)
t_in = np.fromfile(f"{dump_dir}/tensix_scratch_in.bin", dtype=np.uint16)
g_out = np.fromfile(f"{dump_dir}/golden_scratch_out.bin", dtype=np.uint16)
t_out = np.fromfile(f"{dump_dir}/tensix_scratch_out.bin", dtype=np.uint16)

print(f"=== Scratch In Differences ===")
in_diff = np.where(g_in != t_in)[0]
print(f"Total in_diff: {len(in_diff)}")

print(f"\n=== Scratch Out Differences per Face ===")
for f in range(4):
    f_words_g_out = g_out[256*f : 256*(f+1)]
    f_words_t_out = t_out[256*f : 256*(f+1)]
    diff_count = np.count_nonzero(f_words_g_out != f_words_t_out)
    print(f"Face {f}: {diff_count} / 256 differ")

print(f"\n=== Inspect Input Values in Face 0, 1, 2, 3 ===")
for f in range(4):
    f_in = g_in[256*f : 256*(f+1)]
    f_in_flt = [b2f(x) for x in f_in]
    num_zeros = sum(1 for x in f_in_flt if x == 0.0)
    min_v = min(abs(x) for x in f_in_flt)
    max_v = max(abs(x) for x in f_in_flt)
    print(f"Face {f} Inputs: num_zeros={num_zeros}, min_abs={min_v:.4f}, max_abs={max_v:.4f}")
