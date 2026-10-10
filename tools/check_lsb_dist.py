import numpy as np
import struct

def bf16_to_fp32(val):
    return struct.unpack('>f', struct.pack('>HH', int(val), 0))[0]

g = np.fromfile('/home/ttuser/LDPC_QB2_GA/engines/tile_matrix_4cw/dump/golden_final_vnodes.bin', dtype=np.uint16)
t = np.fromfile('/home/ttuser/LDPC_QB2_GA/engines/tile_matrix_4cw/dump/tensix_final_vnodes.bin', dtype=np.uint16)

diff_lsb = np.abs(g.astype(np.int32) - t.astype(np.int32))
print('Max LSB diff:', np.max(diff_lsb))
print('Distribution of LSB diffs:')
unique, counts = np.unique(diff_lsb, return_counts=True)
for u, c in zip(unique, counts):
    if c > 5 or u <= 5:
        print(f'  {u} LSB difference: {c} words ({c/len(g)*100:.2f}%)')

large_idx = np.where(diff_lsb > 50)[0]
print(f'\nTotal words with raw uint16 diff > 50: {len(large_idx)}')
for idx in large_idx:
    g_val = bf16_to_fp32(g[idx])
    t_val = bf16_to_fp32(t[idx])
    print(f'Idx {idx}: Golden=0x{g[idx]:04x} ({g_val:+.6f}) vs Tensix=0x{t[idx]:04x} ({t_val:+.6f}) float_diff={abs(g_val - t_val):.6f}')
