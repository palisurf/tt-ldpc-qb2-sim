import numpy as np

g = np.fromfile('/home/ttuser/LDPC_QB2_GA/engines/tile_matrix_4cw/dump/golden_scratch_out.bin', dtype=np.uint16)
t = np.fromfile('/home/ttuser/LDPC_QB2_GA/engines/tile_matrix_4cw/dump/tensix_scratch_out.bin', dtype=np.uint16)
m = np.where(g != t)[0]
rows = m // 32
cols = m % 32
print('Total mismatches:', len(m))
print('Unique mismatched rows:', np.unique(rows))
for r in np.unique(rows):
    c_in_r = cols[rows == r]
    print(f'Row {r}: {len(c_in_r)} mismatches in cols: {c_in_r}')
