import numpy as np

with open('/home/ttuser/LDPC_QB2_GA/dump/golden_scratch_in.bin', 'rb') as f:
    gin = np.frombuffer(f.read(), dtype=np.uint16)
with open('/home/ttuser/LDPC_QB2_GA/dump/golden_scratch_out.bin', 'rb') as f:
    gout = np.frombuffer(f.read(), dtype=np.uint16)
with open('/home/ttuser/LDPC_QB2_GA/dump/tensix_scratch_out.bin', 'rb') as f:
    tout = np.frombuffer(f.read(), dtype=np.uint16)

for face in range(4):
    gi = gin[face*256:(face+1)*256].reshape((16,16))
    g = gout[face*256:(face+1)*256].reshape((16,16))
    t = tout[face*256:(face+1)*256].reshape((16,16))
    diffs_per_row = [int(np.count_nonzero(g[r] != t[r])) for r in range(16)]
    gin_nz = [int(np.count_nonzero(gi[r] != 0)) for r in range(16)]
    print(f'Face {face}: total_diffs={np.count_nonzero(g != t):3d} | diffs={diffs_per_row} | in_nz={gin_nz}')
