import numpy as np
import struct

def u2f(u):
    return struct.unpack('>f', struct.pack('>I', int(u) << 16))[0]

with open('/home/ttuser/LDPC_QB2_GA/dump/golden_scratch_out.bin', 'rb') as f:
    g = np.frombuffer(f.read(), dtype=np.uint16)
with open('/home/ttuser/LDPC_QB2_GA/dump/tensix_scratch_out.bin', 'rb') as f:
    t = np.frombuffer(f.read(), dtype=np.uint16)

diff = (g != t)
for face in range(4):
    f_diff = diff[face*256 : (face+1)*256].reshape((16, 16))
    print(f'=== Face {face} Diff Rows ===')
    for r in range(16):
        nd = np.count_nonzero(f_diff[r])
        if nd > 0:
            row_g = g[face*256 + r*16 : face*256 + (r+1)*16]
            row_t = t[face*256 + r*16 : face*256 + (r+1)*16]
            print(f'  Row {r:2d}: {nd:2d} diffs | g0={u2f(row_g[0]):+.3f} (0x{row_g[0]:04x}) vs t0={u2f(row_t[0]):+.3f} (0x{row_t[0]:04x})')
