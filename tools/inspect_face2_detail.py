import numpy as np
import struct

def u2f(u):
    return struct.unpack('>f', struct.pack('>I', int(u) << 16))[0]

with open('/home/ttuser/LDPC_QB2_GA/dump/golden_scratch_in.bin', 'rb') as f:
    gin = np.frombuffer(f.read(), dtype=np.uint16)
with open('/home/ttuser/LDPC_QB2_GA/dump/golden_scratch_out.bin', 'rb') as f:
    gout = np.frombuffer(f.read(), dtype=np.uint16)
with open('/home/ttuser/LDPC_QB2_GA/dump/tensix_scratch_out.bin', 'rb') as f:
    tout = np.frombuffer(f.read(), dtype=np.uint16)

print("=== Face 2 Detailed Inspection ===")
f_gin = gin[2*256 : 3*256].reshape((16, 16))
f_gout = gout[2*256 : 3*256].reshape((16, 16))
f_tout = tout[2*256 : 3*256].reshape((16, 16))

for r in range(16):
    diffs = np.count_nonzero(f_gout[r] != f_tout[r])
    print(f"Row {r:2d} (diffs={diffs:2d}):")
    print(f"  in   : {[f'{u2f(x):+.2f}' for x in f_gin[r, :8]]}")
    print(f"  gout : {[f'{u2f(x):+.2f}' for x in f_gout[r, :8]]}")
    print(f"  tout : {[f'{u2f(x):+.2f}' for x in f_tout[r, :8]]}")
