import numpy as np
import struct

with open('/home/ttuser/LDPC_QB2_GA/dump/tensix_initial_vnodes.bin', 'rb') as f:
    init_v = np.frombuffer(f.read(), dtype=np.uint16)

with open('/home/ttuser/LDPC_QB2_GA/dump/tensix_scratch_in.bin', 'rb') as f:
    tin = np.frombuffer(f.read(), dtype=np.uint16)
with open('/home/ttuser/LDPC_QB2_GA/dump/golden_scratch_in.bin', 'rb') as f:
    gin = np.frombuffer(f.read(), dtype=np.uint16)
with open('/home/ttuser/LDPC_QB2_GA/dump/golden_scratch_out.bin', 'rb') as f:
    gout = np.frombuffer(f.read(), dtype=np.uint16)

print(f"tin == gin matches: {np.count_nonzero(tin == gin)} / {len(tin)}")
for face in range(4):
    gi = gin[face*256:(face+1)*256]
    go = gout[face*256:(face+1)*256]
    print(f"Face {face}: gin == gout matches: {np.count_nonzero(gi == go)} / 256")

# Let's inspect init_v size
print(f"init_v size: {len(init_v)}")
# Variable nodes 2048 to 2559 are punctured (=0)
print(f"init_v non-zeros: {np.count_nonzero(init_v != 0)}")
print(f"init_v[2048*4:] non-zeros: {np.count_nonzero(init_v[2048*4:] != 0)}")
