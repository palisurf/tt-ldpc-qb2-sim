import numpy as np
import struct

def u2f(u):
    return struct.unpack('>f', struct.pack('>I', int(u) << 16))[0]

t_out = np.fromfile('/home/ttuser/LDPC_QB2_GA/dump/tensix_scratch_out.bin', dtype=np.uint16)
f0 = t_out[:256].reshape((16, 16))

print("=== Tensix Output for subvec_transp Probe ===")
# Remember:
# dst_reg[32 + 0] is Reg 0 (Rows 0..3 even cols)
# dst_reg[32 + 1] is Reg 1 (Rows 0..3 odd cols)
# dst_reg[32 + 2] is Reg 2 (Rows 4..7 even cols)
# dst_reg[32 + 3] is Reg 3 (Rows 4..7 odd cols)

# In each register:
# lanes 0..7: row 4k + 0, even cols
# lanes 8..15: row 4k + 1, even cols
# lanes 16..23: row 4k + 2, even cols
# lanes 24..31: row 4k + 3, even cols

# Let's inspect the rows of Face 0:
for r in range(8):
    vals = [u2f(x) for x in f0[r, :8]]
    print(f"Row {r}: {vals}")
