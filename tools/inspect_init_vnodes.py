import numpy as np
import struct

def bf16_to_fp32(val_u16):
    u32 = int(val_u16) << 16
    return struct.unpack('>f', struct.pack('>I', u32))[0]

g_init = np.fromfile('/home/ttuser/LDPC_QB2_GA/dump/tensix_initial_vnodes.bin', dtype=np.uint16)
print(f"Total words in initial vnodes: {len(g_init)}")
print(f"Non-zero count in initial vnodes: {np.count_nonzero(g_init)}")
