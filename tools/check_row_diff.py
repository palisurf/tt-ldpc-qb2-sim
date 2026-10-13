import numpy as np
import struct

def bf16_to_fp32(val_u16):
    u32 = int(val_u16) << 16
    return struct.unpack('>f', struct.pack('>I', u32))[0]

g = np.fromfile('/home/ttuser/LDPC_QB2_GA/dump/golden_scratch_out.bin', dtype=np.uint16).reshape((32, 32))
t = np.fromfile('/home/ttuser/LDPC_QB2_GA/dump/tensix_scratch_out.bin', dtype=np.uint16).reshape((32, 32))

print("=== Row Values & Diffs ===")
for r in range(32):
    diffs = np.where(g[r] != t[r])[0]
    g_nz = np.count_nonzero(g[r])
    t_nz = np.count_nonzero(t[r])
    if len(diffs) == 0:
        print(f"Row {r:2d}: MATCH. Non-zero count: g={g_nz}, t={t_nz}")
    else:
        print(f"Row {r:2d}: {len(diffs)} / 32 MISMATCHES. g_nz={g_nz}, t_nz={t_nz}. Col 0: g={bf16_to_fp32(g[r,0]):.3f} vs t={bf16_to_fp32(t[r,0]):.3f}")
