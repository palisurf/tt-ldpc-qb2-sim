import numpy as np
import struct

def bf16_to_fp32(val_u16):
    u32 = int(val_u16) << 16
    return struct.unpack('>f', struct.pack('>I', u32))[0]

g_out = np.fromfile('/home/ttuser/LDPC_QB2_GA/dump/golden_scratch_out.bin', dtype=np.uint16)
t_out = np.fromfile('/home/ttuser/LDPC_QB2_GA/dump/tensix_scratch_out.bin', dtype=np.uint16)

print("=== Face 0 Output Analysis (16 rows x 16 cols) ===")
# Face 0 is words 0..255
g_f0 = g_out[:256].reshape((16, 16))
t_f0 = t_out[:256].reshape((16, 16))

for r in range(16):
    diffs = np.where(g_f0[r] != t_f0[r])[0]
    g_nz = np.count_nonzero(g_f0[r])
    t_nz = np.count_nonzero(t_f0[r])
    if len(diffs) == 0:
        print(f"Face0 Row {r:2d}: 100% BIT-EXACT MATCH (16/16). nz={g_nz}")
    else:
        print(f"Face0 Row {r:2d}: {len(diffs):2d} / 16 MISMATCHES. g_nz={g_nz}, t_nz={t_nz}")
        for c in range(min(4, len(diffs))):
            col = diffs[c]
            print(f"   col {col:2d}: g=0x{g_f0[r,col]:04x} ({bf16_to_fp32(g_f0[r,col]):+.3f}) vs t=0x{t_f0[r,col]:04x} ({bf16_to_fp32(t_f0[r,col]):+.3f})")
