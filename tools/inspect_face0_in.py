import numpy as np
import struct

def bf16_to_fp32(val_u16):
    u32 = int(val_u16) << 16
    return struct.unpack('>f', struct.pack('>I', u32))[0]

g_in = np.fromfile('/home/ttuser/LDPC_QB2_GA/dump/golden_scratch_in.bin', dtype=np.uint16)

print("=== Face 0 Rows (each 16 elements = 32 bytes) ===")
# Face 0 is words 0..255 (16 rows x 16 words)
face0 = g_in[:256].reshape((16, 16))
for r in range(16):
    row_vals = [f"{bf16_to_fp32(x):.2f}" for x in face0[r, :8]]
    nz = np.count_nonzero(face0[r])
    print(f"Row {r:2d} (nz={nz:2d}): {row_vals}")
