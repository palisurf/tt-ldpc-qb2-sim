import numpy as np
import struct

def bf16_to_fp32(val_u16):
    return struct.unpack('>f', struct.pack('>I', int(val_u16) << 16))[0]

tin = np.fromfile('dump/tensix_scratch_in.bin', dtype=np.uint16)
gin = np.fromfile('dump/golden_scratch_in.bin', dtype=np.uint16)
print(f"Scratch IN mismatch count: {np.sum(tin != gin)} out of {len(tin)}")

tout = np.fromfile('dump/tensix_scratch_out.bin', dtype=np.uint16)
gout = np.fromfile('dump/golden_scratch_out.bin', dtype=np.uint16)
print(f"Scratch OUT mismatch count: {np.sum(tout != gout)} out of {len(tout)}")

if np.sum(tout != gout) > 0:
    diff_idx = np.where(tout != gout)[0]
    print(f"First 20 mismatches in scratch OUT (Tile 0, Iter 0):")
    for idx in diff_idx[:20]:
        t_val = bf16_to_fp32(tout[idx])
        g_val = bf16_to_fp32(gout[idx])
        print(f"  idx {idx:4d} (0x{idx*2:04x}): Tensix=0x{tout[idx]:04x} ({t_val:+8.4f}) | Gold=0x{gout[idx]:04x} ({g_val:+8.4f}) | diff={abs(t_val - g_val):.4f}")
