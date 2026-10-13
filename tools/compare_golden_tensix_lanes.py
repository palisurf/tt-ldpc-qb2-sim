import numpy as np
import struct

def u2f(u):
    return struct.unpack('>f', struct.pack('>I', int(u) << 16))[0]

def get_scratch_byte_offset(reg_idx, lane):
    face = reg_idx // 8
    r_in_face = reg_idx % 8
    k = r_in_face // 2
    is_odd_reg = r_in_face % 2
    row_in_quartet = lane // 8
    c_pair = lane % 8
    row_in_face = 4 * k + row_in_quartet
    col_in_face = 2 * c_pair + is_odd_reg
    return face * 512 + row_in_face * 32 + col_in_face * 2

g_out = np.fromfile('dump/golden_scratch_out.bin', dtype=np.uint16)
t_out = np.fromfile('dump/tensix_scratch_out.bin', dtype=np.uint16)

for f in range(4):
    print(f"\n=== Face {f}: Golden vs Tensix Scratch Out per Register ===")
    for r in range(6):
        reg = f * 8 + r
        g_lanes = [round(u2f(g_out[get_scratch_byte_offset(reg, l) // 2]), 4) for l in range(32)]
        t_lanes = [round(u2f(t_out[get_scratch_byte_offset(reg, l) // 2]), 4) for l in range(32)]
        diffs = [abs(g - t) for g, t in zip(g_lanes, t_lanes)]
        max_d = max(diffs)
        mismatches = sum(1 for d in diffs if d > 0.035)
        print(f"Face {f} Reg {r} (reg_idx={reg}): Mismatches (>1 LSB) = {mismatches}/32, Max Diff = {max_d:.4f}")
        if mismatches > 0:
            print(f"  Golden 0..15 : {g_lanes[:16]}")
            print(f"  Tensix 0..15 : {t_lanes[:16]}")
            print(f"  Golden 16..31: {g_lanes[16:]}")
            print(f"  Tensix 16..31: {t_lanes[16:]}")

