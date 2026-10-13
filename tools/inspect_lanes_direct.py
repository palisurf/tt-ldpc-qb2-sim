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

g_in = np.fromfile('dump/golden_scratch_in.bin', dtype=np.uint16)
t_out = np.fromfile('dump/tensix_scratch_out.bin', dtype=np.uint16)

print("=== Scratch In: Reg 0 & Reg 2 ===")
for reg in [0, 2, 4]:
    lanes = [round(u2f(g_in[get_scratch_byte_offset(reg, l) // 2]), 3) for l in range(32)]
    print(f"Reg {reg} (Face 0):")
    print(f"  Lanes  0..15: {lanes[:16]}")
    print(f"  Lanes 16..31: {lanes[16:]}")

print("\n=== Tensix Scratch Out: Reg 0..3 ===")
for reg in range(4):
    lanes = [round(u2f(t_out[get_scratch_byte_offset(reg, l) // 2]), 3) for l in range(32)]
    print(f"Reg {reg}:")
    print(f"  Lanes  0..15: {lanes[:16]}")
    print(f"  Lanes 16..31: {lanes[16:]}")
