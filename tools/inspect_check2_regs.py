import struct, numpy as np

def bf(u):
    return struct.unpack('>f', struct.pack('>I', int(u) << 16))[0]

gin = np.fromfile('Z:/LDPC_QB2_GA/dump/golden_scratch_in.bin', dtype=np.uint16)

def get_scratch_byte_offset(reg_idx, lane):
    face = reg_idx // 8
    r_in_face = reg_idx % 8
    k = r_in_face // 2
    is_odd_reg = r_in_face % 2
    row_in_quartet = lane // 8
    c_pair = lane % 8
    return face * 512 + (4 * k + row_in_quartet) * 32 + (2 * c_pair + is_odd_reg) * 2

print("=== Check 2 (Lanes 4 and 20) across all 3 registers (Reg 0, 2, 4) ===")
for r_idx in [0, 2, 4]:
    off4 = get_scratch_byte_offset(r_idx, 4) // 2
    off20 = get_scratch_byte_offset(r_idx, 20) // 2
    v4 = bf(gin[off4])
    v20 = bf(gin[off20])
    s4 = 1.0 if v4 >= 0 else -1.0
    s20 = 1.0 if v20 >= 0 else -1.0
    print(f"Reg {r_idx}: Lane 4 = {v4:+8.4f} (sign {s4:+2.0f}) | Lane 20 = {v20:+8.4f} (sign {s20:+2.0f}) | Product = {s4*s20:+2.0f}")
