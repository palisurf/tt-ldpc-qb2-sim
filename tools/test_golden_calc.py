import numpy as np
import struct

def bf16_to_fp32(val_u16):
    u32 = int(val_u16) << 16
    return struct.unpack('>f', struct.pack('>I', u32))[0]

def fp32_to_bf16(val_f):
    u32 = struct.unpack('>I', struct.pack('>f', val_f))[0]
    return np.uint16(u32 >> 16)

def to_bf16(val_f):
    return bf16_to_fp32(fp32_to_bf16(val_f))

def approx_min_star(a, b):
    min_v = min(a, b)
    max_v = max(a, b)
    diff = max_v - min_v
    line1 = 0.69315 - 0.40 * diff
    line2 = 0.65715 - 0.328 * diff
    corr = max(line1, line2, 0.0)
    return max(min_v - corr, 0.0)

# Load scratch_in
g_in = np.fromfile('/home/ttuser/LDPC_QB2_GA/dump/golden_scratch_in.bin', dtype=np.uint16)

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

p0, p1, p2 = 0, 2, 4
print("--- Golden Calculation for Face 0, Sub 0 ---")
for lane in [0, 8, 16, 24]:
    s_i = (lane + 16) if lane < 16 else (lane - 16)
    off0 = get_scratch_byte_offset(p0, lane)
    off1 = get_scratch_byte_offset(p1, lane)
    off2 = get_scratch_byte_offset(p2, lane)
    
    l0 = bf16_to_fp32(g_in[off0 // 2])
    l1 = bf16_to_fp32(g_in[off1 // 2])
    l2 = bf16_to_fp32(g_in[off2 // 2])
    
    off0_s = get_scratch_byte_offset(p0, s_i)
    off1_s = get_scratch_byte_offset(p1, s_i)
    off2_s = get_scratch_byte_offset(p2, s_i)
    
    l0_s = bf16_to_fp32(g_in[off0_s // 2])
    l1_s = bf16_to_fp32(g_in[off1_s // 2])
    l2_s = bf16_to_fp32(g_in[off2_s // 2])
    
    print(f"Lane {lane} (s_i={s_i}):")
    print(f"  l0={l0:.4f}, l0_s={l0_s:.4f}")
    print(f"  l1={l1:.4f}, l1_s={l1_s:.4f}")
    print(f"  l2={l2:.4f}, l2_s={l2_s:.4f}")
    
    m0, m0_s = abs(l0), abs(l0_s)
    m1, m1_s = abs(l1), abs(l1_s)
    m2, m2_s = abs(l2), abs(l2_s)
    
    m01 = approx_min_star(m0, m0_s)
    m23 = approx_min_star(m1, m1_s)
    m45 = approx_min_star(m2, m2_s)
    
    r_mag0 = approx_min_star(approx_min_star(m23, m45), m0_s)
    print(f"  m01={m01:.4f}, m23={m23:.4f}, m45={m45:.4f} -> r_mag0={r_mag0:.4f}")
