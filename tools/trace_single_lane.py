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

c_0_69315   = u2f(16177)
c_neg_0_4   = u2f(48844)
c_0_65715   = u2f(16168)
c_neg_0_328 = u2f(48807)

def ams(a, b):
    min_v = min(a, b)
    max_v = max(a, b)
    diff = max_v - min_v
    line1 = c_0_69315 + c_neg_0_4 * diff
    line2 = c_0_65715 + c_neg_0_328 * diff
    corr = max(max(line1, line2), 0.0)
    return max(min_v - corr, 0.0)

gin = np.fromfile('dump/golden_scratch_in.bin', dtype=np.uint16)
gout = np.fromfile('dump/golden_scratch_out.bin', dtype=np.uint16)
tout = np.fromfile('dump/tensix_scratch_out.bin', dtype=np.uint16)

def trace_check(lane, sub=0):
    p0 = 0 + sub
    p1 = 2 + sub
    p2 = 4 + sub
    s_i = lane + 16 if lane < 16 else lane - 16

    q0 = u2f(gin[get_scratch_byte_offset(p0, lane) // 2])
    q1 = u2f(gin[get_scratch_byte_offset(p1, lane) // 2])
    q2 = u2f(gin[get_scratch_byte_offset(p2, lane) // 2])

    q0_sw = u2f(gin[get_scratch_byte_offset(p0, s_i) // 2])
    q1_sw = u2f(gin[get_scratch_byte_offset(p1, s_i) // 2])
    q2_sw = u2f(gin[get_scratch_byte_offset(p2, s_i) // 2])

    print(f"=== TRACE Lane {lane} (paired with {s_i}) ===")
    print(f"q0 = {q0:+.4f}, q0_swap = {q0_sw:+.4f}")
    print(f"q1 = {q1:+.4f}, q1_swap = {q1_sw:+.4f}")
    print(f"q2 = {q2:+.4f}, q2_swap = {q2_sw:+.4f}")

    s0 = -1.0 if q0 < 0 else 1.0
    s1 = -1.0 if q1 < 0 else 1.0
    s2 = -1.0 if q2 < 0 else 1.0
    s0_sw = -1.0 if q0_sw < 0 else 1.0
    s1_sw = -1.0 if q1_sw < 0 else 1.0
    s2_sw = -1.0 if q2_sw < 0 else 1.0

    s_tot = (s0 * s0_sw) * (s1 * s1_sw) * (s2 * s2_sw)
    print(f"Signs: s0={s0}, s0_sw={s0_sw}, s1={s1}, s1_sw={s1_sw}, s2={s2}, s2_sw={s2_sw} => s_tot={s_tot}")

    m0, m0_sw = abs(q0), abs(q0_sw)
    m1, m1_sw = abs(q1), abs(q1_sw)
    m2, m2_sw = abs(q2), abs(q2_sw)

    m01 = ams(m0, m0_sw)
    m23 = ams(m1, m1_sw)
    m45 = ams(m2, m2_sw)
    print(f"Mins: m01={m01:.4f}, m23={m23:.4f}, m45={m45:.4f}")

    m_oth0 = ams(m23, m45)
    rmag0 = ams(m_oth0, m0_sw)
    rout0 = (s_tot * s0) * rmag0
    lnew0 = q0 + rout0

    g_lnew0 = u2f(gout[get_scratch_byte_offset(p0, lane) // 2])
    t_lnew0 = u2f(tout[get_scratch_byte_offset(p0, lane) // 2])
    print(f"Edge 0: Golden Calc = {lnew0:+.4f} (r_out={rout0:+.4f}), File Golden = {g_lnew0:+.4f}, File Tensix = {t_lnew0:+.4f}")

for l in [16, 18, 20, 22]:
    trace_check(l)
    print()
