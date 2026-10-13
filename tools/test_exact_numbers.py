import numpy as np
import struct

def u2f(u):
    return struct.unpack('>f', struct.pack('>I', int(u) << 16))[0]

def f2u(f):
    u = struct.unpack('>I', struct.pack('>f', f))[0]
    return u >> 16

def to_bf16(f):
    return u2f(f2u(f))

def approx_min_star(a, b):
    min_v = min(a, b)
    max_v = max(a, b)
    diff = max_v - min_v

    c_0_69315 = u2f(16177)
    c_neg_0_4 = u2f(48844)
    c_0_65715 = u2f(16168)
    c_neg_0_328 = u2f(48807)

    line1 = c_0_69315 + c_neg_0_4 * diff
    line2 = c_0_65715 + c_neg_0_328 * diff
    corr = max(max(line1, line2), 0.0)
    return max(min_v - corr, 0.0)

with open('/home/ttuser/LDPC_QB2_GA/dump/golden_scratch_in.bin', 'rb') as f:
    gin = np.frombuffer(f.read(), dtype=np.uint16)
with open('/home/ttuser/LDPC_QB2_GA/dump/golden_scratch_out.bin', 'rb') as f:
    gout = np.frombuffer(f.read(), dtype=np.uint16)
with open('/home/ttuser/LDPC_QB2_GA/dump/tensix_scratch_out.bin', 'rb') as f:
    tout = np.frombuffer(f.read(), dtype=np.uint16)

# Face 2
f2_in = gin[2*256:3*256].reshape((16,16))
f2_gout = gout[2*256:3*256].reshape((16,16))
f2_tout = tout[2*256:3*256].reshape((16,16))

# Let's inspect Row 0, Col 0
# Row 0, 2, 4, 6, 8, 10
q0 = u2f(f2_in[0, 0])
q0_s = u2f(f2_in[2, 0])
q1 = u2f(f2_in[4, 0])
q1_s = u2f(f2_in[6, 0])
q2 = u2f(f2_in[8, 0])
q2_s = u2f(f2_in[10, 0])

print(f"Inputs: q0={q0}, q0_s={q0_s}, q1={q1}, q1_s={q1_s}, q2={q2}, q2_s={q2_s}")

m0, m0_s = abs(q0), abs(q0_s)
m1, m1_s = abs(q1), abs(q1_s)
m2, m2_s = abs(q2), abs(q2_s)

s0 = -1.0 if q0 < 0 else 1.0
s1 = -1.0 if q1 < 0 else 1.0
s2 = -1.0 if q2 < 0 else 1.0
s0_s = -1.0 if q0_s < 0 else 1.0
s1_s = -1.0 if q1_s < 0 else 1.0
s2_s = -1.0 if q2_s < 0 else 1.0
s_tot = (s0 * s0_s) * (s1 * s1_s) * (s2 * s2_s)

m01 = approx_min_star(m0, m0_s)
m23 = approx_min_star(m1, m1_s)
m45 = approx_min_star(m2, m2_s)

r_mag_0 = approx_min_star(approx_min_star(m23, m45), m0_s)
r_out_0 = to_bf16((s_tot * s0) * r_mag_0)
l_new_0 = to_bf16(q0 + r_out_0)

print(f"Computed l_new_0: {l_new_0} (hex: {f2u(l_new_0):04x})")
print(f"Golden gout[0,0]: {u2f(f2_gout[0,0])} (hex: {f2_gout[0,0]:04x})")
print(f"Tensix tout[0,0]: {u2f(f2_tout[0,0])} (hex: {f2_tout[0,0]:04x})")
