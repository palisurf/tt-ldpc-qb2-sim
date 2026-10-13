import numpy as np
import struct

def u2f(u):
    return struct.unpack('>f', struct.pack('>I', int(u) << 16))[0]

def f2u(f):
    u = struct.unpack('>I', struct.pack('>f', float(f)))[0]
    return u >> 16

def to_bf16(f):
    u = struct.unpack('>I', struct.pack('>f', float(f)))[0]
    lsb = (u >> 16) & 1
    bias = 0x7FFF + lsb
    u_rounded = (u + bias) & 0xFFFF0000
    return struct.unpack('>f', struct.pack('>I', u_rounded))[0]

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

# In Face 0, Col 8 (which is even column 8):
# Check 0 has its 6 edges at:
# E0: Row 0 (reg 0, lane 4)
# E1: Row 2 (reg 0, lane 20)
# E2: Row 4 (reg 2, lane 4)
# E3: Row 6 (reg 2, lane 20)
# E4: Row 8 (reg 4, lane 4)
# E5: Row 10 (reg 4, lane 20)

f0_in = gin[:256].reshape((16, 16))
f0_gout = gout[:256].reshape((16, 16))
f0_tout = tout[:256].reshape((16, 16))

col = 8
q = [
    u2f(f0_in[0, col]),  # E0
    u2f(f0_in[2, col]),  # E1
    u2f(f0_in[4, col]),  # E2
    u2f(f0_in[6, col]),  # E3
    u2f(f0_in[8, col]),  # E4
    u2f(f0_in[10, col]), # E5
]
print(f"Inputs q: {q}")

m = [abs(x) for x in q]
s = [-1.0 if x < 0 else 1.0 for x in q]
s_tot = 1.0
for x in s: s_tot *= x
print(f"s_tot = {s_tot}")

m01 = ams(m[0], m[1])
m23 = ams(m[2], m[3])
m45 = ams(m[4], m[5])
print(f"m01={m01}, m23={m23}, m45={m45}")

# Edge 0 (lane < 16) uses: ams(ams(m23, m45), m[1])
r_mag0 = ams(ams(m23, m45), m[1])
r_out0 = to_bf16((s_tot * s[0]) * r_mag0)
l_new0 = to_bf16(q[0] + r_out0)

# Edge 1 (lane >= 16) uses: ams(ams(m23, m45), m[0])
r_mag1 = ams(ams(m23, m45), m[0])
r_out1 = to_bf16((s_tot * s[1]) * r_mag1)
l_new1 = to_bf16(q[1] + r_out1)

print(f"\n--- Edge 0 (Row 0, lane 4) ---")
print(f"r_mag0 = {r_mag0}, r_out0 = {r_out0}, l_new0 = {l_new0}")
print(f"Golden Row 0: {u2f(f0_gout[0, col])} (0x{f0_gout[0, col]:04X})")
print(f"Tensix Row 0: {u2f(f0_tout[0, col])} (0x{f0_tout[0, col]:04X})")

print(f"\n--- Edge 1 (Row 2, lane 20) ---")
print(f"r_mag1 = {r_mag1}, r_out1 = {r_out1}, l_new1 = {l_new1}")
print(f"Golden Row 2: {u2f(f0_gout[2, col])} (0x{f0_gout[2, col]:04X})")
print(f"Tensix Row 2: {u2f(f0_tout[2, col])} (0x{f0_tout[2, col]:04X})")
