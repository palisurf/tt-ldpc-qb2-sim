import itertools
import numpy as np
import struct

def u2f(u):
    return struct.unpack('>f', struct.pack('>I', int(u) << 16))[0]

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

# Target r_mag = 1.890625 or 1.8901367
target_l_new = 0.10986328125
q1 = 2.0
target_r_out = target_l_new - q1 # -1.8901367
print(f"Target r_out: {target_r_out}")

# Check all possible pairs of edges in the tile or face
# What inputs give ams(A, B) around 1.8901?
for a in np.linspace(1.5, 4.0, 51):
    for b in np.linspace(1.5, 4.0, 51):
        v = ams(a, b)
        if abs(v - 1.8901) < 0.02:
            print(f"ams({a:.2f}, {b:.2f}) = {v:.4f}")
