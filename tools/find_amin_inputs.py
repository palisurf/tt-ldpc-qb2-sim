import struct

def bf_to_f(u):
    return struct.unpack('>f', struct.pack('>HH', int(u), 0))[0]

def f_to_bf(f):
    u = struct.unpack('>I', struct.pack('>f', f))[0]
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16) & 0xFFFF

def to_bf16(f):
    return bf_to_f(f_to_bf(f))

# Let's inspect the two lines in approx_min_star:
# c1 = 0x3F31 (0.69140625)
# c2 = 0xBECC (-0.3984375)
# c3 = 0x3F28 (0.65625)
# c4 = 0xBEA7 (-0.326171875)

# For diff = 1.0:
c1 = bf_to_f(16177)
c2 = bf_to_f(48844)
c3 = bf_to_f(16168)
c4 = bf_to_f(48807)

for diff in [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]:
    diff_bf = to_bf16(diff)
    line1 = to_bf16(to_bf16(c2 * diff_bf) + c1)
    line2 = to_bf16(to_bf16(c4 * diff_bf) + c3)
    corr = max(max(line1, line2), 0.0)
    print(f"diff={diff_bf}: line1={line1:.5f} (0x{f_to_bf(line1):04X}), line2={line2:.5f} (0x{f_to_bf(line2):04X}), corr={corr:.5f}")
