import struct

def bf_to_f(u):
    return struct.unpack('>f', struct.pack('>HH', int(u), 0))[0]

def f_to_bf(f):
    u = struct.unpack('>I', struct.pack('>f', f))[0]
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16) & 0xFFFF

def to_bf16(f):
    return bf_to_f(f_to_bf(f))

def amin_stepper(a, b):
    min_v = min(a, b)
    max_v = max(a, b)
    diff  = to_bf16(max_v - min_v)
    c1 = bf_to_f(16177)
    c2 = bf_to_f(48844)
    c3 = bf_to_f(16168)
    c4 = bf_to_f(48807)
    line1 = to_bf16(to_bf16(c2 * diff) + c1)
    line2 = to_bf16(to_bf16(c4 * diff) + c3)
    corr = max(max(line1, line2), 0.0)
    return max(to_bf16(min_v - corr), 0.0)

def amin_sfpu_internal(a, b):
    min_v = min(a, b)
    max_v = max(a, b)
    diff  = max_v - min_v
    c1 = bf_to_f(16177)
    c2 = bf_to_f(48844)
    c3 = bf_to_f(16168)
    c4 = bf_to_f(48807)
    line1 = c1 + c2 * diff
    line2 = c3 + c4 * diff
    corr = max(max(line1, line2), 0.0)
    return max(min_v - corr, 0.0)

q = [3.5, 4.2, 2.8, 5.1, 3.9, 4.8]
mags = [abs(x) for x in q]
min_mag = min(mags)
min_idx = mags.index(min_mag)

m_others_a = 1000.0 if min_idx == 0 else mags[0]
m_others_b = 1000.0 if min_idx == 0 else mags[0]

for r in range(1, 6):
    m = 1000.0 if min_idx == r else mags[r]
    m_others_a = amin_stepper(m_others_a, m)
    m_others_b = amin_sfpu_internal(m_others_b, m)

m_all_a = amin_stepper(m_others_a, min_mag)
m_all_b = amin_sfpu_internal(m_others_b, min_mag)

print(f"Model A: m_others=0x{f_to_bf(m_others_a):04X} ({m_others_a:.5f}), m_all=0x{f_to_bf(m_all_a):04X} ({m_all_a:.5f})")
print(f"Model B: m_others=0x{f_to_bf(m_others_b):04X} ({m_others_b:.5f}), m_all=0x{f_to_bf(m_all_b):04X} ({m_all_b:.5f})")
print(f"Difference m_others: {f_to_bf(m_others_a) - f_to_bf(m_others_b)} LSBs")
print(f"Difference m_all:    {f_to_bf(m_all_a) - f_to_bf(m_all_b)} LSBs")
