import struct

def bf_to_f(u):
    return struct.unpack('>f', struct.pack('>HH', u, 0))[0]

def f_to_bf(f):
    u = struct.unpack('>I', struct.pack('>f', f))[0]
    lsb = (u >> 16) & 1
    rounding_bias = 0x7FFF + lsb
    u_rounded = u + rounding_bias
    return (u_rounded >> 16) & 0xFFFF

def to_bf16_f(f):
    return bf_to_f(f_to_bf(f))

def amin_model1(a, b):
    min_v = min(a, b)
    max_v = max(a, b)
    diff = to_bf16_f(max_v - min_v)
    c1 = to_bf16_f(0.69315)
    c2 = to_bf16_f(-0.4000)
    c3 = to_bf16_f(0.65715)
    c4 = to_bf16_f(-0.3280)
    line1 = to_bf16_f(c1 + to_bf16_f(c2 * diff))
    line2 = to_bf16_f(c3 + to_bf16_f(c4 * diff))
    corr = max(max(line1, line2), 0.0)
    return max(to_bf16_f(min_v - corr), 0.0)

def amin_model2(a, b):
    min_v = min(a, b)
    max_v = max(a, b)
    diff = to_bf16_f(max_v - min_v)
    c1 = bf_to_f(16177) # 0x3F31
    c2 = bf_to_f(48844) # 0xBECC
    c3 = bf_to_f(16168) # 0x3F28
    c4 = bf_to_f(48807) # 0xBEA7
    line1 = to_bf16_f(c1 + to_bf16_f(c2 * diff))
    line2 = to_bf16_f(c3 + to_bf16_f(c4 * diff))
    corr = max(max(line1, line2), 0.0)
    return max(to_bf16_f(min_v - corr), 0.0)

diffs = 0
for i in range(1, 1000):
    a = 2.5
    b = 2.5 + i * 0.005
    res1 = amin_model1(a, b)
    res2 = amin_model2(a, b)
    if f_to_bf(res1) != f_to_bf(res2):
        diffs += 1
        if diffs <= 5:
            print(f"Diff at diff={b-a:.3f}: M1=0x{f_to_bf(res1):04X} ({res1}) vs M2=0x{f_to_bf(res2):04X} ({res2})")

print(f"Differences between Model 1 and Model 2: {diffs} / 999")
