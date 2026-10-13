import numpy as np
import struct

def u2f(u):
    return struct.unpack('>f', struct.pack('>I', int(u) << 16))[0]

def f2u(f):
    u = struct.unpack('>I', struct.pack('>f', f))[0]
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

# Face 0 is first 256 words
f0_in = gin[:256].reshape((16, 16))
f0_gout = gout[:256].reshape((16, 16))
f0_tout = tout[:256].reshape((16, 16))

col = 8
print(f"=== Face 0, Col {col} All Rows ===")
for r in range(16):
    in_v = u2f(f0_in[r, col])
    g_v = u2f(f0_gout[r, col])
    t_v = u2f(f0_tout[r, col])
    print(f"Row {r:2d}: In={in_v:8.4f}, Golden={g_v:8.4f} (0x{f0_gout[r, col]:04X}), Tensix={t_v:8.4f} (0x{f0_tout[r, col]:04X}), diff={abs(g_v - t_v):8.4f}")
