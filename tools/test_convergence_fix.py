import struct
import numpy as np

def bf_to_f(u):
    return struct.unpack('>f', struct.pack('>HH', int(u), 0))[0]

def f_to_bf(f):
    u = struct.unpack('>I', struct.pack('>f', f))[0]
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16) & 0xFFFF

def to_bf16(f):
    return bf_to_f(f_to_bf(f))

# Let's load matrix
check_to_var = {}
M, N = 1536, 2560
with open('matrices/AR4JA_r45_4c_128c_r12.chinn.out') as f:
    for line in f:
        p = line.split()
        if len(p) == 3:
            r, c, val = int(p[0]) - 1, int(p[1]) - 1, float(p[2])
            if val != 0.0:
                check_to_var.setdefault(r, []).append(c)

print(f"Matrix loaded: {len(check_to_var)} checks")

# Let's verify that with the corrected face mapping, checks 0..1535 map cleanly to all 4 faces
# and test decoding convergence!
