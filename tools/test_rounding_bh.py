import struct
import numpy as np

# Round to nearest even (standard IEEE)
def f_to_bf_rne(f):
    u = struct.unpack('>I', struct.pack('>f', f))[0]
    lsb = (u >> 16) & 1
    bias = 0x7FFF + lsb
    return ((u + bias) >> 16) & 0xFFFF

# Round to nearest away from zero (Blackhole Tensix hardware!)
def f_to_bf_rna(f):
    u = struct.unpack('>I', struct.pack('>f', f))[0]
    bias = 0x8000
    return ((u + bias) >> 16) & 0xFFFF

def bf_to_f(u):
    return struct.unpack('>f', struct.pack('>HH', int(u), 0))[0]

print("Testing RNE vs RNA:")
diff_count = 0
for i in range(100000):
    f = 1.0 + i * 0.0001
    u_rne = f_to_bf_rne(f)
    u_rna = f_to_bf_rna(f)
    if u_rne != u_rna:
        diff_count += 1
        if diff_count <= 5:
            print(f"f={f:.6f}: RNE=0x{u_rne:04X} ({bf_to_f(u_rne)}) vs RNA=0x{u_rna:04X} ({bf_to_f(u_rna)}) [diff={u_rna - u_rne}]")

print(f"Total differences in 100,000 floats: {diff_count}")
