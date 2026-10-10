import struct

def bf_to_f(u):
    return struct.unpack('>f', struct.pack('>HH', int(u), 0))[0]

def f_to_bf_rne(f):
    u = struct.unpack('>I', struct.pack('>f', f))[0]
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16) & 0xFFFF

def f_to_bf_trunc(f):
    u = struct.unpack('>I', struct.pack('>f', f))[0]
    return (u >> 16) & 0xFFFF

# Let's test on the 20 mismatch variables:
# In the golden stepper, let's see what the unquantized float l_new value was!
# For instance, Var 0 CW2: Golden had 0x40A6, Tensix had 0x40A5
# 0x40A6 as float is 5.1875 (u = 0x40A60000)
# 0x40A5 as float is 5.15625 (u = 0x40A50000)
# If the true unquantized float was, say, 5.180000:
# In float hex:
f_test = 5.180000
u_test = struct.unpack('>I', struct.pack('>f', f_test))[0]
print(f"Float {f_test} (0x{u_test:08X}):")
print(f"  RNE   = 0x{f_to_bf_rne(f_test):04X}")
print(f"  Trunc = 0x{f_to_bf_trunc(f_test):04X}")
