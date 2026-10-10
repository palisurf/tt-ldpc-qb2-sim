import struct

def bf_to_f(u):
    return struct.unpack('>f', struct.pack('>HH', int(u), 0))[0]

def f_to_bf(f):
    u = struct.unpack('>I', struct.pack('>f', f))[0]
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16) & 0xFFFF

# Let's inspect Variable 0 CW 2:
# q = 5.46875 (0x40AF)
# What was r_out for Variable 0 in Check 1515?
# Tensix produced: 0x40A5 (5.15625)
# Golden produced: 0x40A6 (5.18750)
tens_diff = bf_to_f(0x40A5) - bf_to_f(0x40AF)
gold_diff = bf_to_f(0x40A6) - bf_to_f(0x40AF)
print(f"Tensix r_out = {tens_diff} (0x{f_to_bf(tens_diff):04X})")
print(f"Golden r_out = {gold_diff} (0x{f_to_bf(gold_diff):04X})")
