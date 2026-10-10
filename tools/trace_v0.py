import struct

def bf_to_f(u):
    return struct.unpack('>f', struct.pack('>HH', int(u), 0))[0]

def f_to_bf(f):
    u = struct.unpack('>I', struct.pack('>f', f))[0]
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16) & 0xFFFF

# Initial values
init_cw2 = bf_to_f(0x40af) # 5.46875
init_cw3 = bf_to_f(0x40cd) # 6.40625

tensix_cw2 = bf_to_f(0x40a5) # 5.15625
tensix_cw3 = bf_to_f(0x40c3) # 6.09375

golden_cw2 = bf_to_f(0x40a6) # 5.18750
golden_cw3 = bf_to_f(0x40c4) # 6.12500

print(f"CW2 delta Tensix: {tensix_cw2 - init_cw2:.5f} (0x{0x40a5:04x} - 0x{0x40af:04x} = {0x40a5 - 0x40af} LSBs)")
print(f"CW2 delta Golden: {golden_cw2 - init_cw2:.5f} (0x{0x40a6:04x} - 0x{0x40af:04x} = {0x40a6 - 0x40af} LSBs)")
print(f"CW3 delta Tensix: {tensix_cw3 - init_cw3:.5f} (0x{0x40c3:04x} - 0x{0x40cd:04x} = {0x40c3 - 0x40cd} LSBs)")
print(f"CW3 delta Golden: {golden_cw3 - init_cw3:.5f} (0x{0x40c4:04x} - 0x{0x40cd:04x} = {0x40c4 - 0x40cd} LSBs)")
