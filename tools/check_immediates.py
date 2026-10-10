import struct

def bf_to_f(u):
    b = struct.pack(">HH", u, 0)
    return struct.unpack(">f", b)[0]

print("48844 (0x{:04X}) -> {}".format(48844, bf_to_f(48844)))
print("16177 (0x{:04X}) -> {}".format(16177, bf_to_f(16177)))
print("48807 (0x{:04X}) -> {}".format(48807, bf_to_f(48807)))
print("16168 (0x{:04X}) -> {}".format(16168, bf_to_f(16168)))

# Compare with our constants in C++
# c_0_69315 = fp32_to_bf16(0.69315f)
f = 0.69315
u = struct.unpack(">I", struct.pack(">f", f))[0] >> 16
print(f"0.69315f -> 0x{u:04X} ({bf_to_f(u)})")

f = -0.4000
u = struct.unpack(">I", struct.pack(">f", f))[0] >> 16
print(f"-0.4000f -> 0x{u:04X} ({bf_to_f(u)})")

f = 0.65715
u = struct.unpack(">I", struct.pack(">f", f))[0] >> 16
print(f"0.65715f -> 0x{u:04X} ({bf_to_f(u)})")

f = -0.3280
u = struct.unpack(">I", struct.pack(">f", f))[0] >> 16
print(f"-0.3280f -> 0x{u:04X} ({bf_to_f(u)})")
