import struct

with open("dump/tensix_initial_vnodes.bin", "rb") as f:
    raw = f.read()

words = struct.unpack(f"<{len(raw)//2}H", raw)
for i in range(len(words) - 3):
    if words[i] == 0x4018 and words[i+1] == 0x3f03:
        print(f"Match 4018, 3f03 at word {i} (byte 0x{i*2:04x}): next = 0x{words[i+2]:04x}, 0x{words[i+3]:04x}")
    if words[i] == 0x40b3 and words[i+1] == 0x408e:
        print(f"Match 40b3, 408e at word {i} (byte 0x{i*2:04x}): prev = 0x{words[i-2]:04x}, 0x{words[i-1]:04x}")
