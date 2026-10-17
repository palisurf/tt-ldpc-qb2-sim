import struct

with open('dump/tensix_initial_vnodes.bin', 'rb') as f:
    data = f.read()

tensix_w = [0x405a, 0x4018, 0x4040, 0x408f]
golden_w = [0x403d, 0xbedc, 0x40a3, 0x408e]

for i in range(0, len(data) - 8, 2):
    w = list(struct.unpack('<4H', data[i:i+8]))
    if w == tensix_w:
        print(f"Tensix words match at byte offset {i} (0x{i:04x})")
    if w == golden_w:
        print(f"Golden words match at byte offset {i} (0x{i:04x})")
