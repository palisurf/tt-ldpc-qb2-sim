import struct
with open("dump/tensix_initial_vnodes.bin", "rb") as f:
    raw = f.read()
words = struct.unpack(f"<{len(raw)//2}H", raw)
for i in range(16):
    print(f"  vnode_word[{i}] = 0x{words[i]:04x} (offset 0x{i*2:04x})")
