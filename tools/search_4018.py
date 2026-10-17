import struct

with open("dump/tensix_initial_vnodes.bin", "rb") as f:
    raw = f.read()

words = struct.unpack(f"<{len(raw)//2}H", raw)
print(f"Total words: {len(words)}")
for i, w in enumerate(words):
    if w == 0x4018:
        context = [f"0x{words[j]:04x}" for j in range(max(0, i-4), min(len(words), i+5))]
        print(f"  Match at word {i} (byte 0x{i*2:04x}): context = {context}")
