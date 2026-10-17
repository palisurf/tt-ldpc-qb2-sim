import struct

def dump_file(path, label):
    with open(path, "rb") as f:
        raw = f.read()
    data = struct.unpack(f"<{len(raw)//2}H", raw)
    print(f"\n--- {label}: {path} ({len(data)} uint16) ---")
    print("Face 0 Row 0 (first 16 words):")
    for i in range(16):
        print(f"  [{i:2d}] = 0x{data[i]:04x} ({data[i]})")

dump_file("dump/golden_scratch_in.bin", "GOLDEN")
dump_file("dump/tensix_scratch_in.bin", "TENSIX")
