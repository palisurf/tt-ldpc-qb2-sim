import struct

for fname in ["tensix_final_vnodes.bin", "golden_final_vnodes.bin"]:
    with open(f"dump/{fname}", "rb") as f:
        raw = f.read()

    target_w0 = 0x3f034018
    target_w1 = 0x408e40b3

    print(f"\nSearching for w0=0x{target_w0:08x}, w1=0x{target_w1:08x} in dump/{fname}...")
    found = False
    for i in range(0, len(raw) - 8, 4):
        w0, w1 = struct.unpack_from("<2I", raw, i)
        if w0 == target_w0 and w1 == target_w1:
            print(f"  EXACT MATCH found at byte offset 0x{i:04x} ({i})!")
            found = True
        elif w0 == target_w0:
            print(f"  w0 match at byte offset 0x{i:04x} ({i}), w1=0x{w1:08x}")
    if not found:
        print(f"No exact match in dump/{fname}!")
