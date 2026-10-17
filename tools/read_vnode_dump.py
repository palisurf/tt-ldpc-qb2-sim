import struct

with open("dump/tensix_initial_vnodes.bin", "rb") as f:
    raw = f.read()

off = 0x10f8
print(f"Offset 0x{off:04x} in dump/tensix_initial_vnodes.bin:")
w = struct.unpack_from("<4H", raw, off)
print("  Hex:", [f"0x{x:04x}" for x in w])
print("  Raw uint32:", [f"0x{x:08x}" for x in struct.unpack_from("<2I", raw, off)])

off_v43 = 0x00b8
print(f"\nOffset 0x{off_v43:04x} in dump/tensix_initial_vnodes.bin:")
w43 = struct.unpack_from("<4H", raw, off_v43)
print("  Hex:", [f"0x{x:04x}" for x in w43])
