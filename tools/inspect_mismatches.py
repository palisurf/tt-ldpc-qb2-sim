import struct

with open('dump/golden_scratch_out.bin', 'rb') as f:
    g = f.read()
with open('dump/tensix_scratch_out.bin', 'rb') as f:
    t = f.read()

def bf16_to_fp32(val_u16):
    u32 = int(val_u16) << 16
    return struct.unpack('>f', struct.pack('>I', u32))[0]

print('=== FACE 2 MISMATCHES ===')
for idx in range(512, 768):
    gw = struct.unpack('<H', g[idx*2:(idx+1)*2])[0]
    tw = struct.unpack('<H', t[idx*2:(idx+1)*2])[0]
    if gw != tw:
        r = (idx - 512) // 16
        c = (idx - 512) % 16
        print(f'Face 2: Row {r:2d}, Col {c:2d} | Golden: 0x{gw:04X} ({bf16_to_fp32(gw):+8.4f}) | Tensix: 0x{tw:04X} ({bf16_to_fp32(tw):+8.4f})')
