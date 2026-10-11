import struct

def read_u16(path):
    with open(path, 'rb') as f:
        data = f.read()
    return [struct.unpack('<H', data[i:i+2])[0] for i in range(0, len(data), 2)]

def bf16_to_fp32(val_u16):
    u32 = int(val_u16) << 16
    return struct.unpack('>f', struct.pack('>I', u32))[0]

g_in = read_u16('dump/golden_scratch_in.bin')
t_in = read_u16('dump/tensix_scratch_in.bin')
g_out = read_u16('dump/golden_scratch_out.bin')
t_out = read_u16('dump/tensix_scratch_out.bin')

print('--- FACE 2 (Rows 0..11): Compare Golden Out vs Tensix Out ---')
for r in range(12):
    diffs = 0
    max_d = 0.0
    for c in range(16):
        idx = 512 + r * 16 + c
        gw = g_out[idx]
        tw = t_out[idx]
        if gw != tw:
            diffs += 1
            d = abs(bf16_to_fp32(gw) - bf16_to_fp32(tw))
            if d > max_d: max_d = d
    print(f'Row {r:2d} (Edge {r//2}): {diffs:2d} diffs | max_diff = {max_d:.5f}')
