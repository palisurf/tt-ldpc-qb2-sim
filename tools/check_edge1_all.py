import struct

def read_u16(path):
    with open(path, 'rb') as f:
        data = f.read()
    return [struct.unpack('<H', data[i:i+2])[0] for i in range(0, len(data), 2)]

def bf16_to_fp32(val_u16):
    u32 = int(val_u16) << 16
    return struct.unpack('>f', struct.pack('>I', u32))[0]

g_in = read_u16('dump/golden_scratch_in.bin')
g_out = read_u16('dump/golden_scratch_out.bin')
t_out = read_u16('dump/tensix_scratch_out.bin')

print('--- Edge 1: Row 2 and Row 3: check if In == GoldenOut == TensixOut ---')
diff_in_gout = 0
diff_in_tout = 0
for r in [6, 7]:
    for c in range(16):
        idx = 512 + r * 16 + c
        vin = g_in[idx]
        vg_out = g_out[idx]
        vt_out = t_out[idx]
        if vin != vg_out: diff_in_gout += 1
        if vin != vt_out: diff_in_tout += 1
print(f'Diffs between In and GoldenOut: {diff_in_gout} / 32')
print(f'Diffs between In and TensixOut: {diff_in_tout} / 32')
