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

print('--- Edge 1 (Row 2, cols 0..7) ---')
for c in range(8):
    idx = 512 + 2 * 16 + c
    vin = bf16_to_fp32(g_in[idx])
    vg_out = bf16_to_fp32(g_out[idx])
    vt_out = bf16_to_fp32(t_out[idx])
    print(f'Col {c}: In = {vin:+8.4f} | GoldenOut = {vg_out:+8.4f} | TensixOut = {vt_out:+8.4f}')
