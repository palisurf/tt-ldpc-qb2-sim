import struct

def read_u16(path):
    with open(path, 'rb') as f: data = f.read()
    return [struct.unpack('<H', data[i:i+2])[0] for i in range(0, len(data), 2)]

def bf16_to_fp32(val_u16):
    u32 = int(val_u16) << 16
    return struct.unpack('>f', struct.pack('>I', u32))[0]

def fp32_to_bf16_val(f_val):
    u32 = struct.unpack('>I', struct.pack('>f', f_val))[0]
    return u32 >> 16

def get_scratch_byte_offset(reg_idx, lane):
    f = reg_idx // 8
    r_face = reg_idx % 8
    row_in_face = (2 * r_face) if (lane < 16) else (2 * r_face + 1)
    col_in_face = lane if (lane < 16) else (lane - 16)
    return f * 512 + row_in_face * 32 + col_in_face * 2

t_in = read_u16('dump/tensix_scratch_in.bin')
t_out = read_u16('dump/tensix_scratch_out.bin')

# Let's inspect pass = 2 (Face 2), lane = 0
base_r = 16
for lane in range(4):
    q_vals = []
    for r in range(6):
        off = get_scratch_byte_offset(base_r + r, lane)
        val = bf16_to_fp32(t_in[off // 2])
        q_vals.append(val)
    print(f'Lane {lane} input Q[0..5]:', [f'{v:+6.2f}' for v in q_vals])
