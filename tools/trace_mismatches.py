import struct

def bf_to_f(u):
    return struct.unpack('>f', struct.pack('>HH', int(u), 0))[0]

def f_to_bf(f):
    u = struct.unpack('>I', struct.pack('>f', f))[0]
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16) & 0xFFFF

def to_bf16(f):
    return bf_to_f(f_to_bf(f))

# Let's inspect the first 100 mismatches and see which variable nodes they are!
tensix_final = open('dump/tensix_final_vnodes.bin', 'rb').read()
golden_final = open('dump/golden_final_vnodes.bin', 'rb').read()

print("Listing first 20 mismatches with their exact variable, codeword, and float values:")
count = 0
for v in range(2560):
    for cw in range(4):
        # find byte offset of v, cw
        t = v // 256
        v_loc = v % 256
        row = v_loc // 8
        k = v_loc % 8
        face = 0 if row < 16 and k < 4 else (1 if row < 16 and k >= 4 else (2 if row >= 16 and k < 4 else 3))
        col_grp = k if k < 4 else k - 4
        off = t * 2048 + face * 512 + ((row & 15) << 5) + (col_grp << 3) + cw * 2
        
        g_u16 = struct.unpack('<H', golden_final[off:off+2])[0]
        t_u16 = struct.unpack('<H', tensix_final[off:off+2])[0]
        if g_u16 != t_u16:
            diff_lsb = g_u16 - t_u16
            print(f"Var {v:4d} CW{cw}: Golden=0x{g_u16:04X} ({bf_to_f(g_u16):.5f}) vs Tensix=0x{t_u16:04X} ({bf_to_f(t_u16):.5f}) | diff={diff_lsb:+d} LSBs")
            count += 1
            if count >= 20: break
    if count >= 20: break
