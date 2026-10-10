import struct

f_in = open('engines/tile_matrix_4cw/dump/tensix_initial_vnodes.bin', 'rb').read()
f_tensix = open('engines/tile_matrix_4cw/dump/tensix_final_vnodes.bin', 'rb').read()
f_golden = open('dump/golden_lockstep_final_vnodes.bin', 'rb').read()

def get_f(b, idx):
    u = struct.unpack('<H', b[idx*2:idx*2+2])[0]
    return struct.unpack('>f', struct.pack('>I', u << 16))[0], hex(u)

for v in range(10):
    for cw in range(4):
        idx = v * 4 + cw
        init_val, init_hex = get_f(f_in, idx)
        tensix_val, tensix_hex = get_f(f_tensix, idx)
        golden_val, golden_hex = get_f(f_golden, idx)
        match = "MATCH" if tensix_hex == golden_hex else "DIFF "
        print(f"[{match}] Var {v:2d} CW{cw}: init={init_val:+.4f} | golden={golden_val:+.4f} ({golden_hex}) | tensix={tensix_val:+.4f} ({tensix_hex})")
