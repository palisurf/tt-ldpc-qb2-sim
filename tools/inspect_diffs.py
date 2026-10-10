import struct, numpy as np, os

dump_dir = 'dump'
if not os.path.exists(os.path.join(dump_dir, 'golden_final_vnodes.bin')):
    dump_dir = 'engines/tile_matrix_4cw/dump'

with open(os.path.join(dump_dir, 'golden_final_vnodes.bin'), 'rb') as f: gold_b = f.read()
with open(os.path.join(dump_dir, 'tensix_final_vnodes.bin'), 'rb') as f: ten_b = f.read()
with open(os.path.join(dump_dir, 'tensix_initial_vnodes.bin'), 'rb') as f: init_b = f.read()

w_gold = np.frombuffer(gold_b, dtype=np.uint16)
w_ten  = np.frombuffer(ten_b,  dtype=np.uint16)
w_init = np.frombuffer(init_b, dtype=np.uint16)

def bf16_to_fp32(u):
    return struct.unpack('>f', struct.pack('>HH', u, 0))[0]

count = 0
for v in range(2560):
    t = v // 256
    v_loc = v % 256
    row = v_loc // 8
    k = v_loc % 8
    face = (0 if k < 4 else 1) if row < 16 else (2 if k < 4 else 3)
    col_grp = k if k < 4 else (k - 4)
    base_byte = t * 2048 + (face * 512) + ((row & 15) << 5) + (col_grp << 3)
    idx = base_byte // 2

    # Check CW0
    g = w_gold[idx]
    ten = w_ten[idx]
    ini = w_init[idx]

    if g != ten:
        print(f"Var {v:4d} (tile {t:2d}, r {row:2d}, k {k}): Init={bf16_to_fp32(ini):+7.3f} (0x{ini:04x}) | Gold={bf16_to_fp32(g):+7.3f} (0x{g:04x}) | Ten={bf16_to_fp32(ten):+7.3f} (0x{ten:04x}) | Diff={bf16_to_fp32(ten)-bf16_to_fp32(g):+7.3f}")
        count += 1
        if count >= 30:
            break
