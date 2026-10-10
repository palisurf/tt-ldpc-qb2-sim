with open('dump/tensix_initial_vnodes.bin', 'rb') as f: init_b = f.read()

import numpy as np
w_init = np.frombuffer(init_b, dtype=np.uint16)

def get_v_offset(v):
    t = v // 256
    v_loc = v % 256
    row = v_loc // 8
    k = v_loc % 8
    face = (0 if k < 4 else 1) if row < 16 else (2 if k < 4 else 3)
    col_grp = k if k < 4 else (k - 4)
    return t * 2048 + (face * 512) + ((row & 15) << 5) + (col_grp << 3)

off_2497 = get_v_offset(2497) // 2
print(f"Var 2497 uint16 offset: {off_2497}")
print(f"Var 2497 values (4 CWs): {[hex(w_init[off_2497 + cw]) for cw in range(4)]}")
