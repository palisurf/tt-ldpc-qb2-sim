import struct
import numpy as np

def check_dump(path):
    data = open(path, "rb").read()[:20480]
    errs = [0, 0, 0, 0]
    for v in range(1024):
        t = v // 256
        v_loc = v % 256
        row = v_loc // 8
        k = v_loc % 8
        face = (0 if k < 4 else 1) if row < 16 else (2 if k < 4 else 3)
        col_grp = k if k < 4 else (k - 4)
        off = t * 2048 + face * 512 + (row & 15) * 32 + col_grp * 8
        words = np.frombuffer(data[off:off+8], dtype=np.uint16)
        for c in range(4):
            if words[c] & 0x8000:
                errs[c] += 1
    print(f"{path} Info BE: {errs}")

check_dump("engines/tile_matrix_4cw/dump/tensix_initial_vnodes.bin")
check_dump("engines/tile_matrix_4cw/dump/tensix_final_vnodes.bin")
check_dump("engines/tile_matrix_4cw/dump/golden_final_vnodes.bin")

data = open("engines/tile_matrix_4cw/dump/tensix_initial_vnodes.bin", "rb").read()[:20480]
for v in range(4):
    t = v // 256
    v_loc = v % 256
    row = v_loc // 8
    k = v_loc % 8
    face = (0 if k < 4 else 1) if row < 16 else (2 if k < 4 else 3)
    col_grp = k if k < 4 else (k - 4)
    off = t * 2048 + face * 512 + (row & 15) * 32 + col_grp * 8
    words = np.frombuffer(data[off:off+8], dtype=np.uint16)
    hex_str = [hex(w) for w in words]
    print(f"v={v}: off={off} words={hex_str}")

