#!/usr/bin/env python3
import subprocess, numpy as np, struct

def bf16_to_fp32(val_u16):
    return struct.unpack('>f', struct.pack('>I', int(val_u16) << 16))[0]

def get_cw_errs(fpath):
    with open(fpath, 'rb') as f:
        words = np.frombuffer(f.read()[:20480], dtype=np.uint16)
    errs = [0, 0, 0, 0]
    for c in range(4):
        for v in range(1024):
            t, v_loc = v // 256, v % 256
            row, k = v_loc // 8, v_loc % 8
            face = (0 if k < 4 else 1) if row < 16 else (2 if k < 4 else 3)
            col_grp = k if k < 4 else (k - 4)
            byte_off = t * 2048 + face * 512 + ((row & 15) << 5) + (col_grp << 3) + c * 2
            val = bf16_to_fp32(words[byte_off // 2])
            if val < 0: errs[c] += 1
    return errs

print("--- GOLDEN STEPPER SWEEP (iters 25 to 50) ---")
for it in [25, 28, 30, 32, 34, 35, 36, 37, 38, 40, 50]:
    subprocess.run(
        f"cd /home/ttuser/LDPC_QB2_GA && ./test_40tile_stepper {it} dump/tensix_initial_vnodes.bin dump/tensix_final_vnodes.bin >/dev/null 2>&1",
        shell=True, check=True
    )
    g_errs = get_cw_errs('/home/ttuser/LDPC_QB2_GA/dump/golden_lockstep_final_vnodes.bin')
    print(f"Golden Iter {it:2d}: CW Bit Errors = {g_errs}")
