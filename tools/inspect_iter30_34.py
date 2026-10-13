#!/usr/bin/env python3
import subprocess, numpy as np, struct

def bf16_to_fp32(val_u16):
    return struct.unpack('>f', struct.pack('>I', int(val_u16) << 16))[0]

def extract_cw2(fpath):
    with open(fpath, 'rb') as f:
        words = np.frombuffer(f.read()[:20480], dtype=np.uint16)
    llrs = np.zeros(2560, dtype=np.float32)
    for v in range(2560):
        t, v_loc = v // 256, v % 256
        row, k = v_loc // 8, v_loc % 8
        face = (0 if k < 4 else 1) if row < 16 else (2 if k < 4 else 3)
        col_grp = k if k < 4 else (k - 4)
        byte_off = t * 2048 + face * 512 + ((row & 15) << 5) + (col_grp << 3) + 2 * 2
        llrs[v] = bf16_to_fp32(words[byte_off // 2])
    return llrs

for it in [30, 32, 34]:
    subprocess.run(
        f"export TT_METAL_HOME=/home/ttuser/tt-metal && export TT_METAL_RUNTIME_ROOT=/home/ttuser/tt-metal && "
        f"export ARCH_NAME=blackhole && cd /home/ttuser/LDPC_QB2_GA && "
        f"./engines/tile_matrix_4cw/ldpc_sim_tile_matrix --diff-test-real 2.0 {it} 689 10 >/dev/null 2>&1",
        shell=True, check=True
    )
    subprocess.run(
        f"cd /home/ttuser/LDPC_QB2_GA && ./test_40tile_stepper {it} dump/tensix_initial_vnodes.bin dump/tensix_final_vnodes.bin >/dev/null 2>&1",
        shell=True, check=True
    )
    g = extract_cw2("/home/ttuser/LDPC_QB2_GA/dump/golden_lockstep_final_vnodes.bin")
    t = extract_cw2("/home/ttuser/LDPC_QB2_GA/dump/tensix_final_vnodes.bin")
    g_err_idx = set(np.where(g[:1024] < 0)[0])
    t_err_idx = set(np.where(t[:1024] < 0)[0])
    common = g_err_idx.intersection(t_err_idx)
    g_only = g_err_idx - t_err_idx
    t_only = t_err_idx - g_err_idx
    print(f"Iter {it}: Common Errs={len(common)}, Gold-only={len(g_only)}, Tensix-only={len(t_only)}")
    print(f"   Gold CW2 Mean|L|={np.mean(np.abs(g[:2048])):.2f}, Max|L|={np.max(np.abs(g[:2048])):.2f}")
    print(f"   Ten  CW2 Mean|L|={np.mean(np.abs(t[:2048])):.2f}, Max|L|={np.max(np.abs(t[:2048])):.2f}")
