import numpy as np

def parse_pcm(path):
    lines = open(path).readlines()
    edges = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            r, c, val = int(parts[0]), int(parts[1]), float(parts[2])
            if abs(val) > 1e-6:
                edges.append((r-1, c-1))
    dim_line = lines[-1].strip().split()
    M, N = int(dim_line[0]), int(dim_line[1])
    return M, N, edges

M, N, edges = parse_pcm("matrices/AR4JA_r45_4c_128c_r12.chinn.out")
check_to_var = {}
for r, c in edges:
    if r not in check_to_var: check_to_var[r] = []
    check_to_var[r].append(c)

import os
dump_dir = "engines/tile_matrix_4cw/dump"
if not os.path.exists(f"{dump_dir}/tensix_final_vnodes.bin"):
    dump_dir = "Z:/LDPC_QB2_GA/engines/tile_matrix_4cw/dump"

tensix_bytes = open(f"{dump_dir}/tensix_final_vnodes.bin", "rb").read()[:20480]
init_bytes   = open(f"{dump_dir}/tensix_initial_vnodes.bin", "rb").read()[:20480]
golden_bytes = open(f"{dump_dir}/golden_final_vnodes.bin", "rb").read()[:20480]
t_words = np.frombuffer(tensix_bytes, dtype=np.uint16)
i_words = np.frombuffer(init_bytes, dtype=np.uint16)
g_words = np.frombuffer(golden_bytes, dtype=np.uint16)

def get_var_words(buf_words, v):
    t = v // 256
    v_loc = v % 256
    row = v_loc // 8
    k = v_loc % 8
    face = (0 if k < 4 else 1) if row < 16 else (2 if k < 4 else 3)
    col_grp = k if k < 4 else (k - 4)
    word_off = (t * 2048 + face * 512 + (row & 15) * 32 + col_grp * 8) // 2
    return buf_words[word_off : word_off + 4]

updated_vars = set()
golden_updated_vars = set()

for v in range(N):
    iw = get_var_words(i_words, v)
    tw = get_var_words(t_words, v)
    gw = get_var_words(g_words, v)
    if not np.array_equal(iw, tw):
        updated_vars.add(v)
    if not np.array_equal(iw, gw):
        golden_updated_vars.add(v)

print(f"Total variables updated in Tensix: {len(updated_vars)} / {N}")
print(f"Total variables updated in Golden: {len(golden_updated_vars)} / {N}")
common = updated_vars.intersection(golden_updated_vars)
tensix_only = updated_vars - golden_updated_vars
golden_only = golden_updated_vars - updated_vars
neither = set(range(N)) - updated_vars - golden_updated_vars
print(f"Updated in BOTH: {len(common)}")
print(f"Updated in TENSIX ONLY: {len(tensix_only)}")
print(f"Updated in GOLDEN ONLY: {len(golden_only)}")
print(f"Updated in NEITHER: {len(neither)}")



# Let's inspect which tile in the 40-tile schedule touches these updated variables
# Parse schedule from test_40tile_stepper or sim
# Let's read the binary dumps or log
