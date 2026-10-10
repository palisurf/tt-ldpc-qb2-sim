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
var_to_check = {}
for r, c in edges:
    if r not in check_to_var: check_to_var[r] = []
    check_to_var[r].append(c)
    if c not in var_to_check: var_to_check[c] = []
    var_to_check[c].append(r)

tensix_bytes = open("engines/tile_matrix_4cw/dump/tensix_final_vnodes.bin", "rb").read()[:20480]
init_bytes   = open("engines/tile_matrix_4cw/dump/tensix_initial_vnodes.bin", "rb").read()[:20480]
t_words = np.frombuffer(tensix_bytes, dtype=np.uint16)
i_words = np.frombuffer(init_bytes, dtype=np.uint16)

def get_var_words(buf_words, v):
    t = v // 256
    v_loc = v % 256
    row = v_loc // 8
    k = v_loc % 8
    face = (0 if k < 4 else 1) if row < 16 else (2 if k < 4 else 3)
    col_grp = k if k < 4 else (k - 4)
    word_off = (t * 2048 + face * 512 + (row & 15) * 32 + col_grp * 8) // 2
    return buf_words[word_off : word_off + 4]

# Variables that ONLY connect to deg-3 checks:
pure_deg3_vars = [v for v in range(N) if all(len(check_to_var[c]) == 3 for c in var_to_check.get(v, []))]
print(f"Total pure deg-3 variables: {len(pure_deg3_vars)}")

deg3_updated_tensix = 0
for v in pure_deg3_vars:
    iw = get_var_words(i_words, v)
    tw = get_var_words(t_words, v)
    if not np.array_equal(iw, tw):
        deg3_updated_tensix += 1

print(f"Pure deg-3 variables updated in Tensix: {deg3_updated_tensix} / {len(pure_deg3_vars)}")

# Variables that ONLY connect to deg-6 checks:
pure_deg6_vars = [v for v in range(N) if all(len(check_to_var[c]) == 6 for c in var_to_check.get(v, []))]
print(f"\nTotal pure deg-6 variables: {len(pure_deg6_vars)}")

deg6_updated_tensix = 0
for v in pure_deg6_vars:
    iw = get_var_words(i_words, v)
    tw = get_var_words(t_words, v)
    if not np.array_equal(iw, tw):
        deg6_updated_tensix += 1

print(f"Pure deg-6 variables updated in Tensix: {deg6_updated_tensix} / {len(pure_deg6_vars)}")
