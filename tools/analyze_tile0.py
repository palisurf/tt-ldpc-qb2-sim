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

golden_bytes = open("engines/tile_matrix_4cw/dump/golden_final_vnodes.bin", "rb").read()[:20480]
tensix_bytes = open("engines/tile_matrix_4cw/dump/tensix_final_vnodes.bin", "rb").read()[:20480]
init_bytes   = open("engines/tile_matrix_4cw/dump/tensix_initial_vnodes.bin", "rb").read()[:20480]

g_words = np.frombuffer(golden_bytes, dtype=np.uint16)
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

g_touched = set()
t_touched = set()
N = 2560
for v in range(N):
    iw = get_var_words(i_words, v)
    gw = get_var_words(g_words, v)
    tw = get_var_words(t_words, v)
    if not np.array_equal(iw, gw): g_touched.add(v)
    if not np.array_equal(iw, tw): t_touched.add(v)

print(f"Tile 0 variables touched by Golden: {len(g_touched)}")
print(f"Tile 0 variables touched by Tensix: {len(t_touched)}")
common = g_touched.intersection(t_touched)
print(f"Common touched variables: {len(common)}")
print(f"Golden only touched: {len(g_touched - t_touched)}")
print(f"Tensix only touched: {len(t_touched - g_touched)}")

# For the common touched variables, how many words match?
common_exact_match = 0
for v in common:
    gw = get_var_words(g_words, v)
    tw = get_var_words(t_words, v)
    if np.array_equal(gw, tw):
        common_exact_match += 1
print(f"Common variables with 100% exact match across all 4 CWs: {common_exact_match} / {len(common)}")

# Let's inspect the first 10 differing variables
print("\nFirst 10 common variables comparison:")
count = 0
for v in sorted(common):
    gw = get_var_words(g_words, v)
    tw = get_var_words(t_words, v)
    iw = get_var_words(i_words, v)
    if not np.array_equal(gw, tw):
        print(f"v={v:4d}: Init={[f'{w:04x}' for w in iw]}")
        print(f"        Gold={[f'{w:04x}' for w in gw]}")
        print(f"        Tenx={[f'{w:04x}' for w in tw]}")
        count += 1
        if count >= 10: break

if len(g_touched - t_touched) > 0:
    print(f"\nGolden only variables (first 10): {sorted(list(g_touched - t_touched))[:10]}")
if len(t_touched - g_touched) > 0:
    print(f"\nTensix only variables (first 10): {sorted(list(t_touched - g_touched))[:10]}")
