import numpy as np

golden_bytes = open("engines/tile_matrix_4cw/dump/golden_final_vnodes.bin", "rb").read()[:20480]
tensix_bytes = open("engines/tile_matrix_4cw/dump/tensix_final_vnodes.bin", "rb").read()[:20480]
init_bytes   = open("engines/tile_matrix_4cw/dump/tensix_initial_vnodes.bin", "rb").read()[:20480]

g_words = np.frombuffer(golden_bytes, dtype=np.uint16)
t_words = np.frombuffer(tensix_bytes, dtype=np.uint16)
i_words = np.frombuffer(init_bytes, dtype=np.uint16)

# Check per variable (4 codewords per variable, each variable has 4 words)
# Total 2560 variables
print(f"Total words: {len(g_words)}")

def get_var_words(buf_words, v):
    t = v // 256
    v_loc = v % 256
    row = v_loc // 8
    k = v_loc % 8
    face = (0 if k < 4 else 1) if row < 16 else (2 if k < 4 else 3)
    col_grp = k if k < 4 else (k - 4)
    word_off = (t * 2048 + face * 512 + (row & 15) * 32 + col_grp * 8) // 2
    return buf_words[word_off : word_off + 4]

# Let's inspect first 20 variables
print("\nFirst 20 Variables comparison:")
print("Var | Initial (Hex)          | Golden (Hex)           | Tensix (Hex)           | Match?")
print("-" * 80)
for v in range(20):
    iw = [f"{w:04x}" for w in get_var_words(i_words, v)]
    gw = [f"{w:04x}" for w in get_var_words(g_words, v)]
    tw = [f"{w:04x}" for w in get_var_words(t_words, v)]
    match = (gw == tw)
    print(f"{v:3d} | {' '.join(iw)} | {' '.join(gw)} | {' '.join(tw)} | {match}")
