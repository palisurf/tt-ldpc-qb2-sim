import struct, numpy as np

with open('dump/golden_final_vnodes.bin', 'rb') as f: gold_b = f.read()
with open('dump/tensix_final_vnodes.bin', 'rb') as f: ten_b = f.read()
with open('dump/tensix_initial_vnodes.bin', 'rb') as f: init_b = f.read()

w_gold = np.frombuffer(gold_b, dtype=np.uint16)
w_ten  = np.frombuffer(ten_b,  dtype=np.uint16)
w_init = np.frombuffer(init_b, dtype=np.uint16)

def get_v_offset(v):
    t = v // 256
    v_loc = v % 256
    row = v_loc // 8
    k = v_loc % 8
    face = (0 if k < 4 else 1) if row < 16 else (2 if k < 4 else 3)
    col_grp = k if k < 4 else (k - 4)
    return t * 2048 + (face * 512) + ((row & 15) << 5) + (col_grp << 3)

k_lt_4_mismatches = 0
k_ge_4_mismatches = 0
k_lt_4_total = 0
k_ge_4_total = 0

tensix_updated_k_lt_4 = 0
tensix_updated_k_ge_4 = 0
golden_updated_k_lt_4 = 0
golden_updated_k_ge_4 = 0

for v in range(2560):
    k = (v % 256) % 8
    base_off = get_v_offset(v) // 2 # uint16 index
    for cw in range(4):
        idx = base_off + cw
        g_val = w_gold[idx]
        t_val = w_ten[idx]
        i_val = w_init[idx]
        
        is_mismatch = (g_val != t_val)
        t_updated = (t_val != i_val)
        g_updated = (g_val != i_val)
        
        if k < 4:
            k_lt_4_total += 1
            if is_mismatch: k_lt_4_mismatches += 1
            if t_updated: tensix_updated_k_lt_4 += 1
            if g_updated: golden_updated_k_lt_4 += 1
        else:
            k_ge_4_total += 1
            if is_mismatch: k_ge_4_mismatches += 1
            if t_updated: tensix_updated_k_ge_4 += 1
            if g_updated: golden_updated_k_ge_4 += 1

print(f"k < 4 (Faces 0 & 2): Mismatches = {k_lt_4_mismatches} / {k_lt_4_total} ({k_lt_4_mismatches/k_lt_4_total*100:.2f}%)")
print(f"  Golden updated: {golden_updated_k_lt_4} / {k_lt_4_total}")
print(f"  Tensix updated: {tensix_updated_k_lt_4} / {k_lt_4_total}")

print(f"\nk >= 4 (Faces 1 & 3): Mismatches = {k_ge_4_mismatches} / {k_ge_4_total} ({k_ge_4_mismatches/k_ge_4_total*100:.2f}%)")
print(f"  Golden updated: {golden_updated_k_ge_4} / {k_ge_4_total}")
print(f"  Tensix updated: {tensix_updated_k_ge_4} / {k_ge_4_total}")
