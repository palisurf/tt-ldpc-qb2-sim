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

gold_changed = []
ten_changed = []

for v in range(2560):
    idx = get_v_offset(v) // 2
    if w_gold[idx] != w_init[idx]:
        gold_changed.append(v)
    if w_ten[idx] != w_init[idx]:
        ten_changed.append(v)

print(f"Total vars changed in Golden: {len(gold_changed)} / 2560")
print(f"Total vars changed in Tensix: {len(ten_changed)} / 2560")
print(f"First 20 vars changed in Golden: {gold_changed[:20]}")
print(f"First 20 vars changed in Tensix: {ten_changed[:20]}")

# Check Var 0 in particular
idx0 = get_v_offset(0) // 2
print(f"\nVar 0: init=0x{w_init[idx0]:04x}, gold=0x{w_gold[idx0]:04x}, ten=0x{w_ten[idx0]:04x}")
for cw in range(4):
    print(f"  CW{cw}: init=0x{w_init[idx0+cw]:04x}, gold=0x{w_gold[idx0+cw]:04x}, ten=0x{w_ten[idx0+cw]:04x}")
