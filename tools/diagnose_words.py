import numpy as np

with open('dump/tensix_initial_vnodes.bin', 'rb') as f: init_b = f.read()
with open('dump/tensix_final_vnodes.bin', 'rb') as f: ten_b = f.read()
with open('dump/golden_final_vnodes.bin', 'rb') as f: gold_b = f.read()

w_init = np.frombuffer(init_b, dtype=np.uint16)
w_ten  = np.frombuffer(ten_b,  dtype=np.uint16)
w_gold = np.frombuffer(gold_b, dtype=np.uint16)

# Look at words in Tile 0 (0..1023)
changed_ten = np.where(w_ten[:1024] != w_init[:1024])[0]
changed_gold = np.where(w_gold[:1024] != w_init[:1024])[0]

print('Count changed in Tensix:', len(changed_ten))
print('Count changed in Gold:  ', len(changed_gold))
print('First 20 changed in Tensix:', changed_ten[:20])
print('First 20 changed in Gold:  ', changed_gold[:20])
ten_not_gold = sorted(list(set(changed_ten) - set(changed_gold)))
print('Set difference (in Ten but NOT in Gold):', len(ten_not_gold))
print('First 20 in Ten but NOT Gold:', ten_not_gold[:20])

for idx in ten_not_gold[:5]:
    byte_off = idx * 2
    print(f"Word {idx} (byte {byte_off}): init=0x{w_init[idx]:04x}, gold=0x{w_gold[idx]:04x}, ten=0x{w_ten[idx]:04x}")
