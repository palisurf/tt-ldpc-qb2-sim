import numpy as np

with open('dump/tensix_initial_vnodes.bin', 'rb') as f: init_b = f.read()
with open('dump/tensix_final_vnodes.bin', 'rb') as f: ten_b = f.read()
with open('dump/golden_final_vnodes.bin', 'rb') as f: gold_b = f.read()

w_init = np.frombuffer(init_b, dtype=np.uint16)
w_ten  = np.frombuffer(ten_b,  dtype=np.uint16)
w_gold = np.frombuffer(gold_b, dtype=np.uint16)

# Check tile by tile (10 tiles in CB_VNODE_A, each 1024 uint16 elements)
for t in range(10):
    start = t * 1024
    end = start + 1024
    t_init = w_init[start:end]
    t_ten  = w_ten[start:end]
    t_gold = w_gold[start:end]
    
    diff_ten_init = np.count_nonzero(t_ten != t_init)
    diff_gold_init = np.count_nonzero(t_gold != t_init)
    diff_ten_gold = np.count_nonzero(t_ten != t_gold)
    
    print(f"Tile {t}: Ten!=Init: {diff_ten_init}/1024 | Gold!=Init: {diff_gold_init}/1024 | Ten!=Gold: {diff_ten_gold}/1024")

# Check first few words in Tile 0:
print("\nFirst 16 words of Tile 0:")
print("Init: ", [f"0x{x:04x}" for x in w_init[:16]])
print("Gold: ", [f"0x{x:04x}" for x in w_gold[:16]])
print("Ten:  ", [f"0x{x:04x}" for x in w_ten[:16]])
