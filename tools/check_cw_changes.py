import numpy as np

with open('dump/tensix_initial_vnodes.bin', 'rb') as f: init_b = f.read()
with open('dump/golden_final_vnodes.bin', 'rb') as f: gold_b = f.read()

w_init = np.frombuffer(init_b, dtype=np.uint16)
w_gold = np.frombuffer(gold_b, dtype=np.uint16)

# In Tile 0 (words 0..1023):
# Each group of 4 words is: CW0, CW1, CW2, CW3 of a variable!
cw_changes = [0, 0, 0, 0]
for v in range(256):
    for cw in range(4):
        idx = 4 * v + cw
        if w_gold[idx] != w_init[idx]:
            cw_changes[cw] += 1

print(f"Golden Tile 0 changes by CW: CW0={cw_changes[0]}, CW1={cw_changes[1]}, CW2={cw_changes[2]}, CW3={cw_changes[3]} / 256")
