import numpy as np

# We can replicate the golden simulation in python for each tile and track when Tile 0 variables change!
# First let's read the binary dump files
with open('dump/tensix_initial_vnodes.bin', 'rb') as f: init_b = f.read()
with open('dump/golden_final_vnodes.bin', 'rb') as f: gold_b = f.read()
with open('dump/tensix_final_vnodes.bin', 'rb') as f: ten_b = f.read()

w_init = np.frombuffer(init_b, dtype=np.uint16)
w_gold = np.frombuffer(gold_b, dtype=np.uint16)
w_ten  = np.frombuffer(ten_b,  dtype=np.uint16)

print("Difference overall:")
diff_all = np.count_nonzero(w_gold != w_ten)
print(f"Total words differing: {diff_all} / 10240")

# Compare Tile 0..9
for t in range(10):
    start = t * 1024
    end = start + 1024
    diff_t = np.count_nonzero(w_gold[start:end] != w_ten[start:end])
    print(f"Tile {t} diff: {diff_t} / 1024")
