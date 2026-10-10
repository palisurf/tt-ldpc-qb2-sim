import numpy as np

# Load initial LLRs from dump
with open('dump/tensix_initial_vnodes.bin', 'rb') as f:
    init_b = f.read()

w_init = np.frombuffer(init_b, dtype=np.uint16)
print(f"Loaded {len(w_init)} words from tensix_initial_vnodes.bin")

with open('dump/golden_final_vnodes.bin', 'rb') as f:
    gold_b = f.read()

w_gold = np.frombuffer(gold_b, dtype=np.uint16)

# Check all tiles in golden vs init:
for t in range(10):
    start = t * 1024
    end = start + 1024
    diff = np.count_nonzero(w_gold[start:end] != w_init[start:end])
    print(f"Tile {t}: changed {diff} / 1024 words")
