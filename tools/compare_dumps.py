import numpy as np
import os

p_gold = "dump/golden_final_vnodes.bin"
p_tensix = "dump/tensix_final_vnodes.bin"

if not os.path.exists(p_gold):
    p_gold = "z:/LDPC_QB2_GA/" + p_gold
    p_tensix = "z:/LDPC_QB2_GA/" + p_tensix

g = np.fromfile(p_gold, dtype=np.uint16)
t = np.fromfile(p_tensix, dtype=np.uint16)

diff = (g != t)
print(f"Total elements: {len(g)}")
print(f"Mismatch count: {np.sum(diff)}")
if np.sum(diff) == 0:
    print(">>> 100% BIT-EXACT MATCH! (0 / 10240 mismatches) <<<")
else:
    print(f"Mismatches: {np.sum(diff)}")
