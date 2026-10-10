import numpy as np

with open('dump/tensix_initial_vnodes.bin', 'rb') as f: init_b = f.read()
with open('dump/tensix_final_vnodes.bin', 'rb') as f: ten_b = f.read()
with open('dump/golden_final_vnodes.bin', 'rb') as f: gold_b = f.read()

w_init = np.frombuffer(init_b, dtype=np.uint16)
w_ten  = np.frombuffer(ten_b,  dtype=np.uint16)
w_gold = np.frombuffer(gold_b, dtype=np.uint16)

# In Tile 0, check which indices differ
mismatches = []
for i in range(1024):
    if w_ten[i] != w_gold[i]:
        mismatches.append(i)

print(f"Total mismatches in Tile 0: {len(mismatches)} / 1024")
print("First 20 mismatched indices in Tile 0:")
for idx in mismatches[:20]:
    # Calculate row, col, face
    face = idx // 256
    face_idx = idx % 256
    row_in_face = face_idx // 16
    col_in_face = face_idx % 16
    print(f"  idx {idx:4d} (Face {face}, Row {row_in_face:2d}, Col {col_in_face:2d}): Init=0x{w_init[idx]:04x} Gold=0x{w_gold[idx]:04x} Ten=0x{w_ten[idx]:04x}")
