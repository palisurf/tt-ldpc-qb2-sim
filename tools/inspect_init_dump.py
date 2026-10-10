import numpy as np

def bf16_to_fp32(val):
    u32 = np.uint32(val) << 16
    return np.array([u32], dtype=np.uint32).view(np.float32)[0]

data = np.fromfile("engines/tile_matrix_4cw/dump/tensix_initial_vnodes.bin", dtype=np.uint16)
print(f"Total words: {len(data)}")

# Let's inspect the first 64 words of Tile 0
print("Tile 0 first 32 words:")
for i in range(0, 32, 4):
    vals = [float(bf16_to_fp32(data[i+j])) for j in range(4)]
    hexs = [hex(data[i+j]) for j in range(4)]
    print(f"words {i:2d}..{i+3:2d}: {hexs} -> {vals}")

# Let's check how many are negative in data
neg_count = sum(1 for x in data if (x & 0x8000) != 0)
print(f"Total negative words in 10 tiles: {neg_count} / {len(data)}")
