import numpy as np

f_init = open("engines/tile_matrix_4cw/dump/tensix_initial_vnodes.bin", "rb").read()[:20480]
print(f"Read {len(f_init)} bytes from initial dump")

# Let's inspect the first 16 words (Variable 0 and 1)
w = np.frombuffer(f_init, dtype=np.uint16)
print("Initial Words [0..7]:", [hex(x) for x in w[:8]])
