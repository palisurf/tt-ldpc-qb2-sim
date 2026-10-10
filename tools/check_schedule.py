import subprocess, sys

# Let's see how schedule is constructed in ldpc_sim_tile_matrix
# Or let's inspect the offsets in dump/
with open('dump/tensix_initial_vnodes.bin', 'rb') as f:
    init_b = f.read()

print(f"Dump size: {len(init_b)} bytes")
