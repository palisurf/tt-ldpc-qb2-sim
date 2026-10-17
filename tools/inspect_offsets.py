import numpy as np

t_init = np.fromfile('dump/tensix_initial_vnodes.bin', dtype=np.uint8)
t_scat_in = np.fromfile('dump/tensix_scratch_in.bin', dtype=np.uint16)
g_scat_in = np.fromfile('dump/golden_scratch_in.bin', dtype=np.uint16)

print('t_scat_in first 4 words:', [hex(int(x)) for x in t_scat_in[:4]])
print('g_scat_in first 4 words:', [hex(int(x)) for x in g_scat_in[:4]])

# Let's find where in t_init the 8 bytes of t_scat_in[:4] exist!
t_target_bytes = t_scat_in[:4].tobytes()
g_target_bytes = g_scat_in[:4].tobytes()

t_init_bytes = bytes(t_init)
t_pos = t_init_bytes.find(t_target_bytes)
g_pos = t_init_bytes.find(g_target_bytes)

print('Position of t_scat_in[:4] in tensix_initial_vnodes:', t_pos)
print('Position of g_scat_in[:4] in tensix_initial_vnodes:', g_pos)

# In ldpc_sim_tile_matrix.cpp:
# stats[200] is pcm.h_offsets_pairs[0]!
# Let's inspect dump/tensix_final_vnodes or check what was printed for Tile 0!
# We printed:
# Tile 0 deg from stats[64]: 6
# Tile 0 all 32 checks: 2 107 186 264 379 461 ...

def get_tile_vnode_byte_offset(v):
    t = v // 256
    v_loc = v % 256
    row = v_loc // 8
    k = v_loc % 8
    face = 0 if k < 4 else 1
    k_in_face = k if k < 4 else (k - 4)
    if row < 16:
        sub = 0 if row < 8 else 1
        r_in_sub = row if row < 8 else (row - 8)
        byte_offset = t * 2048 + face * 512 + (2 * r_in_sub + sub) * 32 + k_in_face * 8
    else:
        r2 = row - 16
        face2 = 2 if k < 4 else 3
        sub2 = 0 if r2 < 8 else 1
        r_in_sub2 = r2 if r2 < 8 else (r2 - 8)
        byte_offset = t * 2048 + face2 * 512 + (2 * r_in_sub2 + sub2) * 32 + k_in_face * 8
    return byte_offset

v_184 = [v for v in range(2560) if get_tile_vnode_byte_offset(v) == 184]
v_4344 = [v for v in range(2560) if get_tile_vnode_byte_offset(v) == 4344]
print('Variable node for byte offset 184:', v_184)
print('Variable node for byte offset 4344:', v_4344)
