import struct

def bf_to_f(u):
    return struct.unpack('>f', struct.pack('>HH', int(u), 0))[0]

def f_to_bf(f):
    u = struct.unpack('>I', struct.pack('>f', f))[0]
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16) & 0xFFFF

# Parse PCM
check_to_var = {}
import os
pcm_path = '../../matrices/AR4JA_r45_4c_128c_r12.chinn.out' if os.path.exists('../../matrices/AR4JA_r45_4c_128c_r12.chinn.out') else 'matrices/AR4JA_r45_4c_128c_r12.chinn.out'
with open(pcm_path) as f:
    for line in f:
        p = line.split()
        if len(p) == 3:
            r, c, val = int(p[0]) - 1, int(p[1]) - 1, float(p[2])
            if val != 0.0:
                check_to_var.setdefault(r, []).append(c)

v_nodes_212 = check_to_var[212]
print(f"Check 212 connected variables: {v_nodes_212}")
v_nodes_1515 = check_to_var[1515]
print(f"Check 1515 connected variables: {v_nodes_1515}")

# Now let's read initial values from dump
tensix_init = open('dump/tensix_initial_vnodes.bin', 'rb').read()
# Each variable node has 4 codewords (8 bytes)
# In L1 layout, where is variable v?
# get_tile_vnode_byte_offset(v)
def get_v_offset(v):
    t = v // 256
    v_loc = v % 256
    row = v_loc // 8
    k = v_loc % 8
    face = 0 if row < 16 and k < 4 else (1 if row < 16 and k >= 4 else (2 if row >= 16 and k < 4 else 3))
    col_grp = k if k < 4 else k - 4
    return t * 2048 + face * 512 + ((row & 15) << 5) + (col_grp << 3)

for v in v_nodes_212:
    off = get_v_offset(v)
    cw_vals = [bf_to_f(struct.unpack('<H', tensix_init[off + i*2 : off + i*2 + 2])[0]) for i in range(4)]
    print(f"Var {v:4d} initial: CW0={cw_vals[0]:.4f}, CW1={cw_vals[1]:.4f}, CW2={cw_vals[2]:.4f}, CW3={cw_vals[3]:.4f}")
