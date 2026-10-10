import struct

with open("matrices/AR4JA_r45_4c_128c_r12.chinn.out") as f:
    lines = [line.strip().split() for line in f if line.strip()]

M = int(lines[-1][0])
edges = [(int(p[0])-1, int(p[1])-1) for p in lines[:-1] if float(p[2]) > 0.5]

def get_tile_vnode_byte_offset(v):
    t = v // 256
    v_in_t = v % 256
    p = v_in_t // 16
    idx_in_p = v_in_t % 16
    w = idx_in_p // 4
    k = idx_in_p % 4
    return t * 2048 + p * 128 + w * 64 + k * 16

# Check which variables map to byte offset 0x0000
mapping_to_0 = [v for v in range(2560) if get_tile_vnode_byte_offset(v) == 0]
print("Variables mapping to offset 0x0000:", mapping_to_0)
