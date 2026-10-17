import struct

with open("dump/tensix_initial_vnodes.bin", "rb") as f:
    raw = f.read()

vnodes = struct.unpack(f"<{len(raw)//2}H", raw)
# There are 2560 vnodes, each has 4 words (CW0, CW1, CW2, CW3)
# But wait! How are vnodes laid out in tensix_initial_vnodes.bin?
# In memory, vnode v is at byte offset get_tile_vnode_byte_offset(v)

def get_tile_vnode_byte_offset(v):
    t = v // 256
    v_loc = v % 256
    row = v_loc // 8
    k = v_loc % 8
    face = (0 if k < 4 else 1) if row < 16 else (2 if k < 4 else 3)
    col_grp = k if k < 4 else (k - 4)
    return t * 2048 + (face * 512) + ((row & 15) << 5) + (col_grp << 3)

v_to_quad = {}
quad_to_v = {}
for v in range(2560):
    off = get_tile_vnode_byte_offset(v)
    w_idx = off // 2
    quad = (vnodes[w_idx], vnodes[w_idx+1], vnodes[w_idx+2], vnodes[w_idx+3])
    v_to_quad[v] = quad
    quad_to_v[quad] = v

with open("dump/tensix_scratch_in.bin", "rb") as f:
    raw_s = f.read()
scat = struct.unpack(f"<{len(raw_s)//2}H", raw_s)

print("Face 0 Row 0 (kh=0..3):")
for kh in range(4):
    q = (scat[kh*4], scat[kh*4+1], scat[kh*4+2], scat[kh*4+3])
    v = quad_to_v.get(q, None)
    hex_str = " ".join([f"0x{x:04x}" for x in q])
    print(f"  kh={kh}: quad=[{hex_str}] => vnode {v}")

print("\nFace 0 Row 1 (kh=0..3, odd checks):")
for kh in range(4):
    idx = 16 + kh*4
    q = (scat[idx], scat[idx+1], scat[idx+2], scat[idx+3])
    v = quad_to_v.get(q, None)
    hex_str = " ".join([f"0x{x:04x}" for x in q])
    print(f"  kh={kh}: quad=[{hex_str}] => vnode {v}")
