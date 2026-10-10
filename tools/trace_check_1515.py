import struct
import numpy as np

f_init  = open("engines/tile_matrix_4cw/dump/tensix_initial_vnodes.bin", "rb").read()[:20480]
f_final = open("engines/tile_matrix_4cw/dump/tensix_final_vnodes.bin", "rb").read()[:20480]
f_gold  = open("engines/tile_matrix_4cw/dump/golden_final_vnodes.bin", "rb").read()[:20480]

w_init  = np.frombuffer(f_init, dtype=np.uint16)
w_final = np.frombuffer(f_final, dtype=np.uint16)
w_gold  = np.frombuffer(f_gold, dtype=np.uint16)

def bf_to_f(u):
    return struct.unpack('>f', struct.pack('>HH', int(u), 0))[0]

def f_to_bf(f):
    u = struct.unpack('>I', struct.pack('>f', f))[0]
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16) & 0xFFFF

def to_bf16(f):
    return bf_to_f(f_to_bf(f))

def get_tile_vnode_byte_offset(v):
    t = v // 256
    v_loc = v % 256
    row = v_loc // 8
    k = v_loc % 8
    face = 0 if (row < 16 and k < 4) else (1 if row < 16 else (2 if k < 4 else 3))
    col_grp = k if k < 4 else (k - 4)
    return t * 2048 + (face * 512) + ((row & 15) << 5) + (col_grp << 3)

c1515_vars = [0, 84, 1318, 2104, 2274, 2532]
print("Check 1515 Variable values (CW 2):")
q = []
for r, v in enumerate(c1515_vars):
    off = get_tile_vnode_byte_offset(v)
    u16_idx = off // 2 + 2 # CW 2
    u_i = w_init[u16_idx]
    u_f = w_final[u16_idx]
    u_g = w_gold[u16_idx]
    val = bf_to_f(u_i)
    q.append(val)
    print(f"  Edge {r} (v={v}, off=0x{off:04X}): init=0x{u_i:04X} ({val:.4f}) | tens=0x{u_f:04X} | gold=0x{u_g:04X}")

# Let's trace Check 1515
def amin_star_sim(a, b):
    min_v = min(a, b)
    max_v = max(a, b)
    diff = to_bf16(max_v - min_v)
    c1 = bf_to_f(16177)
    c2 = bf_to_f(48844)
    c3 = bf_to_f(16168)
    c4 = bf_to_f(48807)
    line1 = to_bf16(to_bf16(c2 * diff) + c1)
    line2 = to_bf16(to_bf16(c4 * diff) + c3)
    corr = max(max(line1, line2), 0.0)
    return max(to_bf16(min_v - corr), 0.0)

signs = [1.0 if x >= 0 else -1.0 for x in q]
mags = [abs(x) for x in q]
total_sign = 1.0
for s in signs: total_sign *= s

min_mag = min(mags)
min_idx = mags.index(min_mag)

m_others = 1000.0 if min_idx == 0 else mags[0]
for r in range(1, 6):
    m = 1000.0 if min_idx == r else mags[r]
    m_others = amin_star_sim(m_others, m)

m_all = amin_star_sim(m_others, min_mag)

print(f"\nmin_idx={min_idx}, min_mag={min_mag:.4f}, m_others={m_others:.4f}, m_all={m_all:.4f}")

for r in range(6):
    edge_mag = m_others if r == min_idx else m_all
    edge_sign = total_sign * signs[r]
    r_new = to_bf16(edge_sign * edge_mag)
    l_new = to_bf16(q[r] + r_new)
    print(f"  Edge {r} (v={c1515_vars[r]}): r_new=0x{f_to_bf(r_new):04X} ({r_new:.4f}), l_new=0x{f_to_bf(l_new):04X} ({l_new:.4f})")
