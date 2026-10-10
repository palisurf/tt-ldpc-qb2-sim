import struct
import numpy as np

def to_bf16(val):
    u = struct.unpack('>I', struct.pack('>f', float(val)))[0]
    lsb = (u >> 16) & 1
    u += 0x7fff + lsb
    u16 = (u >> 16) & 0xffff
    return struct.unpack('>f', struct.pack('>I', u16 << 16))[0], u16

# Check 212 variables:
# Let's read matrix
pcm_M = 1536
edges = {}
with open('matrices/AR4JA_r45_4c_128c_r12.chinn.out', 'r') as f:
    for line in f:
        p = line.strip().split()
        if len(p) >= 3:
            r, c, v = int(p[0]) - 1, int(p[1]) - 1, float(p[2])
            if v != 0.0:
                edges.setdefault(r, []).append(c)

check212_vars = edges[1515]
print(f"Check 1515 variables: {check212_vars}")

init_v = np.fromfile("engines/tile_matrix_4cw/dump/tensix_initial_vnodes.bin", dtype=np.uint16)

def get_v_cw(v, cw):
    # byte offset of v
    t = v // 256
    v_loc = v % 256
    row = v_loc // 8
    k = v_loc % 8
    face = (0 if k < 4 else 1) if row < 16 else (2 if k < 4 else 3)
    col_grp = k if k < 4 else (k - 4)
    byte_off = t * 2048 + face * 512 + (row & 15) * 32 + col_grp * 8 + cw * 2
    u16 = init_v[byte_off // 2]
    flt, _ = to_bf16(struct.unpack('>f', struct.pack('>I', int(u16) << 16))[0])
    return flt, u16

print("\nInitial values for Check 212 (CW 2):")
q_cw2 = []
for idx, v in enumerate(check212_vars):
    flt, u16 = get_v_cw(v, 2)
    q_cw2.append(flt)
    print(f"  Edge {idx} (Var {v}): {flt:+.4f} (0x{u16:04x})")

# Let's trace Approx-Min* on q_cw2
min_idx = np.argmin(np.abs(q_cw2))
min_mag = np.abs(q_cw2[min_idx])
total_sign = np.prod(np.sign(q_cw2))
print(f"min_idx={min_idx}, min_mag={min_mag}, total_sign={total_sign}")

# Formulations
def amin_star_fp32(a, b):
    min_v = min(a, b)
    max_v = max(a, b)
    diff = max_v - min_v
    line1 = 0.69315 - 0.4000 * diff
    line2 = 0.65715 - 0.3280 * diff
    corr = max(max(line1, line2), 0.0)
    return max(min_v - corr, 0.0)

def amin_star_bf16_step(a, b):
    min_v = min(a, b)
    max_v = max(a, b)
    diff, _ = to_bf16(max_v - min_v)
    c1, _ = to_bf16(0.69315)
    c2, _ = to_bf16(-0.4000)
    c3, _ = to_bf16(0.65715)
    c4, _ = to_bf16(-0.3280)
    p1, _ = to_bf16(c2 * diff)
    line1, _ = to_bf16(c1 + p1)
    p2, _ = to_bf16(c4 * diff)
    line2, _ = to_bf16(c3 + p2)
    corr = max(max(line1, line2), 0.0)
    res, _ = to_bf16(min_v - corr)
    return max(res, 0.0)

# Accumulate m_others
m_others_fp32 = 1000.0 if min_idx == 0 else abs(q_cw2[0])
m_others_bf16 = 1000.0 if min_idx == 0 else abs(q_cw2[0])
for r in range(1, 6):
    m = 1000.0 if min_idx == r else abs(q_cw2[r])
    m_others_fp32 = amin_star_fp32(m_others_fp32, m)
    m_others_bf16 = amin_star_bf16_step(m_others_bf16, m)

m_all_fp32 = amin_star_fp32(m_others_fp32, min_mag)
m_all_bf16 = amin_star_bf16_step(m_others_bf16, min_mag)

print(f"m_others: FP32={m_others_fp32:.5f}, BF16={m_others_bf16:.5f}")
print(f"m_all   : FP32={m_all_fp32:.5f}, BF16={m_all_bf16:.5f}")

for r in range(6):
    m_d_fp32 = m_others_fp32 if r == min_idx else m_all_fp32
    m_d_bf16 = m_others_bf16 if r == min_idx else m_all_bf16
    sign = total_sign * np.sign(q_cw2[r])
    
    r_fp32 = sign * m_d_fp32
    l_fp32 = q_cw2[r] + r_fp32
    _, u_l_fp32 = to_bf16(l_fp32)
    
    r_bf16, _ = to_bf16(sign * m_d_bf16)
    l_bf16, u_l_bf16 = to_bf16(q_cw2[r] + r_bf16)
    
    v = check212_vars[r]
    print(f"  Var {v} (Edge {r}): FP32 L_new={l_fp32:.4f} (0x{u_l_fp32:04x}) | BF16 L_new={l_bf16:.4f} (0x{u_l_bf16:04x})")
