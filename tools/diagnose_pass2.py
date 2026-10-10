import struct
import numpy as np

def bf16_to_fp32(val_u16):
    u32 = int(val_u16) << 16
    return struct.unpack('>f', struct.pack('>I', u32))[0]

def fp32_to_bf16(f):
    u32 = struct.unpack('>I', struct.pack('>f', f))[0]
    return np.uint16(u32 >> 16)

def approx_min_star(a, b):
    min_v = min(a, b)
    max_v = max(a, b)
    diff = max_v - min_v
    line1 = 0.69315 - 0.4000 * diff
    line2 = 0.65715 - 0.3280 * diff
    corr = max(max(line1, line2), 0.0)
    return max(min_v - corr, 0.0)

g_in = np.fromfile('/home/ttuser/LDPC_QB2_GA/engines/tile_matrix_4cw/dump/golden_scratch_in.bin', dtype=np.uint16)
g_out = np.fromfile('/home/ttuser/LDPC_QB2_GA/engines/tile_matrix_4cw/dump/golden_scratch_out.bin', dtype=np.uint16)
t_out = np.fromfile('/home/ttuser/LDPC_QB2_GA/engines/tile_matrix_4cw/dump/tensix_scratch_out.bin', dtype=np.uint16)

# In golden model:
# for pass=2, base_r = 16, lane = 0:
# r = 0..5:
q_vals = []
offsets = []
for r in range(6):
    reg_idx = 16 + r
    f = reg_idx // 8
    r_in_face = reg_idx % 8
    lane = 0
    row_in_face = 2 * r_in_face if lane < 16 else 2 * r_in_face + 1
    col_in_face = lane % 16
    offset = f * 512 + row_in_face * 32 + col_in_face * 2
    word_idx = offset // 2
    offsets.append(word_idx)
    q_vals.append(bf16_to_fp32(g_in[word_idx]))

print("Offsets for pass=2, lane=0 (r=0..5):", offsets)
print("q_vals:", [f"{v:+.4f}" for v in q_vals])

# Compute Golden:
signs = [(-1.0 if v < 0 else 1.0) for v in q_vals]
mags = [abs(v) for v in q_vals]
total_sign = 1.0
for s in signs: total_sign *= s
min_idx = np.argmin(mags)
min_mag = mags[min_idx]

m_others = 1000.0 if min_idx == 0 else mags[0]
for r in range(1, 6):
    m = 1000.0 if min_idx == r else mags[r]
    m_others = approx_min_star(m_others, m)
m_all = approx_min_star(m_others, min_mag)

print(f"min_idx: {min_idx}, min_mag: {min_mag:.4f}, m_others: {m_others:.4f}, m_all: {m_all:.4f}, total_sign: {total_sign}")

for r in range(6):
    edge_sign = total_sign * signs[r]
    edge_mag = m_others if r == min_idx else m_all
    r_raw = edge_sign * edge_mag
    l_new = q_vals[r] + r_raw
    w = offsets[r]
    g_val = bf16_to_fp32(g_out[w])
    t_val = bf16_to_fp32(t_out[w])
    print(f"  r={r} (elem {w}): expected l_new={l_new:+.4f}, g_out={g_val:+.4f}, t_out={t_val:+.4f}")
