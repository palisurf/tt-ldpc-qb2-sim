import struct

def bf_to_f(u):
    return struct.unpack('>f', struct.pack('>HH', int(u), 0))[0]

def f_to_bf(f):
    u = struct.unpack('>I', struct.pack('>f', f))[0]
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16) & 0xFFFF

def to_bf16(f):
    return bf_to_f(f_to_bf(f))

def approx_min_star_sfpu(a, b):
    min_v = min(a, b)
    max_v = max(a, b)
    diff  = max_v - min_v
    c1 = bf_to_f(16177)
    c2 = bf_to_f(48844)
    c3 = bf_to_f(16168)
    c4 = bf_to_f(48807)
    line1 = c1 + c2 * diff
    line2 = c3 + c4 * diff
    corr = max(max(line1, line2), 0.0)
    return max(min_v - corr, 0.0)

# Check 1515 variables: [0, 84, 1318, 2104, 2274, 2532]
# Notice 2104, 2274, 2532 are >= 2048, so they are punctured!
# Initial values for punctured nodes: 0.0
# Initial value for 0 is 5.4688 (CW2)
# Let's inspect initial values for 84, 1318:
tensix_init = open('dump/tensix_initial_vnodes.bin', 'rb').read()

def get_v_offset(v):
    t = v // 256
    v_loc = v % 256
    row = v_loc // 8
    k = v_loc % 8
    face = 0 if row < 16 and k < 4 else (1 if row < 16 and k >= 4 else (2 if row >= 16 and k < 4 else 3))
    col_grp = k if k < 4 else k - 4
    return t * 2048 + face * 512 + ((row & 15) << 5) + (col_grp << 3)

for cw in range(4):
    q_vals = []
    for v in [0, 84, 1318, 2104, 2274, 2532]:
        off = get_v_offset(v)
        u16 = struct.unpack('<H', tensix_init[off + cw*2 : off + cw*2 + 2])[0]
        q_vals.append(bf_to_f(u16))
    
    mags = [abs(x) for x in q_vals]
    signs = [-1.0 if x < 0 else 1.0 for x in q_vals]
    total_sign = 1.0
    for s in signs: total_sign *= s
    min_mag = min(mags)
    min_idx = mags.index(min_mag)

    m_others = 1000.0 if min_idx == 0 else mags[0]
    for r in range(1, 6):
        m = 1000.0 if min_idx == r else mags[r]
        m_others = approx_min_star_sfpu(m_others, m)
    m_all = approx_min_star_sfpu(m_others, min_mag)

    r_v0 = (m_others if min_idx == 0 else m_all) * total_sign * signs[0]
    print(f"CW{cw}: Check 1515 produces for Var 0: r_out = {r_v0:.5f} (min_idx={min_idx}, min_mag={min_mag})")
