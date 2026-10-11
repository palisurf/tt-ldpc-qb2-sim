import struct
import numpy as np

def to_f(u):
    return struct.unpack('>f', struct.pack('>HH', u, 0))[0]

def to_u16(f):
    u = struct.unpack('>I', struct.pack('>f', f))[0]
    return u >> 16

def approx_min_star(a, b):
    # Matches sfpi_amin_star_branchless
    min_v = min(a, b)
    max_v = max(a, b)
    diff = max_v - min_v
    c_0_69315 = to_f(0x3F31)
    c_neg_0_4 = to_f(0xBECC)
    c_0_65715 = to_f(0x3F28)
    c_neg_0_328 = to_f(0xBEA7)
    line1 = c_0_69315 + c_neg_0_4 * diff
    line2 = c_0_65715 + c_neg_0_328 * diff
    corr = max(max(line1, line2), 0.0)
    return max(min_v - corr, 0.0)

tensix_out = np.frombuffer(open("dump/tensix_scratch_out.bin", "rb").read(), dtype=np.uint16)
tensix_in  = np.frombuffer(open("dump/tensix_scratch_in.bin", "rb").read(), dtype=np.uint16)

for face in [2, 3]:
    f_in_words = tensix_in[face*256:(face+1)*256].reshape(16, 16)
    dst_reg = np.zeros((8, 32), dtype=np.float32)

    for r in range(8):
        for lane in range(32):
            row = 4 * (r // 2) + lane // 8
            col = 2 * (lane % 8) + (r % 2)
            dst_reg[r, lane] = to_f(f_in_words[row, col])

    dst_out = np.zeros((8, 32), dtype=np.float32)

    for lane in range(32):
        q = [dst_reg[r, lane] for r in range(6)]
        signs = [-1.0 if x < 0 else 1.0 for x in q]
        mags = [abs(x) for x in q]
        
        total_sign = 1.0
        for s in signs: total_sign *= s
        
        min_idx = int(np.argmin(mags))
        min_mag = mags[min_idx]
        
        m_others = 1000.0 if min_idx == 0 else mags[0]
        for r in range(1, 6):
            m = 1000.0 if min_idx == r else mags[r]
            m_others = approx_min_star(m_others, m)
            
        m_all = approx_min_star(m_others, min_mag)
        
        for r in range(6):
            m_d = m_others if r == min_idx else m_all
            edge_sign = total_sign * signs[r]
            r_out = edge_sign * m_d
            dst_out[r, lane] = q[r] + r_out

    f_sim_out = np.zeros((16, 16), dtype=np.uint16)
    for r in range(8):
        for lane in range(32):
            row = 4 * (r // 2) + lane // 8
            col = 2 * (lane % 8) + (r % 2)
            if r < 6:
                f_sim_out[row, col] = to_u16(dst_out[r, lane])
            else:
                f_sim_out[row, col] = 0

    f_tensix_out = tensix_out[face*256:(face+1)*256].reshape(16, 16)
    matches = np.sum(f_sim_out == f_tensix_out)
    print(f"Face {face}: Simulated Hardware vs Actual Hardware: {matches} / 256 matches ({256 - matches} diffs)")
