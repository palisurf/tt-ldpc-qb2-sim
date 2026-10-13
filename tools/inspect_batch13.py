import numpy as np

def generate_tile_channel(batch, t):
    seed_base = (0x12345678 + 1013904223 + batch * 31337 + t * 7919) & 0xFFFFFFFF
    regs = np.zeros((32, 32), dtype=np.float32)
    for lane in range(32):
        tile_id = 2 * lane
        st = (seed_base + tile_id) & 0xFFFFFFFF
        st = ((st << 5) ^ tile_id ^ 0x55AA55AA) & 0xFFFFFFFF
        if st == 0: st = 0x13579BDF
        
        rate = 0.5
        eb_n0_db = 2.0
        snr_lin = 10.0 ** (eb_n0_db / 10.0)
        v_mu = 4.0 * rate * snr_lin
        v_sigma = np.sqrt(8.0 * rate * snr_lin)
        
        for p in range(16):
            # U1
            st = (st ^ ((st << 13) & 0xFFFFFFFF)) & 0xFFFFFFFF
            st = (st ^ (st >> 17)) & 0xFFFFFFFF
            st = (st ^ ((st << 5) & 0xFFFFFFFF)) & 0xFFFFFFFF
            u1 = ((st >> 9) & 0x007FFFFF) / 8388608.0
            u1 = max(u1, 1e-7)
            r = np.sqrt(-2.0 * np.log(u1))
            
            # U2
            st = (st ^ ((st << 13) & 0xFFFFFFFF)) & 0xFFFFFFFF
            st = (st ^ (st >> 17)) & 0xFFFFFFFF
            st = (st ^ ((st << 5) & 0xFFFFFFFF)) & 0xFFFFFFFF
            u2 = ((st >> 9) & 0x007FFFFF) / 8388608.0
            theta = u2 * 6.2831853 - 3.14159265
            s_th = theta * theta
            
            # cos
            cos_val = 0.0000248 * s_th - 0.0013889
            cos_val = cos_val * s_th + 0.0416667
            cos_val = cos_val * s_th - 0.5
            cos_val = cos_val * s_th + 1.0
            regs[2 * p + 0, lane] = v_mu + v_sigma * (r * cos_val)
            
            # sin
            sin_val = 0.00000276 * s_th - 0.0001984
            sin_val = sin_val * s_th + 0.0083333
            sin_val = sin_val * s_th - 0.1666667
            sin_val = sin_val * s_th + 1.0
            regs[2 * p + 1, lane] = v_mu + v_sigma * (r * (sin_val * theta))
            
    return regs

# Let's check unpacker mapping from dst_reg to faces
# For Face 0 (regs 0..7):
# reg 2k: even cols of quartet k
# reg 2k+1: odd cols of quartet k
# In each face, quartet k has 4 rows: row 4k + (lane//8)
# In reg 2k, col in face is 2*(lane%8) + 0 -> even col
# In reg 2k+1, col in face is 2*(lane%8) + 1 -> odd col

def regs_to_vnodes(regs):
    # returns face array: [4, 16, 16]
    faces = np.zeros((4, 16, 16), dtype=np.float32)
    for f in range(4):
        base_reg = f * 8
        for k in range(4):
            r_even = regs[base_reg + 2*k + 0]
            r_odd  = regs[base_reg + 2*k + 1]
            for lane in range(32):
                row = 4 * k + (lane // 8)
                col_even = 2 * (lane % 8) + 0
                col_odd  = 2 * (lane % 8) + 1
                faces[f, row, col_even] = r_even[lane]
                faces[f, row, col_odd]  = r_odd[lane]
    return faces

for batch in [0, 13]:
    cw_llrs = {0: [], 1: [], 2: [], 3: []}
    for t in range(8):
        regs = generate_tile_channel(batch, t)
        faces = regs_to_vnodes(regs)
        # in each face:
        # col_grp 0: cols 0..3 (cw0, cw1, cw2, cw3)
        # col_grp 1: cols 4..7
        # col_grp 2: cols 8..11
        # col_grp 3: cols 12..15
        for f in range(4):
            for row in range(16):
                for cg in range(4):
                    cw_llrs[0].append(faces[f, row, 4*cg + 0])
                    cw_llrs[1].append(faces[f, row, 4*cg + 1])
                    cw_llrs[2].append(faces[f, row, 4*cg + 2])
                    cw_llrs[3].append(faces[f, row, 4*cg + 3])
                    
    print(f'=== Batch {batch} ===')
    for cw in range(4):
        vals = np.array(cw_llrs[cw])
        neg = np.sum(vals < 0)
        print(f'CW{cw}: mean={np.mean(vals):.3f}, std={np.std(vals):.3f}, min={np.min(vals):.3f}, neg_count={neg}/2048')
