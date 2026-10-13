import numpy as np

def sim_prng(batch, t):
    seed_base = (0x12345678 + 1013904223 + batch * 31337 + t * 7919) & 0xFFFFFFFF
    tile_vals = []
    for lane in range(32):
        tile_id = 2 * lane
        st = (seed_base + tile_id) & 0xFFFFFFFF
        st = ((st << 5) ^ tile_id ^ 0x55AA55AA) & 0xFFFFFFFF
        if st == 0: st = 0x13579BDF
        lane_vals = []
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
            z0 = r * np.cos(theta)
            z1 = r * np.sin(theta)
            lane_vals.append(z0)
            lane_vals.append(z1)
        tile_vals.append(lane_vals)
    return np.array(tile_vals)

for b in range(15):
    all_z = []
    for t in range(8):
        z = sim_prng(b, t)
        all_z.append(z)
    all_z = np.array(all_z)
    print(f'Batch {b:2d}: mean = {np.mean(all_z):.4f}, std = {np.std(all_z):.4f}, min = {np.min(all_z):.2f}, max = {np.max(all_z):.2f}')
