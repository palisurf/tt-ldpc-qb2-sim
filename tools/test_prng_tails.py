#!/usr/bin/env python3
import math
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

print("Simulating 1,000 batches (8,192,000 Gaussian draws) of Tensix Xorshift32 PRNG...")
all_samples = []
for b in range(1000):
    for t in range(8):
        z = sim_prng(b, t)
        all_samples.append(z.flatten())

all_samples = np.concatenate(all_samples)
total = len(all_samples)
print(f"Total Samples Generated: {total}")
mean = np.mean(all_samples)
std = np.std(all_samples)
min_v = np.min(all_samples)
max_v = np.max(all_samples)
print(f"Empirical Mean: {mean:.6f} (Ideal: 0.0)")
print(f"Empirical Std:  {std:.6f} (Ideal: 1.0)")
print(f"Min Sample:     {min_v:.4f}")
print(f"Max Sample:     {max_v:.4f}")

# Tail analysis
print("\n--- GAUSSIAN TAIL DISTRIBUTION BENCHMARK ---")
for thresh in [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
    cnt = np.count_nonzero(all_samples <= -thresh)
    emp_prob = cnt / total
    # Theoretical Gaussian Q(thresh)
    theory_prob = 0.5 * math.erfc(thresh / math.sqrt(2.0))
    ratio = emp_prob / theory_prob if theory_prob > 0 else 0
    print(f"P(Z <= -{thresh:.1f}): Hits={cnt:6d} | Emp={emp_prob:.4e} | Theory={theory_prob:.4e} | Ratio={ratio:.2f}x")
