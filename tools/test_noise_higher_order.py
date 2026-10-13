#!/usr/bin/env python3
import numpy as np

# Simulate 200 batches and examine:
# 1. 32-lane Cross-correlation matrix
# 2. Lag-k autocorrelation
# 3. Check node neighbor joint error distribution

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from simulate_tensix_noise_distribution import simulate_batch

def main():
    N_batches = 200
    # Let's inspect the correlation across 32 lanes
    # In each batch, 8 tiles, each has 32 rows x 32 lanes
    # Let's extract per-lane series: shape (N_batches * 8 * 32, 32)
    lane_data = []
    for b in range(N_batches):
        inv = b // 64
        batch_idx = b % 64
        # Return 8192 elements. In our script, each tile was 32x32 = 1024 elements.
        # tile_elements has shape (32, 32) where col is lane!
        # Let's see:
        num_tiles = 8
        vConstTileId = np.arange(32, dtype=np.uint32)
        rate = 0.5
        eb_n0_db = 2.0
        snr_lin = 10.0 ** (eb_n0_db / 10.0)
        mu_val = float(4.0 * rate * snr_lin)
        sigma_val = float(np.sqrt(8.0 * rate * snr_lin))

        from simulate_tensix_noise_distribution import sfpu_fast_ln_vec, sfpu_fast_sqrt_vec

        for t in range(num_tiles):
            seed_base = np.uint32((0x12345678 + (inv * 1013904223) + batch_idx * 31337 + t * 7919) & 0xFFFFFFFF)
            prng_state = seed_base + vConstTileId
            prng_state = ((prng_state << np.uint32(5)) ^ vConstTileId ^ np.uint32(0x55AA55AA)) & np.uint32(0xFFFFFFFF)
            prng_state[prng_state == 0] = np.uint32(0x13579BDF)

            tile_elements = np.zeros((32, 32), dtype=np.float32)
            for p in range(16):
                prng_state ^= (prng_state << np.uint32(13)) & np.uint32(0xFFFFFFFF)
                prng_state ^= (prng_state >> np.uint32(17)) & np.uint32(0xFFFFFFFF)
                prng_state ^= (prng_state << np.uint32(5)) & np.uint32(0xFFFFFFFF)
                u1_bits = ((prng_state >> np.uint32(9)) & np.uint32(0x007FFFFF)) | np.uint32(0x3F800000)
                u1 = u1_bits.view(np.float32) - 1.0
                u1 = np.maximum(u1, 1e-7)
                ln_u1 = sfpu_fast_ln_vec(u1)
                r = sfpu_fast_sqrt_vec(ln_u1 * -2.0)

                prng_state ^= (prng_state << np.uint32(13)) & np.uint32(0xFFFFFFFF)
                prng_state ^= (prng_state >> np.uint32(17)) & np.uint32(0xFFFFFFFF)
                prng_state ^= (prng_state << np.uint32(5)) & np.uint32(0xFFFFFFFF)
                u2_bits = ((prng_state >> np.uint32(9)) & np.uint32(0x007FFFFF)) | np.uint32(0x3F800000)
                u2 = u2_bits.view(np.float32) - 1.0
                theta = u2 * 6.2831853 - 3.14159265
                s_th = theta * theta

                cos_val = 0.0000248 * s_th - 0.0013889
                cos_val = cos_val * s_th + 0.0416667
                cos_val = cos_val * s_th - 0.5
                cos_val = cos_val * s_th + 1.0
                row_0 = mu_val + sigma_val * (r * cos_val)

                sin_val = 0.00000276 * s_th - 0.0001984
                sin_val = sin_val * s_th + 0.0833333
                sin_val = sin_val * s_th - 0.1666667
                sin_val = sin_val * s_th + 1.0
                row_1 = mu_val + sigma_val * (r * (sin_val * theta))

                tile_elements[2 * p + 0, :] = row_0
                tile_elements[2 * p + 1, :] = row_1

            lane_data.append(tile_elements)

    # lane_data is (N_batches * 8, 32, 32) -> reshape to (N_samples, 32)
    all_lanes = np.vstack(lane_data) # shape: (51200, 32)
    print(f"Total lane samples: {all_lanes.shape[0]} rows x 32 lanes")

    # Compute correlation matrix across 32 lanes
    corr = np.corrcoef(all_lanes, rowvar=False)
    # Zero out diagonal
    np.fill_diagonal(corr, 0.0)
    max_corr = np.max(np.abs(corr))
    mean_corr = np.mean(np.abs(corr))
    print(f"Cross-Lane Correlation: Max |rho| = {max_corr:.6f}, Mean |rho| = {mean_corr:.6f}")

    # Check lag-k autocorrelation along lanes
    lags = [1, 2, 3, 4, 8, 16, 32]
    print("\nAutocorrelation along sample sequence (Lane 0):")
    x = all_lanes[:, 0] - np.mean(all_lanes[:, 0])
    var_x = np.var(all_lanes[:, 0])
    for lag in lags:
        rho = np.mean(x[:-lag] * x[lag:]) / var_x
        print(f"  Lag {lag:2d}: rho = {rho:+.6f}")

if __name__ == '__main__':
    main()
