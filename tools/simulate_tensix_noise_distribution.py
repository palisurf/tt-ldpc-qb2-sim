#!/usr/bin/env python3
import numpy as np
import struct
import math

def float_to_uint(f):
    return struct.unpack('>I', struct.pack('>f', f))[0]

def uint_to_float(u):
    return struct.unpack('>f', struct.pack('>I', u))[0]

# Precompute lookup / vector operations for speed
def sfpu_fast_ln_vec(a):
    a = np.maximum(a, 1e-7)
    # IEEE-754 bit extraction
    # float32 view as uint32
    u = a.view(np.uint32)
    exp_val = ((u >> 23) & 0xFF).astype(np.int32) - 127
    m_bits = (u & 0x007FFFFF) | 0x3F800000
    m = m_bits.view(np.float32)
    z = m - 1.0

    p = -0.0528
    p = p * z + 0.1264
    p = p * z - 0.1678
    p = p * z + 0.2057
    p = p * z - 0.2490
    p = p * z + 0.3328
    p = p * z - 0.5
    p = p * z + 1.0
    ln_m = p * z
    return ln_m + exp_val.astype(np.float32) * 0.69314718

def sfpu_fast_sqrt_vec(val):
    val = np.maximum(val, 1e-7)
    u = val.view(np.uint32)
    magic = np.uint32(0x5F375A86)
    approx_u = magic - (u >> 1)
    approx = approx_u.view(np.float32)
    neg_half_val = val * -0.5
    approx = ((approx * approx) * neg_half_val + 1.5) * approx
    approx = ((approx * approx) * neg_half_val + 1.5) * approx
    return approx * val

def simulate_batch(invocation_id, batch, eb_n0_db=2.0):
    rate = 0.5
    snr_lin = 10.0 ** (eb_n0_db / 10.0)
    mu_val = float(4.0 * rate * snr_lin)
    sigma_val = float(np.sqrt(8.0 * rate * snr_lin))

    num_tiles = 8 # 8 transmitted tiles = 8192 elements
    all_elements = []

    # 32 lanes
    vConstTileId = np.arange(32, dtype=np.uint32)

    for t in range(num_tiles):
        seed_base = np.uint32((0x12345678 + (invocation_id * 1013904223) + batch * 31337 + t * 7919) & 0xFFFFFFFF)
        prng_state = seed_base + vConstTileId
        prng_state = ((prng_state << np.uint32(5)) ^ vConstTileId ^ np.uint32(0x55AA55AA)) & np.uint32(0xFFFFFFFF)
        prng_state[prng_state == 0] = np.uint32(0x13579BDF)

        tile_elements = np.zeros((32, 32), dtype=np.float32)

        for p in range(16):
            # U1
            prng_state ^= (prng_state << np.uint32(13)) & np.uint32(0xFFFFFFFF)
            prng_state ^= (prng_state >> np.uint32(17)) & np.uint32(0xFFFFFFFF)
            prng_state ^= (prng_state << np.uint32(5)) & np.uint32(0xFFFFFFFF)
            u1_bits = ((prng_state >> np.uint32(9)) & np.uint32(0x007FFFFF)) | np.uint32(0x3F800000)
            u1 = u1_bits.view(np.float32) - 1.0
            u1 = np.maximum(u1, 1e-7)

            ln_u1 = sfpu_fast_ln_vec(u1)
            r = sfpu_fast_sqrt_vec(ln_u1 * -2.0)

            # U2
            prng_state ^= (prng_state << np.uint32(13)) & np.uint32(0xFFFFFFFF)
            prng_state ^= (prng_state >> np.uint32(17)) & np.uint32(0xFFFFFFFF)
            prng_state ^= (prng_state << np.uint32(5)) & np.uint32(0xFFFFFFFF)
            u2_bits = ((prng_state >> np.uint32(9)) & np.uint32(0x007FFFFF)) | np.uint32(0x3F800000)
            u2 = u2_bits.view(np.float32) - 1.0

            theta = u2 * 6.2831853 - 3.14159265
            s_th = theta * theta

            # Cosine series
            cos_val = 0.0000248 * s_th - 0.0013889
            cos_val = cos_val * s_th + 0.0416667
            cos_val = cos_val * s_th - 0.5
            cos_val = cos_val * s_th + 1.0
            row_0 = mu_val + sigma_val * (r * cos_val)

            # Sine series
            sin_val = 0.00000276 * s_th - 0.0001984
            sin_val = sin_val * s_th + 0.0083333
            sin_val = sin_val * s_th - 0.1666667
            sin_val = sin_val * s_th + 1.0
            row_1 = mu_val + sigma_val * (r * (sin_val * theta))

            tile_elements[2 * p + 0, :] = row_0
            tile_elements[2 * p + 1, :] = row_1

        all_elements.append(tile_elements.flatten())

    return np.concatenate(all_elements)

def main():
    eb_n0_db = 2.0
    rate = 0.5
    snr_lin = 10.0 ** (eb_n0_db / 10.0)
    theory_mu = 4.0 * rate * snr_lin # 3.169786
    theory_sigma = np.sqrt(8.0 * rate * snr_lin) # 2.517851
    theory_ber = 0.5 * math.erfc(np.sqrt(rate * snr_lin)) # 0.103986

    print(f"Theoretical at 2.0 dB: mu={theory_mu:.4f}, sigma={theory_sigma:.4f}, BER={theory_ber:.4f}")

    # Check Batch 689 (the trapped failing batch from invocation 10)
    b689 = simulate_batch(invocation_id=10, batch=689, eb_n0_db=2.0)
    print(f"\n--- BATCH 689 (Invocation 10, the failing batch trapped on silicon) ---")
    print(f"Mean:  {np.mean(b689):.4f}")
    print(f"Std:   {np.std(b689):.4f}")
    print(f"BER:   {np.count_nonzero(b689 < 0) / len(b689):.4f} (Count: {np.count_nonzero(b689 < 0)} / {len(b689)})")

    # Now test across 500 batches
    print(f"\n--- SIMULATING 500 BATCHES (4,096,000 symbols) ---")
    all_means = []
    all_stds = []
    all_bers = []
    total_negs = 0
    total_syms = 0

    for b in range(500):
        inv = b // 64
        batch_idx = b % 64
        vals = simulate_batch(invocation_id=inv, batch=batch_idx, eb_n0_db=2.0)
        m = np.mean(vals)
        s = np.std(vals)
        negs = np.count_nonzero(vals < 0)
        ber = negs / len(vals)

        all_means.append(m)
        all_stds.append(s)
        all_bers.append(ber)
        total_negs += negs
        total_syms += len(vals)

    print(f"500-Batch Overall Mean:  {np.mean(all_means):.4f} (Theory: {theory_mu:.4f})")
    print(f"500-Batch Overall Std:   {np.mean(all_stds):.4f} (Theory: {theory_sigma:.4f})")
    print(f"500-Batch Overall BER:   {total_negs / total_syms:.4f} (Theory: {theory_ber:.4f})")
    print(f"Min batch BER:           {np.min(all_bers):.4f}")
    print(f"Max batch BER:           {np.max(all_bers):.4f}")

if __name__ == '__main__':
    main()
