#!/usr/bin/env python3
import numpy as np
import math
import struct
import os, sys

def load_matrix(filepath):
    checks = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
    dim_line = lines[-1].strip().split()
    M = int(dim_line[0])
    N = int(dim_line[1])
    
    check_to_var = [[] for _ in range(M)]
    for line in lines[:-1]:
        parts = line.strip().split()
        if len(parts) >= 3:
            r = int(parts[0]) - 1
            c = int(parts[1]) - 1
            val = float(parts[2])
            if abs(val) > 1e-6:
                check_to_var[r].append(c)
    return M, N, check_to_var

def get_tile_vnode_byte_offset(v):
    t = v // 256
    v_loc = v % 256
    row = v_loc // 8
    k = v_loc % 8
    face = (0 if k < 4 else 1) if (row < 16) else (2 if k < 4 else 3)
    col_grp = k if k < 4 else (k - 4)
    return t * 2048 + (face * 512) + ((row & 15) << 5) + (col_grp << 3)

# Xoshiro128+ implementation for Golden Stepper
class Xoshiro128Plus:
    def __init__(self, seed):
        # Splitmix64 init
        s = seed & 0xFFFFFFFFFFFFFFFF
        def splitmix64():
            nonlocal s
            s = (s + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
            z = s
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
            return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF
        u0 = splitmix64()
        u1 = splitmix64()
        self.s0 = np.uint32(u0 & 0xFFFFFFFF)
        self.s1 = np.uint32((u0 >> 32) & 0xFFFFFFFF)
        self.s2 = np.uint32(u1 & 0xFFFFFFFF)
        self.s3 = np.uint32((u1 >> 32) & 0xFFFFFFFF)

    def next(self):
        res = (int(self.s0) + int(self.s3)) & 0xFFFFFFFF
        t = (int(self.s1) << 9) & 0xFFFFFFFF
        self.s2 ^= self.s0
        self.s3 ^= self.s1
        self.s1 ^= self.s2
        self.s0 ^= self.s3
        self.s2 ^= np.uint32(t)
        s3_rot = ((int(self.s3) << 11) | (int(self.s3) >> 21)) & 0xFFFFFFFF
        self.s3 = np.uint32(s3_rot)
        return int(res)

def generate_awgn_pair_golden(rng, mu_llr, sigma_llr):
    INT32_TO_UNIT = 4.656612875245797e-10
    while True:
        r1 = rng.next()
        r2 = rng.next()
        u1 = r1 if r1 < 0x80000000 else r1 - 0x100000000
        u2 = r2 if r2 < 0x80000000 else r2 - 0x100000000
        v1 = float(u1) * INT32_TO_UNIT
        v2 = float(u2) * INT32_TO_UNIT
        s = v1 * v1 + v2 * v2
        if 1e-15 < s < 1.0:
            break
    scale = sigma_llr * math.sqrt(-2.0 * math.log(s)) / math.sqrt(s)
    return mu_llr + v1 * scale, mu_llr + v2 * scale

def sfpu_fast_ln_vec(a):
    a = np.maximum(a, 1e-7)
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

def simulate_tensix_batch(inv, batch, mu_val, sigma_val):
    vConstTileId = np.arange(32, dtype=np.uint32)
    tiles = []
    for t in range(8):
        seed_base = np.uint32((0x12345678 + (inv * 1013904223) + batch * 31337 + t * 7919) & 0xFFFFFFFF)
        prng_state = seed_base + vConstTileId
        prng_state = ((prng_state << np.uint32(5)) ^ vConstTileId ^ np.uint32(0x55AA55AA)) & np.uint32(0xFFFFFFFF)
        prng_state[prng_state == 0] = np.uint32(0x13579BDF)

        tile = np.zeros((32, 32), dtype=np.float32)
        for p in range(16):
            prng_state ^= (prng_state << np.uint32(13)) & np.uint32(0xFFFFFFFF)
            prng_state ^= (prng_state >> np.uint32(17)) & np.uint32(0xFFFFFFFF)
            prng_state ^= (prng_state << np.uint32(5)) & np.uint32(0xFFFFFFFF)
            u1_bits = ((prng_state >> np.uint32(9)) & np.uint32(0x007FFFFF)) | np.uint32(0x3F800000)
            u1 = np.maximum(u1_bits.view(np.float32) - 1.0, 1e-7)
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
            tile[2 * p + 0, :] = mu_val + sigma_val * (r * cos_val)

            sin_val = 0.00000276 * s_th - 0.0001984
            sin_val = sin_val * s_th + 0.0083333
            sin_val = sin_val * s_th - 0.1666667
            sin_val = sin_val * s_th + 1.0
            tile[2 * p + 1, :] = mu_val + sigma_val * (r * (sin_val * theta))
        tiles.append(tile)
    return tiles

def simulate_golden_batch(rng, mu_val, sigma_val):
    # Generates 2048 transmitted vnodes across 4 codewords
    # vnode_l1 has 8 tiles (2048 bytes each = 1024 uint16 words = 32x32 floats)
    tiles = [np.zeros((32, 32), dtype=np.float32) for _ in range(8)]
    for v in range(2048):
        l0, l1 = generate_awgn_pair_golden(rng, mu_val, sigma_val)
        l2, l3 = generate_awgn_pair_golden(rng, mu_val, sigma_val)
        
        t = v // 256
        v_loc = v % 256
        row = v_loc // 8
        k = v_loc % 8
        face = (0 if k < 4 else 1) if (row < 16) else (2 if k < 4 else 3)
        col_grp = k if k < 4 else (k - 4)
        
        # In tile matrix layout:
        # face 0: rows 0..15, cols 0..15
        # face 1: rows 0..15, cols 16..31
        # face 2: rows 16..31, cols 0..15
        # face 3: rows 16..31, cols 16..31
        row_in_tile = (row & 15) + (16 if face >= 2 else 0)
        col_base = (col_grp * 4) + (16 if (face % 2 == 1) else 0)
        
        # Word p[0]=l0 (CW0), p[1]=l1 (CW1), p[2]=l2 (CW2), p[3]=l3 (CW3)
        # Note: In each face (16 cols), 4 CWs are interleaved across 4 bytes (2 uint16 pairs)
        # Let's record them into the tile grid
        tiles[t][row_in_tile, col_base + 0] = l0
        tiles[t][row_in_tile, col_base + 1] = l1
        tiles[t][row_in_tile, col_base + 2] = l2
        tiles[t][row_in_tile, col_base + 3] = l3
    return tiles

def extract_vnodes_from_tiles(tiles, cw_idx):
    # Extracts the 2048 transmitted vnode LLRs for a specific codeword (0, 1, 2, or 3)
    vnodes = np.zeros(2048, dtype=np.float32)
    for v in range(2048):
        t = v // 256
        v_loc = v % 256
        row = v_loc // 8
        k = v_loc % 8
        face = (0 if k < 4 else 1) if (row < 16) else (2 if k < 4 else 3)
        col_grp = k if k < 4 else (k - 4)
        row_in_tile = (row & 15) + (16 if face >= 2 else 0)
        col_base = (col_grp * 4) + (16 if (face % 2 == 1) else 0)
        vnodes[v] = tiles[t][row_in_tile, col_base + cw_idx]
    return vnodes

def main():
    eb_n0_db = 2.0
    rate = 0.5
    snr_lin = 10.0 ** (eb_n0_db / 10.0)
    mu_val = float(4.0 * rate * snr_lin)
    sigma_val = float(np.sqrt(8.0 * rate * snr_lin))
    theory_ber = 0.5 * math.erfc(np.sqrt(rate * snr_lin))

    print(f"Theory Benchmark: mu={mu_val:.4f}, sigma={sigma_val:.4f}, BER={theory_ber:.4f}")

    matrix_path = "matrices/AR4JA_r45_4c_128c_r12.chinn.out"
    if not os.path.exists(matrix_path):
        matrix_path = "../../matrices/AR4JA_r45_4c_128c_r12.chinn.out"
    M, N, check_to_var = load_matrix(matrix_path)
    deg6_checks = [check_to_var[c] for c in range(1024)]
    print(f"Loaded matrix: {M} checks, {N} vnodes. Total degree-6 checks: {len(deg6_checks)}")

    N_batches = 300
    print(f"Simulating {N_batches} batches for both generators ({N_batches * 8192:,} symbols each)...")

    # 1. Tensix Simulation
    tensix_all_lanes = []
    tensix_all_syms = []
    tensix_deg6_err_counts = np.zeros(7, dtype=np.int64)
    total_tensix_checks = 0

    for b in range(N_batches):
        inv = b // 64
        b_idx = b % 64
        tiles = simulate_tensix_batch(inv, b_idx, mu_val, sigma_val)
        for tile in tiles:
            tensix_all_lanes.append(tile) # shape (32, 32)
            tensix_all_syms.extend(tile.flatten())
        
        # Check node error distribution for CW0 (first CW)
        for cw in range(4):
            vnodes = extract_vnodes_from_tiles(tiles, cw)
            for c_edges in deg6_checks:
                # count negative LLRs on transmitted edges (< 2048)
                errs = sum(1 for v in c_edges if v < 2048 and vnodes[v] < 0.0)
                tensix_deg6_err_counts[errs] += 1
                total_tensix_checks += 1

    tensix_lane_matrix = np.vstack(tensix_all_lanes) # (N_batches * 8 * 32, 32)
    tensix_syms = np.array(tensix_all_syms)

    # 2. Golden Stepper Simulation
    golden_rng = Xoshiro128Plus(seed=1234567)
    golden_all_lanes = []
    golden_all_syms = []
    golden_deg6_err_counts = np.zeros(7, dtype=np.int64)
    total_golden_checks = 0

    for b in range(N_batches):
        tiles = simulate_golden_batch(golden_rng, mu_val, sigma_val)
        for tile in tiles:
            golden_all_lanes.append(tile)
            golden_all_syms.extend(tile.flatten())

        for cw in range(4):
            vnodes = extract_vnodes_from_tiles(tiles, cw)
            for c_edges in deg6_checks:
                errs = sum(1 for v in c_edges if v < 2048 and vnodes[v] < 0.0)
                golden_deg6_err_counts[errs] += 1
                total_golden_checks += 1

    golden_lane_matrix = np.vstack(golden_all_lanes)
    golden_syms = np.array(golden_all_syms)

    # Compute Statistics
    def stats_summary(x, name):
        m = np.mean(x)
        s = np.std(x)
        z = (x - m) / s
        skew = np.mean(z**3)
        kurt = np.mean(z**4) - 3.0
        p_3sig = np.mean(np.abs(z) > 3.0)
        p_4sig = np.mean(np.abs(z) > 4.0)
        p_neg3 = np.mean(z < -3.0)
        ber = np.mean(x < 0.0)
        return {
            'mean': m, 'std': s, 'skew': skew, 'kurt': kurt,
            'p_3sig': p_3sig, 'p_4sig': p_4sig, 'p_neg3': p_neg3, 'ber': ber
        }

    t_stat = stats_summary(tensix_syms, "Tensix")
    g_stat = stats_summary(golden_syms, "Golden")

    # Cross-Lane Correlation
    t_corr = np.corrcoef(tensix_lane_matrix, rowvar=False)
    np.fill_diagonal(t_corr, 0.0)
    g_corr = np.corrcoef(golden_lane_matrix, rowvar=False)
    np.fill_diagonal(g_corr, 0.0)

    N_samples = tensix_lane_matrix.shape[0]
    three_sig = 3.0 / np.sqrt(N_samples)

    t_tri = np.abs(t_corr[np.triu_indices(32, 1)])
    g_tri = np.abs(g_corr[np.triu_indices(32, 1)])

    # Output detailed report
    print("\n" + "="*80)
    print("COMPARATIVE NOISE GENERATOR BENCHMARK REPORT")
    print("="*80)
    print(f"Total Batches: {N_batches} | Symbols per Generator: {len(tensix_syms):,} | Lane Samples: {N_samples:,}")
    print(f"3-Sigma Correlation Threshold: {three_sig:.6f}\n")

    print(f"{'Metric':<30} | {'Theory (Ideal)':<16} | {'Tensix (SFPU)':<16} | {'Golden (Stepper)':<16}")
    print("-" * 84)
    print(f"{'Mean LLR (mu)':<30} | {mu_val:<16.4f} | {t_stat['mean']:<16.4f} | {g_stat['mean']:<16.4f}")
    print(f"{'Std Dev LLR (sigma)':<30} | {sigma_val:<16.4f} | {t_stat['std']:<16.4f} | {g_stat['std']:<16.4f}")
    print(f"{'Uncoded BER':<30} | {theory_ber:<16.4f} | {t_stat['ber']:<16.4f} | {g_stat['ber']:<16.4f}")
    print(f"{'Skewness (gamma1)':<30} | {0.0:<16.4f} | {t_stat['skew']:<16.4f} | {g_stat['skew']:<16.4f}")
    print(f"{'Excess Kurtosis (gamma2)':<30} | {0.0:<16.4f} | {t_stat['kurt']:<16.4f} | {g_stat['kurt']:<16.4f}")
    print(f"{'P(|Z| > 3.0)':<30} | {0.002700:<16.6f} | {t_stat['p_3sig']:<16.6f} | {g_stat['p_3sig']:<16.6f}")
    print(f"{'P(|Z| > 4.0)':<30} | {0.000063:<16.6f} | {t_stat['p_4sig']:<16.6f} | {g_stat['p_4sig']:<16.6f}")
    print(f"{'P(Z < -3.0)':<30} | {0.001350:<16.6f} | {t_stat['p_neg3']:<16.6f} | {g_stat['p_neg3']:<16.6f}")
    print("-" * 84)
    print(f"{'Max Cross-Lane |rho|':<30} | {'< ' + f'{three_sig:.4f}':<16} | {np.max(t_tri):<16.4f} | {np.max(g_tri):<16.4f}")
    print(f"{'Mean Cross-Lane |rho|':<30} | {'0.0000':<16} | {np.mean(t_tri):<16.4f} | {np.mean(g_tri):<16.4f}")
    print(f"{'Lane Pairs > 3*sigma':<30} | {'0.27% (1 pair)':<16} | {f'{np.mean(t_tri > three_sig)*100:.1f}% ({np.sum(t_tri > three_sig)})':<16} | {f'{np.mean(g_tri > three_sig)*100:.1f}% ({np.sum(g_tri > three_sig)})':<16}")
    print(f"{'Lane Pairs > 0.05':<30} | {'0 (0.0%)':<16} | {f'{np.sum(t_tri > 0.05)} pairs':<16} | {f'{np.sum(g_tri > 0.05)} pairs':<16}")
    print(f"{'Lane Pairs > 0.10':<30} | {'0 (0.0%)':<16} | {f'{np.sum(t_tri > 0.10)} pairs':<16} | {f'{np.sum(g_tri > 0.10)} pairs':<16}")
    print("=" * 84)

    # Check Node Joint Error Distribution Table
    p = theory_ber
    print("\nCHECK-NODE JOINT CHANNEL ERROR DISTRIBUTION P(k) (Degree-6 Checks):")
    print(f"{'Errors k':<10} | {'Theory Binomial':<18} | {'Tensix (SFPU)':<18} | {'Golden (Stepper)':<18} | {'Tensix/Theory Ratio':<20}")
    print("-" * 90)
    for k in range(7):
        n_comb = math.comb(6, k)
        th_pk = n_comb * (p**k) * ((1.0 - p)**(6 - k))
        t_pk = tensix_deg6_err_counts[k] / total_tensix_checks
        g_pk = golden_deg6_err_counts[k] / total_golden_checks
        ratio = t_pk / th_pk if th_pk > 0 else 0.0
        print(f"k = {k:<6} | {th_pk:<18.6f} | {t_pk:<18.6f} | {g_pk:<18.6f} | {ratio:<20.4f}")
    print("=" * 90)

    # Print clusters k >= 3
    t_k3 = np.sum(tensix_deg6_err_counts[3:]) / total_tensix_checks
    g_k3 = np.sum(golden_deg6_err_counts[3:]) / total_golden_checks
    th_k3 = sum(math.comb(6, k) * (p**k) * ((1.0 - p)**(6 - k)) for k in range(3, 7))
    print(f"\nSevere Error Clusters (k >= 3 simultaneous check errors):")
    print(f"  Theory: {th_k3:.6f} | Tensix: {t_k3:.6f} (Ratio: {t_k3/th_k3:.4f}) | Golden: {g_k3:.6f} (Ratio: {g_k3/th_k3:.4f})")

if __name__ == '__main__':
    main()
