#!/usr/bin/env python3
import struct
import math
import numpy as np

def bf16_to_fp32(u):
    return struct.unpack('>f', struct.pack('>I', int(u) << 16))[0]

raw = np.fromfile("dump/tensix_initial_vnodes.bin", dtype=np.uint16)
print(f"Total elements loaded: {len(raw)} words (expected 10240)")

tx_raw = raw[:8192]
punct_raw = raw[8192:]
print(f"Transmitted words: {len(tx_raw)}, Punctured words: {len(punct_raw)}")
print(f"Punctured non-zero words: {np.count_nonzero(punct_raw)} (expected 0)")

vals = np.array([bf16_to_fp32(x) for x in tx_raw])
empirical_mean = np.mean(vals)
empirical_std  = np.std(vals)
neg_count      = np.count_nonzero(vals < 0)
uncoded_ber    = neg_count / len(vals)

print(f"\n--- EMPIRICAL TENSIX NOISE STATISTICS ---")
print(f"Mean LLR:        {empirical_mean:.6f}")
print(f"Std LLR:         {empirical_std:.6f}")
print(f"Min LLR:         {np.min(vals):.4f}")
print(f"Max LLR:         {np.max(vals):.4f}")
print(f"Negative LLRs:   {neg_count} / {len(vals)}")
print(f"Uncoded BER:     {uncoded_ber:.6f}")

# Theory for Eb/N0 = 2.0 dB:
eb_n0_db = 2.0
eb_n0_lin = 10.0 ** (eb_n0_db / 10.0) # ~1.584893
rate = 1024.0 / 2048.0 # Rate 1/2
es_n0_lin = rate * eb_n0_lin # ~0.7924466 (Es/N0 = -1.01 dB)
noise_var = 1.0 / (2.0 * es_n0_lin)
noise_std = np.sqrt(noise_var)
theory_mu = 2.0 / noise_var # 4 * Es/N0 = 3.169786
theory_sigma = 2.0 / noise_std # 2.517850
theory_ber = 0.5 * math.erfc(np.sqrt(es_n0_lin))

print(f"\n--- THEORETICAL BPSK AWGN BENCHMARK (Eb/N0 = 2.0 dB, Rate 1/2) ---")
print(f"Theory Mean LLR: {theory_mu:.6f} (Diff: {empirical_mean - theory_mu:+.6f})")
print(f"Theory Std LLR:  {theory_sigma:.6f} (Diff: {empirical_std - theory_sigma:+.6f})")
print(f"Theory BER:      {theory_ber:.6f} (Empirical: {uncoded_ber:.6f}, Ratio: {uncoded_ber/theory_ber:.4f})")
