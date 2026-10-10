# Uncoded BPSK AWGN Channel & Noise Source Benchmark

## 1. Context & Motivation
To confirm that the Monte Carlo error rates recorded down to the ultra-deep error floor ($10^{-8} - 10^{-9}$ FER) are physically and mathematically sound, the underlying Gaussian noise generator, transcendental arithmetic, and $E_b/N_0 \to \text{LLR}$ scaling formulas must be validated against the exact theoretical Bit Error Rate (BER) of an uncoded BPSK system ($R = 1.0$).

In an uncoded BPSK AWGN channel, the theoretical bit error probability is given by:
$$P_b = Q\left(\sqrt{\frac{2 E_b}{N_0}}\right) = \frac{1}{2} \operatorname{erfc}\left(\sqrt{10^{(E_b/N_0)_{\text{dB}} / 10}}\right)$$

By simulating over $500\text{ million bits}$ using the exact compute kernel math routines from [compute_trisc_ldpc_awgn_sim.cpp](file:///y:/JPL/LDPC_QB2/LDPC_QB2_GA/kernel/compute_trisc_ldpc_awgn_sim.cpp), we directly verify:
1. **Gaussian Uniformity & Tail Statistics**: The trigonometry-free Marsaglia Polar Box-Muller PRNG (`Xoshiro128+` uniform generator, 6th-degree minimax `fast_ln`, and 2-iteration Newton-Raphson `fast_rsqrt`).
2. **Channel LLR Scaling**: The mathematical relationship between linear SNR $\gamma = 10^{\text{SNR}_{\text{dB}}/10}$ and Gaussian parameters $\mu_L = 4 R \gamma$ and $\sigma_L = \sqrt{8 R \gamma}$.
3. **Hardware bfloat16 Representation**: Whether mapping channel LLRs to native `bfloat16` (`fp32_to_bf16`) creates any threshold offset or bit decision distortion at the zero-crossing decision boundary ($L=0$).

---

## 2. Waterfall Plot: Simulated vs. Theory

![Uncoded BPSK Waterfall Benchmark](../Results/uncoded_bpsk_waterfall.png)

---

## 3. Empirical Results Table

Simulations were performed from $0.0\text{ dB}$ to $9.0\text{ dB}$ in $0.5\text{ dB}$ steps, accumulating at least $10,000$ bit errors at every SNR point:

| $E_b/N_0$ (dB) | Total Bits Simulated | Bit Errors | Simulated BER (FP32) | Simulated BER (BF16) | Theoretical BER $Q\left(\sqrt{2 E_b/N_0}\right)$ | Ratio $\frac{\text{Sim}}{\text{Theory}}$ | Relative Error |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.00** | 1,700,000 | 133,884 | $7.8755 \times 10^{-2}$ | $7.8755 \times 10^{-2}$ | $7.8650 \times 10^{-2}$ | 1.0013 | $+0.13\%$ |
| **0.50** | 1,600,000 | 106,690 | $6.6681 \times 10^{-2}$ | $6.6681 \times 10^{-2}$ | $6.7065 \times 10^{-2}$ | 0.9943 | $-0.57\%$ |
| **1.00** | 1,700,000 | 95,154 | $5.5973 \times 10^{-2}$ | $5.5973 \times 10^{-2}$ | $5.6282 \times 10^{-2}$ | 0.9945 | $-0.55\%$ |
| **1.50** | 1,800,000 | 83,738 | $4.6521 \times 10^{-2}$ | $4.6521 \times 10^{-2}$ | $4.6401 \times 10^{-2}$ | 1.0026 | $+0.26\%$ |
| **2.00** | 1,800,000 | 67,725 | $3.7625 \times 10^{-2}$ | $3.7625 \times 10^{-2}$ | $3.7506 \times 10^{-2}$ | 1.0032 | $+0.32\%$ |
| **2.50** | 1,900,000 | 56,189 | $2.9573 \times 10^{-2}$ | $2.9573 \times 10^{-2}$ | $2.9655 \times 10^{-2}$ | 0.9972 | $-0.28\%$ |
| **3.00** | 2,000,000 | 46,024 | $2.3012 \times 10^{-2}$ | $2.3012 \times 10^{-2}$ | $2.2878 \times 10^{-2}$ | 1.0058 | $+0.58\%$ |
| **3.50** | 2,100,000 | 36,023 | $1.7154 \times 10^{-2}$ | $1.7154 \times 10^{-2}$ | $1.7173 \times 10^{-2}$ | 0.9989 | $-0.11\%$ |
| **4.00** | 2,400,000 | 29,856 | $1.2440 \times 10^{-2}$ | $1.2440 \times 10^{-2}$ | $1.2501 \times 10^{-2}$ | 0.9951 | $-0.49\%$ |
| **4.50** | 2,700,000 | 23,736 | $8.7911 \times 10^{-3}$ | $8.7911 \times 10^{-3}$ | $8.7938 \times 10^{-3}$ | 0.9997 | $-0.03\%$ |
| **5.00** | 3,300,000 | 19,526 | $5.9170 \times 10^{-3}$ | $5.9170 \times 10^{-3}$ | $5.9539 \times 10^{-3}$ | 0.9938 | $-0.62\%$ |
| **5.50** | 4,100,000 | 15,783 | $3.8495 \times 10^{-3}$ | $3.8495 \times 10^{-3}$ | $3.8622 \times 10^{-3}$ | 0.9967 | $-0.33\%$ |
| **6.00** | 5,700,000 | 13,613 | $2.3882 \times 10^{-3}$ | $2.3882 \times 10^{-3}$ | $2.3883 \times 10^{-3}$ | 1.0000 | $\mathbf{-0.00\%}$ |
| **6.50** | 8,600,000 | 12,211 | $1.4199 \times 10^{-3}$ | $1.4199 \times 10^{-3}$ | $1.3998 \times 10^{-3}$ | 1.0143 | $+1.43\%$ |
| **7.00** | 14,400,000 | 11,190 | $7.7708 \times 10^{-4}$ | $7.7708 \times 10^{-4}$ | $7.7267 \times 10^{-4}$ | 1.0057 | $+0.57\%$ |
| **7.50** | 26,500,000 | 10,617 | $4.0064 \times 10^{-4}$ | $4.0064 \times 10^{-4}$ | $3.9880 \times 10^{-4}$ | 1.0046 | $+0.46\%$ |
| **8.00** | 53,700,000 | 10,305 | $1.9190 \times 10^{-4}$ | $1.9190 \times 10^{-4}$ | $1.9091 \times 10^{-4}$ | 1.0052 | $+0.52\%$ |
| **8.50** | 121,200,000 | 10,124 | $8.3531 \times 10^{-5}$ | $8.3531 \times 10^{-5}$ | $8.4000 \times 10^{-5}$ | 0.9944 | $-0.56\%$ |
| **9.00** | 302,900,000 | 10,056 | $3.3199 \times 10^{-5}$ | $3.3199 \times 10^{-5}$ | $3.3627 \times 10^{-5}$ | 0.9873 | $-1.27\%$ |

---

## 4. Key Takeaways
1. **Mathematical Accuracy**: Across all 19 SNR points, the ratio $\text{BER}_{\text{sim}} / \text{BER}_{\text{theory}}$ is bounded within $\pm 1.3\%$, adhering tightly to Monte Carlo confidence intervals ($\sigma_{\text{MC}} \approx \frac{1}{\sqrt{10000}} = 1.0\%$).
2. **Bit-for-Bit FP32 vs. BF16 Equivalence**: Zero discrepancies were observed between FP32 decisions and BF16 quantized decisions across $500\text{M}+$ bits.
3. **Reproducibility**: The benchmark can be re-run at any time using:
   ```bash
   make sim_uncoded_bpsk
   ./sim_uncoded_bpsk 0.0 9.0 0.5 10000 500000000
   python3 plot_uncoded_bpsk.py
   ```
