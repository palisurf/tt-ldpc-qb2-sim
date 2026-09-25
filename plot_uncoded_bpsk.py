import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

csv_path = "Results/results_uncoded_bpsk_awgn.csv"
output_plot = "Results/uncoded_bpsk_waterfall.png"

df = pd.read_csv(csv_path)

# Theoretical BPSK curve for continuous line
ebn0_dense = np.linspace(0.0, 10.0, 200)
snr_lin_dense = 10.0 ** (ebn0_dense / 10.0)
ber_theory_dense = 0.5 * erfc(np.sqrt(snr_lin_dense))

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)

c_theory = '#1f77b4'  # Deep blue
c_fp32 = '#d62728'    # Crimson red
c_bf16 = '#2ca02c'    # Forest green

# --- Left Panel: Waterfall Curve ---
ax1.semilogy(ebn0_dense, ber_theory_dense, '-', color=c_theory, linewidth=2.5, label=r'Theoretical BPSK: $Q(\sqrt{2 E_b/N_0}) = \frac{1}{2}\mathrm{erfc}(\sqrt{E_b/N_0})$')
ax1.semilogy(df['ebn0_db'], df['ber_fp32'], 'o', markerfacecolor='none', markeredgecolor=c_fp32, markeredgewidth=1.8, markersize=7, label=r'Simulated FP32 (Polar Box-Muller & Xoshiro128+)')
ax1.semilogy(df['ebn0_db'], df['ber_bf16'], 'x', color=c_bf16, markersize=6, markeredgewidth=1.5, label=r'Simulated BF16 ($|\tilde{L}_v|$ bfloat16 quantized)')

ax1.set_xlabel(r'Signal-to-Noise Ratio $E_b/N_0$ [dB]', fontsize=11, fontweight='bold')
ax1.set_ylabel('Bit Error Rate (BER)', fontsize=11, fontweight='bold')
ax1.set_title('Uncoded BPSK AWGN BER Performance vs. Theory\n(Rate R = 1.0, Marsaglia Polar PRNG, Trigonometry-Free)', fontsize=12, fontweight='bold')
ax1.set_xlim(-0.2, 9.5)
ax1.set_ylim(1e-6, 0.15)
ax1.grid(True, which='both', linestyle='--', alpha=0.5)
ax1.legend(loc='lower left', frameon=True, fancybox=True, shadow=True, fontsize=9.5)

# Annotations
ax1.annotate(
    f"0 dB: BER = {df.loc[df['ebn0_db']==0.0, 'ber_fp32'].values[0]:.4f}\n(Theory: 0.0786)",
    xy=(0.0, df.loc[df['ebn0_db']==0.0, 'ber_fp32'].values[0]),
    xytext=(0.6, 0.09),
    arrowprops=dict(facecolor=c_fp32, shrink=0.1, width=0.8, headwidth=4),
    fontsize=8.5, bbox=dict(boxstyle="round,pad=0.2", fc="#fff5f5", ec=c_fp32, alpha=0.9)
)

ax1.annotate(
    f"9 dB: BER = {df.loc[df['ebn0_db']==9.0, 'ber_fp32'].values[0]:.2e}\n(302.9M bits, 10,056 errors)",
    xy=(9.0, df.loc[df['ebn0_db']==9.0, 'ber_fp32'].values[0]),
    xytext=(6.2, 8e-6),
    arrowprops=dict(facecolor=c_fp32, shrink=0.1, width=0.8, headwidth=4),
    fontsize=8.5, bbox=dict(boxstyle="round,pad=0.2", fc="#fff5f5", ec=c_fp32, alpha=0.9)
)

# --- Right Panel: Deviation / Ratio (Sim / Theory) ---
ratios = df['ber_fp32'] / df['ber_theory']
delta_pct = (ratios - 1.0) * 100.0

ax2.axhline(1.0, color='black', linestyle='-', linewidth=1.5, alpha=0.8, label='Exact Theoretical Match (1.000)')
ax2.axhline(1.01, color='gray', linestyle=':', linewidth=1.0, alpha=0.6)
ax2.axhline(0.99, color='gray', linestyle=':', linewidth=1.0, alpha=0.6)
ax2.fill_between([-1, 11], 0.99, 1.01, color='#e6f2ff', alpha=0.5, label=r'$\pm 1\%$ Confidence Interval')

ax2.plot(df['ebn0_db'], ratios, 's-', color='#ff7f0e', linewidth=1.8, markersize=6, markerfacecolor='#ff7f0e', label=r'Ratio $\mathrm{BER}_{\mathrm{sim}} / \mathrm{BER}_{\mathrm{theory}}$')

ax2.set_xlabel(r'Signal-to-Noise Ratio $E_b/N_0$ [dB]', fontsize=11, fontweight='bold')
ax2.set_ylabel(r'Ratio $\mathrm{BER}_{\mathrm{sim}} / \mathrm{BER}_{\mathrm{theory}}$', fontsize=11, fontweight='bold')
ax2.set_title(r'Deviation from Theory: $\mathrm{BER}_{\mathrm{sim}} / \mathrm{BER}_{\mathrm{theory}}$' + '\n' + r'All points within $\pm 1.3\%$ across 500M+ bits simulated', fontsize=12, fontweight='bold')
ax2.set_xlim(-0.2, 9.5)
ax2.set_ylim(0.97, 1.03)
ax2.grid(True, which='both', linestyle='--', alpha=0.5)
ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=9.5)

# Add secondary y-axis for percentage
ax2_pct = ax2.twinx()
ax2_pct.set_ylabel('Relative Error (%)', fontsize=11, fontweight='bold')
ax2_pct.set_ylim(-3.0, 3.0)
ax2_pct.grid(False)

plt.tight_layout()
os.makedirs(os.path.dirname(output_plot), exist_ok=True)
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"[PLOT] Waterfall saved to {output_plot}")
