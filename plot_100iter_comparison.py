import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_16iter = "Results/results_AR4JA_r45_4c_128c_r12_amin_clipL4.2R0.0_ebn0_1.60_2.60_step0.10_max2275000_min25_iter16_440cores.csv"
csv_100iter = "Results/results_AR4JA_r45_4c_128c_r12_amin_clipL4.2R0.0_ebn0_1.20_2.60_step0.10_max2275000_min25_iter100_440cores.csv"
csv_uncl_16iter = "Results/results_AR4JA_r45_4c_128c_r12_amin_ebn0_0.00_3.00_step0.10_max2275000_min25_iter16_440cores.csv"
output_plot = "Results/waterfall_100iter_vs_16iter.png"

df_16 = pd.read_csv(csv_16iter) if os.path.exists(csv_16iter) else None
df_100 = pd.read_csv(csv_100iter) if os.path.exists(csv_100iter) else None
df_uncl = pd.read_csv(csv_uncl_16iter) if os.path.exists(csv_uncl_16iter) else None

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

# Colors
c_uncl = '#7f7f7f'
c_16 = '#1f77b4'
c_100 = '#d62728'

# Panel 1: Waterfall Curves (FER & BER)
if df_uncl is not None:
    sub_uncl = df_uncl[(df_uncl['EbN0_dB'] >= 1.2) & (df_uncl['EbN0_dB'] <= 2.7)]
    ax1.semilogy(sub_uncl['EbN0_dB'], sub_uncl['FER'], 'o:', color=c_uncl, linewidth=1.5, markersize=4, label='16 Iter Unclipped (FER)', alpha=0.7)
    ax1.semilogy(sub_uncl['EbN0_dB'], sub_uncl['BER'], 'x:', color=c_uncl, linewidth=1.2, markersize=3, label='16 Iter Unclipped (BER)', alpha=0.5)

if df_16 is not None:
    ax1.semilogy(df_16['EbN0_dB'], df_16['FER'], 's--', color=c_16, linewidth=1.8, markersize=5, label='16 Iter Clipped (FER)')
    ax1.semilogy(df_16['EbN0_dB'], df_16['BER'], '^--', color='#aec7e8', linewidth=1.5, markersize=4, label='16 Iter Clipped (BER)')

if df_100 is not None:
    ax1.semilogy(df_100['EbN0_dB'], df_100['FER'], 'o-', color=c_100, linewidth=2.4, markersize=7, label=r'$\mathbf{100\ Iter\ Clipped\ (FER)}$')
    ax1.semilogy(df_100['EbN0_dB'], df_100['BER'], 'D-', color='#ff7f0e', linewidth=1.8, markersize=5, label=r'$\mathbf{100\ Iter\ Clipped\ (BER)}$')

ax1.set_xlabel(r'Signal-to-Noise Ratio $E_b/N_0$ [dB]', fontsize=12, fontweight='bold')
ax1.set_ylabel('Error Rate (Log Scale)', fontsize=12, fontweight='bold')
ax1.set_title('CCSDS AR4JA Rate-1/2: 100 Iterations vs. 16 Iterations\n(Approximate-Min* Decoder on 440 Tensix Cores)', fontsize=13, fontweight='bold')
ax1.set_xlim(1.15, 2.7)
ax1.set_ylim(1e-8, 0.3)
ax1.grid(True, which='both', linestyle='--', alpha=0.5)
ax1.legend(loc='lower left', frameon=True, fancybox=True, shadow=True, fontsize=10)

# Panel 2: Coding Gain & Improvement Factor
if df_16 is not None and df_100 is not None:
    merged = pd.merge(df_100, df_16, on='EbN0_dB', suffixes=('_100', '_16'))
    if not merged.empty:
        fer_ratio = merged['FER_16'] / merged['FER_100']  # How many times better 100 iter is
        ber_ratio = merged['BER_16'] / merged['BER_100']

        ax2.axhline(1.0, color='black', linestyle='-', linewidth=1.5, alpha=0.7, label='16-Iter Baseline (1.0x)')
        ax2.plot(merged['EbN0_dB'], fer_ratio, 'o-', color=c_100, linewidth=2.2, markersize=7, label=r'FER Improvement Factor ($\mathrm{FER}_{16} / \mathrm{FER}_{100}$)')
        ax2.plot(merged['EbN0_dB'], ber_ratio, 's--', color='#ff7f0e', linewidth=2.0, markersize=6, label=r'BER Improvement Factor ($\mathrm{BER}_{16} / \mathrm{BER}_{100}$)')

        for _, row in merged.iterrows():
            ax2.annotate(f"{row['FER_16']/row['FER_100']:.1f}x", 
                         xy=(row['EbN0_dB'], row['FER_16']/row['FER_100']),
                         xytext=(0, 8), textcoords="offset points",
                         ha='center', fontweight='bold', fontsize=9, color=c_100)

        ax2.set_yscale('log')
        ax2.set_xlabel(r'Signal-to-Noise Ratio $E_b/N_0$ [dB]', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Improvement Factor (Log Scale)', fontsize=12, fontweight='bold')
        ax2.set_title('Error Reduction Factor from 100 Iterations\n(Values > 1.0 indicate performance gain over 16 iters)', fontsize=13, fontweight='bold')
        ax2.set_xlim(1.55, 2.25)
        ax2.grid(True, which='both', linestyle='--', alpha=0.5)
        ax2.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=10)

plt.tight_layout()
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"[PLOT] 100-iter comparison plot saved to {output_plot}")
