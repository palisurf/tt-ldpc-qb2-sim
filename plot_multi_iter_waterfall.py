import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_16iter = "Results/results_AR4JA_r45_4c_128c_r12_amin_clipL4.2R0.0_ebn0_1.60_2.60_step0.10_max2275000_min25_iter16_440cores.csv"
csv_uncl_16iter = "Results/results_AR4JA_r45_4c_128c_r12_amin_ebn0_0.00_3.00_step0.10_max2275000_min25_iter16_440cores.csv"
csv_100iter = "Results/results_AR4JA_r45_4c_128c_r12_amin_clipL4.2R0.0_ebn0_1.20_2.60_step0.10_max2275000_min25_iter100_440cores.csv"
csv_200iter = "Results/results_AR4JA_r45_4c_128c_r12_amin_clipL4.2R0.0_ebn0_1.20_2.60_step0.10_max2275000_min25_iter200_440cores.csv"
output_plot = "Results/waterfall_iter_comparison_200_100_16.png"

df_16 = pd.read_csv(csv_16iter) if os.path.exists(csv_16iter) else None
df_uncl = pd.read_csv(csv_uncl_16iter) if os.path.exists(csv_uncl_16iter) else None
df_100 = pd.read_csv(csv_100iter) if os.path.exists(csv_100iter) else None
df_200 = pd.read_csv(csv_200iter) if os.path.exists(csv_200iter) else None

if df_200 is not None:
    df_200 = df_200[df_200['EbN0_dB'] <= 2.30].copy()

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5), dpi=300)

# Colors
c_uncl = '#95a5a6'
c_16 = '#3498db'
c_100 = '#e67e22'
c_200 = '#8e44ad'

# Panel 1: Waterfall Curves (FER & BER)
if df_uncl is not None:
    sub = df_uncl[(df_uncl['EbN0_dB'] >= 1.2) & (df_uncl['EbN0_dB'] <= 2.7)]
    ax1.semilogy(sub['EbN0_dB'], sub['FER'], 'x:', color=c_uncl, linewidth=1.2, markersize=4, label='16 Iter Unclipped (FER)', alpha=0.6)

if df_16 is not None:
    ax1.semilogy(df_16['EbN0_dB'], df_16['FER'], 's--', color=c_16, linewidth=1.6, markersize=5, label='16 Iter Clipped (FER)')
    ax1.semilogy(df_16['EbN0_dB'], df_16['BER'], '^:', color='#85c1e9', linewidth=1.2, markersize=4, label='16 Iter Clipped (BER)')

if df_100 is not None:
    ax1.semilogy(df_100['EbN0_dB'], df_100['FER'], 'o-', color=c_100, linewidth=2.0, markersize=6, label='100 Iter Clipped (FER)')
    ax1.semilogy(df_100['EbN0_dB'], df_100['BER'], 'v--', color='#f8c471', linewidth=1.4, markersize=4, label='100 Iter Clipped (BER)')

if df_200 is not None:
    ax1.semilogy(df_200['EbN0_dB'], df_200['FER'], 'D-', color='#8e44ad', linewidth=2.4, markersize=7, label=r'$\mathbf{200\ Iter\ Clipped\ (FER)}$')
    ax1.semilogy(df_200['EbN0_dB'], df_200['BER'], 'p--', color='#d2b4de', linewidth=1.8, markersize=6, label=r'$\mathbf{200\ Iter\ Clipped\ (BER)}$')

    # Annotate deep point (2.30 dB)
    last_pt = df_200.iloc[-1]
    ax1.annotate(f"FER: {last_pt['FER']:.2e}\nBER: {last_pt['BER']:.2e}\n({int(last_pt['Total_Blocks']/1e6)}M blocks)",
                 xy=(last_pt['EbN0_dB'], last_pt['FER']),
                 xytext=(15, 10), textcoords="offset points",
                 fontweight='bold', fontsize=9.5, color='#8e44ad',
                 arrowprops=dict(arrowstyle="->", color='#8e44ad', lw=1.4))

ax1.set_xlabel(r'Signal-to-Noise Ratio $E_b/N_0$ [dB]', fontsize=12, fontweight='bold')
ax1.set_ylabel('Error Rate (Log Scale)', fontsize=12, fontweight='bold')
ax1.set_title('CCSDS AR4JA Rate-1/2: Ultra-Deep Waterfall\n(16 vs. 100 vs. 200 Max Iterations on 440 Tensix Cores)', fontsize=13, fontweight='bold')
ax1.set_xlim(1.15, 2.7)
ax1.set_ylim(1e-9, 0.3)
ax1.grid(True, which='both', linestyle='--', alpha=0.5)
ax1.legend(loc='lower left', frameon=True, fancybox=True, shadow=True, fontsize=9.5)

# Panel 2: Comparative Improvement (Relative to 16 Iterations)
if df_16 is not None and df_200 is not None:
    merged = pd.merge(df_200, df_16, on='EbN0_dB', suffixes=('_200', '_16'))
    if not merged.empty:
        fer_ratio_200 = merged['FER_16'] / merged['FER_200']
        ber_ratio_200 = merged['BER_16'] / merged['BER_200']

        ax2.axhline(1.0, color='black', linestyle='-', linewidth=1.5, alpha=0.7, label='16-Iter Baseline (1.0x)')
        ax2.plot(merged['EbN0_dB'], fer_ratio_200, 'D-', color='#8e44ad', linewidth=2.2, markersize=7, label=r'200-Iter Gain ($\mathrm{FER}_{16} / \mathrm{FER}_{200}$)')
        
        if df_100 is not None:
            m100 = pd.merge(df_100, df_16, on='EbN0_dB', suffixes=('_100', '_16'))
            if not m100.empty:
                r100 = m100['FER_16'] / m100['FER_100']
                ax2.plot(m100['EbN0_dB'], r100, 'o--', color=c_100, linewidth=1.8, markersize=5, label=r'100-Iter Gain ($\mathrm{FER}_{16} / \mathrm{FER}_{100}$)')

        for _, row in merged.iterrows():
            ratio = row['FER_16'] / row['FER_200']
            ax2.annotate(f"{ratio:.1f}x", 
                         xy=(row['EbN0_dB'], ratio),
                         xytext=(0, 8), textcoords="offset points",
                         ha='center', fontweight='bold', fontsize=9, color='#8e44ad')

        ax2.set_yscale('log')
        ax2.set_xlabel(r'Signal-to-Noise Ratio $E_b/N_0$ [dB]', fontsize=12, fontweight='bold')
        ax2.set_ylabel('FER Error Reduction Factor (Log Scale)', fontsize=12, fontweight='bold')
        ax2.set_title('Coding Gain vs. 16-Iteration Baseline\n(Values > 1.0 indicate performance improvement)', fontsize=13, fontweight='bold')
        ax2.set_xlim(1.55, 2.35)
        ax2.grid(True, which='both', linestyle='--', alpha=0.5)
        ax2.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=10)

plt.tight_layout()
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"[PLOT] Multi-iteration comparison plot saved to {output_plot}")
