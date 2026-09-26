import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_16iter = "Results/results_AR4JA_r45_4c_128c_r12_amin_clipL4.2R0.0_ebn0_1.60_2.60_step0.10_max2275000_min25_iter16_440cores.csv"
csv_uncl_16iter = "Results/results_AR4JA_r45_4c_128c_r12_amin_ebn0_0.00_3.00_step0.10_max2275000_min25_iter16_440cores.csv"
csv_100iter = "Results/results_AR4JA_r45_4c_128c_r12_amin_clipL4.2R0.0_ebn0_1.20_2.60_step0.10_max2275000_min25_iter100_440cores.csv"
csv_200iter = "Results/results_AR4JA_r45_4c_128c_r12_amin_clipL4.2R0.0_ebn0_1.20_2.60_step0.10_max2275000_min25_iter200_440cores.csv"
csv_200iter_uncl = "Results/results_AR4JA_r45_4c_128c_r12_amin_ebn0_1.20_2.60_step0.10_max2275000_min25_iter200_440cores.csv"
output_plot = "Results/waterfall_iter_comparison_200_100_16.png"

df_16 = pd.read_csv(csv_16iter) if os.path.exists(csv_16iter) else None
df_uncl_16 = pd.read_csv(csv_uncl_16iter) if os.path.exists(csv_uncl_16iter) else None
df_100 = pd.read_csv(csv_100iter) if os.path.exists(csv_100iter) else None
df_200_clip = pd.read_csv(csv_200iter) if os.path.exists(csv_200iter) else None
df_200_uncl = pd.read_csv(csv_200iter_uncl) if os.path.exists(csv_200iter_uncl) else None

if df_200_clip is not None:
    df_200_clip = df_200_clip[df_200_clip['EbN0_dB'] <= 2.30].copy()

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5), dpi=300)

# Colors
c_uncl_16 = '#95a5a6'
c_16 = '#3498db'
c_100 = '#e67e22'
c_200_clip = '#9b59b6'
c_200_uncl = '#27ae60'

# Panel 1: Waterfall Curves (FER & BER)
if df_uncl_16 is not None:
    sub = df_uncl_16[(df_uncl_16['EbN0_dB'] >= 1.2) & (df_uncl_16['EbN0_dB'] <= 2.7)]
    ax1.semilogy(sub['EbN0_dB'], sub['FER'], 'x:', color=c_uncl_16, linewidth=1.2, markersize=4, label='16 Iter Unclipped (FER)', alpha=0.5)

if df_16 is not None:
    ax1.semilogy(df_16['EbN0_dB'], df_16['FER'], 's--', color=c_16, linewidth=1.6, markersize=5, label='16 Iter Clipped (FER)')
    ax1.semilogy(df_16['EbN0_dB'], df_16['BER'], '^:', color='#85c1e9', linewidth=1.2, markersize=4, label='16 Iter Clipped (BER)')

if df_100 is not None:
    ax1.semilogy(df_100['EbN0_dB'], df_100['FER'], 'o-', color=c_100, linewidth=1.8, markersize=5, label='100 Iter Clipped (FER)')

if df_200_clip is not None:
    ax1.semilogy(df_200_clip['EbN0_dB'], df_200_clip['FER'], 'D--', color=c_200_clip, linewidth=1.8, markersize=5, label='200 Iter Clipped L4.2 (FER)')

if df_200_uncl is not None:
    ax1.semilogy(df_200_uncl['EbN0_dB'], df_200_uncl['FER'], 'o-', color=c_200_uncl, linewidth=2.5, markersize=7, label=r'$\mathbf{200\ Iter\ Unclipped\ (FER)}$')
    ax1.semilogy(df_200_uncl['EbN0_dB'], df_200_uncl['BER'], 's--', color='#52be80', linewidth=1.8, markersize=5, label=r'$\mathbf{200\ Iter\ Unclipped\ (BER)}$')

    # Annotate deep point (2.40 dB)
    last_pt = df_200_uncl.iloc[-1]
    ax1.annotate(f"FER: {last_pt['FER']:.2e}\nBER: {last_pt['BER']:.2e}\n({int(last_pt['Total_Blocks']/1e6)}M blocks)",
                 xy=(last_pt['EbN0_dB'], last_pt['FER']),
                 xytext=(15, 8), textcoords="offset points",
                 fontweight='bold', fontsize=9.5, color=c_200_uncl,
                 arrowprops=dict(arrowstyle="->", color=c_200_uncl, lw=1.5))

ax1.set_xlabel(r'Signal-to-Noise Ratio $E_b/N_0$ [dB]', fontsize=12, fontweight='bold')
ax1.set_ylabel('Error Rate (Log Scale)', fontsize=12, fontweight='bold')
ax1.set_title('CCSDS AR4JA Rate-1/2: Ultra-Deep Waterfall\n(16 vs. 100 vs. 200 Max Iterations; Clipped vs. Unclipped)', fontsize=13, fontweight='bold')
ax1.set_xlim(1.15, 2.7)
ax1.set_ylim(1e-9, 0.3)
ax1.grid(True, which='both', linestyle='--', alpha=0.5)
ax1.legend(loc='lower left', frameon=True, fancybox=True, shadow=True, fontsize=9)

# Panel 2: Comparative Coding Gain vs 16-Iter Baseline
if df_16 is not None and df_200_uncl is not None:
    merged_uncl = pd.merge(df_200_uncl, df_16, on='EbN0_dB', suffixes=('_uncl', '_16'))
    if not merged_uncl.empty:
        fer_ratio_uncl = merged_uncl['FER_16'] / merged_uncl['FER_uncl']

        ax2.axhline(1.0, color='black', linestyle='-', linewidth=1.5, alpha=0.7, label='16-Iter Baseline (1.0x)')
        ax2.plot(merged_uncl['EbN0_dB'], fer_ratio_uncl, 'o-', color=c_200_uncl, linewidth=2.4, markersize=7, label=r'200-Iter Unclipped Gain ($\mathrm{FER}_{16} / \mathrm{FER}_{\mathrm{uncl}}$)')

        if df_200_clip is not None:
            m_clip = pd.merge(df_200_clip, df_16, on='EbN0_dB', suffixes=('_clip', '_16'))
            if not m_clip.empty:
                r_clip = m_clip['FER_16'] / m_clip['FER_clip']
                ax2.plot(m_clip['EbN0_dB'], r_clip, 'D--', color=c_200_clip, linewidth=1.8, markersize=5, label=r'200-Iter Clipped Gain')

        if df_100 is not None:
            m100 = pd.merge(df_100, df_16, on='EbN0_dB', suffixes=('_100', '_16'))
            if not m100.empty:
                r100 = m100['FER_16'] / m100['FER_100']
                ax2.plot(m100['EbN0_dB'], r100, '^:', color=c_100, linewidth=1.5, markersize=4, label=r'100-Iter Clipped Gain')

        for _, row in merged_uncl.iterrows():
            ratio = row['FER_16'] / row['FER_uncl']
            ax2.annotate(f"{ratio:.1f}x", 
                         xy=(row['EbN0_dB'], ratio),
                         xytext=(0, 8), textcoords="offset points",
                         ha='center', fontweight='bold', fontsize=9, color=c_200_uncl)

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
