import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_16iter = "Results/results_AR4JA_r45_4c_128c_r12_amin_clipL4.2R0.0_ebn0_1.60_2.60_step0.10_max2275000_min25_iter16_440cores.csv"
csv_uncl_16iter = "Results/results_AR4JA_r45_4c_128c_r12_amin_ebn0_0.00_3.00_step0.10_max2275000_min25_iter16_440cores.csv"
csv_100iter = "Results/results_AR4JA_r45_4c_128c_r12_amin_clipL4.2R0.0_ebn0_1.20_2.60_step0.10_max2275000_min25_iter100_440cores.csv"
csv_200iter = "Results/results_AR4JA_r45_4c_128c_r12_amin_clipL4.2R0.0_ebn0_1.20_2.60_step0.10_max2275000_min25_iter200_440cores.csv"
csv_200iter_uncl = "Results/results_AR4JA_r45_4c_128c_r12_amin_ebn0_1.20_2.60_step0.10_max2275000_min25_iter200_440cores.csv"
csv_200iter_recursive = "Results/results_AR4JA_r45_4c_128c_r12_amin_ebn0_1.20_2.20_step0.10_max4545454_min25_iter200_440cores.csv"
output_plot = "Results/waterfall_iter_comparison_200_100_16.png"

df_16 = pd.read_csv(csv_16iter) if os.path.exists(csv_16iter) else None
df_uncl_16 = pd.read_csv(csv_uncl_16iter) if os.path.exists(csv_uncl_16iter) else None
df_100 = pd.read_csv(csv_100iter) if os.path.exists(csv_100iter) else None
df_200_clip = pd.read_csv(csv_200iter) if os.path.exists(csv_200iter) else None
df_200_uncl = pd.read_csv(csv_200iter_uncl) if os.path.exists(csv_200iter_uncl) else None
df_200_rec = pd.read_csv(csv_200iter_recursive) if os.path.exists(csv_200iter_recursive) else None

if df_200_clip is not None:
    df_200_clip = df_200_clip[df_200_clip['EbN0_dB'] <= 2.30].copy()

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 8), dpi=300)

# Colors
c_uncl_16 = '#95a5a6'
c_16 = '#3498db'
c_100 = '#e67e22'
c_200_clip = '#9b59b6'
c_200_uncl = '#27ae60'
c_rec_fer = '#d63031'   # Crimson Red for True Recursive AMin*
c_rec_ber = '#e17055'   # Terracotta Red for BER

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
    valid_uncl = df_200_uncl[df_200_uncl['FER'] > 0]
    zero_uncl = df_200_uncl[df_200_uncl['FER'] == 0]

    ax1.semilogy(valid_uncl['EbN0_dB'], valid_uncl['FER'], 'o-', color=c_200_uncl, linewidth=1.8, markersize=6, label=r'200 Iter 3-Min Heuristic (FER)', alpha=0.7)
    ax1.semilogy(valid_uncl['EbN0_dB'], valid_uncl['BER'], 's--', color='#52be80', linewidth=1.4, markersize=4, label=r'200 Iter 3-Min Heuristic (BER)', alpha=0.7)

# New: True Recursive AMin* (MILCOM 2003)
if df_200_rec is not None:
    ax1.semilogy(df_200_rec['EbN0_dB'], df_200_rec['FER'], 'D-', color=c_rec_fer, linewidth=2.6, markersize=7, label=r'$\mathbf{200\ Iter\ True\ Recursive\ AMin^*\ (FER)}$')
    ax1.semilogy(df_200_rec['EbN0_dB'], df_200_rec['BER'], 'v--', color=c_rec_ber, linewidth=2.0, markersize=6, label=r'$\mathbf{200\ Iter\ True\ Recursive\ AMin^*\ (BER)}$')

    # Annotate 2.00 dB point
    pt_20 = df_200_rec[df_200_rec['EbN0_dB'] == 2.00]
    if not pt_20.empty:
        r20 = pt_20.iloc[0]
        ax1.annotate(f"2.00 dB Recursive:\nFER={r20['FER']:.2e}\n(3.1x lower vs 3-min)",
                     xy=(r20['EbN0_dB'], r20['FER']),
                     xytext=(15, -15), textcoords="offset points",
                     ha='left', fontweight='bold', fontsize=8.5, color=c_rec_fer,
                     arrowprops=dict(arrowstyle="->", color=c_rec_fer, lw=1.5))

    # Annotate 2.10 dB point
    pt_21 = df_200_rec[df_200_rec['EbN0_dB'] == 2.10]
    if not pt_21.empty:
        r21 = pt_21.iloc[0]
        ax1.annotate(f"2.10 dB Recursive:\nFER={r21['FER']:.2e}\n(8.9x lower vs 3-min!)",
                     xy=(r21['EbN0_dB'], r21['FER']),
                     xytext=(-15, -20), textcoords="offset points",
                     ha='right', fontweight='bold', fontsize=8.5, color=c_rec_fer,
                     arrowprops=dict(arrowstyle="->", color=c_rec_fer, lw=1.5))

    # Annotate 2.20 dB point
    pt_22 = df_200_rec[df_200_rec['EbN0_dB'] == 2.20]
    if not pt_22.empty:
        r22 = pt_22.iloc[0]
        ax1.annotate(f"2.20 dB Recursive:\nFER={r22['FER']:.2e}\n(391.2M blocks, 28 FE)",
                     xy=(r22['EbN0_dB'], r22['FER']),
                     xytext=(15, -12), textcoords="offset points",
                     ha='left', fontweight='bold', fontsize=8.5, color=c_rec_fer,
                     arrowprops=dict(arrowstyle="->", color=c_rec_fer, lw=1.5))

    # Annotate 2.40 dB point
    pt_24 = valid_uncl[valid_uncl['EbN0_dB'] == 2.40]
    if not pt_24.empty:
        r24 = pt_24.iloc[0]
        ax1.annotate(f"2.40 dB: FER={r24['FER']:.2e}\n(1.001B blocks, 24 FE)",
                     xy=(r24['EbN0_dB'], r24['FER']),
                     xytext=(-15, 12), textcoords="offset points",
                     ha='right', fontweight='bold', fontsize=9, color=c_200_uncl,
                     arrowprops=dict(arrowstyle="->", color=c_200_uncl, lw=1.5))

    # Plot & Annotate 2.50 dB Zero-Error Upper Bound
    if not zero_uncl.empty:
        r25 = zero_uncl.iloc[0]
        bound_fer = 1.0 / r25['Total_Blocks'] # 9.99e-10
        ax1.semilogy(r25['EbN0_dB'], bound_fer, 'v', color='#27ae60', markersize=9,
                     markeredgecolor='black', markeredgewidth=1.2, label=r'$\mathbf{2.50\ dB:\ FER < 10^{-9}\ (0\ FE\ /\ 1.001B\ blocks)}$')
        ax1.annotate(f"2.50 dB: 0 FE in 1.001B blocks\nFER < {bound_fer:.2e} (Upper Bound)",
                     xy=(r25['EbN0_dB'], bound_fer),
                     xytext=(-20, -35), textcoords="offset points",
                     ha='right', fontweight='bold', fontsize=9, color='#1e8449',
                     arrowprops=dict(arrowstyle="->", color='#1e8449', lw=1.5))

ax1.set_xlabel(r'Signal-to-Noise Ratio $E_b/N_0$ [dB]', fontsize=12, fontweight='bold')
ax1.set_ylabel('Error Rate (Log Scale)', fontsize=12, fontweight='bold')
ax1.set_title('CCSDS AR4JA Rate-1/2: Ultra-Deep Waterfall\n(16 vs. 100 vs. 200 Max Iterations; Clipped vs. Unclipped)', fontsize=13, fontweight='bold')
ax1.set_xlim(1.15, 2.7)
ax1.set_ylim(2e-10, 0.3)
ax1.grid(True, which='both', linestyle='--', alpha=0.5)
ax1.legend(loc='lower left', frameon=True, fancybox=True, shadow=True, fontsize=8.5)

# Panel 2: Comparative Coding Gain vs 16-Iter Baseline
if df_16 is not None and df_200_uncl is not None:
    merged_uncl = pd.merge(df_200_uncl, df_16, on='EbN0_dB', suffixes=('_uncl', '_16'))
    merged_uncl = merged_uncl[merged_uncl['FER_uncl'] > 0].copy()
    if not merged_uncl.empty:
        fer_ratio_uncl = merged_uncl['FER_16'] / merged_uncl['FER_uncl']

        ax2.axhline(1.0, color='black', linestyle='-', linewidth=1.5, alpha=0.7, label='16-Iter Baseline (1.0x)')
        ax2.plot(merged_uncl['EbN0_dB'], fer_ratio_uncl, 'o-', color=c_200_uncl, linewidth=2.4, markersize=7, label=r'200-Iter Unclipped Gain ($\mathrm{FER}_{16} / \mathrm{FER}_{\mathrm{uncl}}$)')

        if df_200_clip is not None:
            m_clip = pd.merge(df_200_clip, df_16, on='EbN0_dB', suffixes=('_clip', '_16'))
            m_clip = m_clip[m_clip['FER_clip'] > 0].copy()
            if not m_clip.empty:
                r_clip = m_clip['FER_16'] / m_clip['FER_clip']
                ax2.plot(m_clip['EbN0_dB'], r_clip, 'D--', color=c_200_clip, linewidth=1.8, markersize=5, label=r'200-Iter Clipped Gain')

        if df_100 is not None:
            m100 = pd.merge(df_100, df_16, on='EbN0_dB', suffixes=('_100', '_16'))
            m100 = m100[m100['FER_100'] > 0].copy()
            if not m100.empty:
                r100 = m100['FER_16'] / m100['FER_100']
                ax2.plot(m100['EbN0_dB'], r100, '^:', color=c_100, linewidth=1.5, markersize=4, label=r'100-Iter Clipped Gain')

        if df_200_rec is not None:
            m_rec = pd.merge(df_200_rec, df_16, on='EbN0_dB', suffixes=('_rec', '_16'))
            m_rec = m_rec[m_rec['FER_rec'] > 0].copy()
            if not m_rec.empty:
                r_rec = m_rec['FER_16'] / m_rec['FER_rec']
                ax2.plot(m_rec['EbN0_dB'], r_rec, 'D-', color=c_rec_fer, linewidth=2.6, markersize=7, label=r'$\mathbf{200\ Iter\ True\ Recursive\ Gain}$')
                for _, r_row in m_rec.iterrows():
                    ratio_r = r_row['FER_16'] / r_row['FER_rec']
                    if r_row['EbN0_dB'] in [1.8, 1.9, 2.0, 2.1, 2.2]:
                        offset_y = 12 if r_row['EbN0_dB'] != 2.2 else -18
                        ax2.annotate(f"{ratio_r:.0f}x", 
                                     xy=(r_row['EbN0_dB'], ratio_r),
                                     xytext=(0, offset_y), textcoords="offset points",
                                     ha='center', fontweight='bold', fontsize=9.5, color=c_rec_fer)

        for _, row in merged_uncl.iterrows():
            ratio = row['FER_16'] / row['FER_uncl']
            ax2.annotate(f"{ratio:.1f}x", 
                         xy=(row['EbN0_dB'], ratio),
                         xytext=(0, -14), textcoords="offset points",
                         ha='center', fontweight='normal', fontsize=8, color=c_200_uncl)

        ax2.set_yscale('log')
        ax2.set_xlabel(r'Signal-to-Noise Ratio $E_b/N_0$ [dB]', fontsize=12, fontweight='bold')
        ax2.set_ylabel('FER Error Reduction Factor (Log Scale)', fontsize=12, fontweight='bold')
        ax2.set_title('Coding Gain vs. 16-Iteration Baseline\n(Values > 1.0 indicate performance improvement)', fontsize=13, fontweight='bold')
        ax2.set_xlim(1.55, 2.45)
        ax2.grid(True, which='both', linestyle='--', alpha=0.5)
        ax2.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=10)

plt.tight_layout()
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"[PLOT] Multi-iteration comparison plot saved to {output_plot}")
