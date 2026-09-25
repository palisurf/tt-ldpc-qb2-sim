import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

unclipped_csv = "Results/results_AR4JA_r45_4c_128c_r12_amin_ebn0_0.00_3.00_step0.10_max2275000_min25_iter16_440cores.csv"
clipped_pattern = "Results/results_AR4JA*clipL4.2*.csv"
output_plot = "Results/clipping_comparison_waterfall.png"

# Find the latest clipped results CSV
clipped_files = glob.glob(clipped_pattern)
if not clipped_files:
    print("No clipped results file found yet.")
    exit(0)

clipped_csv = sorted(clipped_files)[-1]
print(f"Reading unclipped baseline: {unclipped_csv}")
print(f"Reading clipped results  : {clipped_csv}")

df_uncl = pd.read_csv(unclipped_csv)
df_clip = pd.read_csv(clipped_csv)

df_uncl = df_uncl.sort_values(by='EbN0_dB').reset_index(drop=True)
df_clip = df_clip.sort_values(by='EbN0_dB').reset_index(drop=True)

# Filter unclipped to the overlapping range or relevant waterfall range (1.4 - 2.8 dB)
df_uncl_sub = df_uncl[(df_uncl['EbN0_dB'] >= 1.5) & (df_uncl['EbN0_dB'] <= 2.8)].reset_index(drop=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)

c_uncl_fer = '#1f77b4'  # Blue
c_uncl_ber = '#aec7e8'  # Light blue
c_clip_fer = '#d62728'  # Crimson red
c_clip_ber = '#ff9896'  # Light red

# --- Panel 1: Waterfall Curves (FER & BER) ---
# Unclipped Baseline
ax1.semilogy(df_uncl_sub['EbN0_dB'], df_uncl_sub['FER'], 'o--', color=c_uncl_fer, linewidth=1.8, markersize=5, label='Unclipped Baseline (FER)')
ax1.semilogy(df_uncl_sub['EbN0_dB'], df_uncl_sub['BER'], 's:', color=c_uncl_ber, linewidth=1.4, markersize=4, label='Unclipped Baseline (BER)')

# Channel Clipped (L_ch = 4.2)
ax1.semilogy(df_clip['EbN0_dB'], df_clip['FER'], 'o-', color=c_clip_fer, linewidth=2.2, markersize=6, label=r'Channel Clipped $L_{\mathrm{ch}}=4.2$ (FER)')
ax1.semilogy(df_clip['EbN0_dB'], df_clip['BER'], 's--', color='#b22222', linewidth=1.6, markersize=5, label=r'Channel Clipped $L_{\mathrm{ch}}=4.2$ (BER)')

ax1.set_xlabel(r'Signal-to-Noise Ratio $E_b/N_0$ [dB]', fontsize=11, fontweight='bold')
ax1.set_ylabel('Error Rate (Log Scale)', fontsize=11, fontweight='bold')
ax1.set_title('CCSDS AR4JA Rate-1/2: Channel Clipping Comparison\n(Unclipped Baseline vs. $L_{\mathrm{ch}}=4.2$ Bounded Noise Spikes)', fontsize=12, fontweight='bold')
ax1.set_xlim(1.5, 2.7)
ax1.set_ylim(1e-8, 0.2)
ax1.grid(True, which='both', linestyle='--', alpha=0.5)
ax1.legend(loc='lower left', frameon=True, fancybox=True, shadow=True, fontsize=9.5)

# --- Panel 2: Error Reduction Ratio (Clipped / Unclipped) ---
merged = pd.merge(df_clip, df_uncl, on='EbN0_dB', suffixes=('_clip', '_uncl'))
if not merged.empty:
    fer_ratio = merged['FER_clip'] / merged['FER_uncl']
    ber_ratio = merged['BER_clip'] / merged['BER_uncl']

    ax2.axhline(1.0, color='black', linestyle='-', linewidth=1.5, alpha=0.7, label='Baseline Parity (1.0x)')
    ax2.plot(merged['EbN0_dB'], fer_ratio, 'o-', color=c_clip_fer, linewidth=2.0, markersize=6, label='FER Ratio (Clipped / Unclipped)')
    ax2.plot(merged['EbN0_dB'], ber_ratio, 's--', color='#ff7f0e', linewidth=1.8, markersize=6, label='BER Ratio (Clipped / Unclipped)')

    ax2.set_xlabel(r'Signal-to-Noise Ratio $E_b/N_0$ [dB]', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Error Rate Ratio (Clipped / Baseline)', fontsize=11, fontweight='bold')
    ax2.set_title('Relative Gain from Channel LLR Clipping ($L_{\mathrm{ch}}=4.2$)\n(Values < 1.0 indicate performance improvement)', fontsize=12, fontweight='bold')
    ax2.set_xlim(1.5, max(2.7, merged['EbN0_dB'].max() + 0.1))
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=9.5)

plt.tight_layout()
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"[PLOT] Comparative waterfall plot saved to {output_plot}")
