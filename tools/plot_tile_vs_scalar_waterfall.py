import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# File paths
csv_tile = "Results/results_tile_matrix_4cw_ar4ja_r12_ebn0_1.50_2.50_step0.10_max1000000_min20_iter200_1core.csv"
csv_scalar = "Results/Scalar_5w/results_AR4JA_r45_4c_128c_r12_amin_ebn0_1.00_2.50_step0.10_max22727272_min25_iter200_440cores.csv"
output_plot = "Results/waterfall_tile_matrix_vs_scalar.png"

# Load datasets
df_tile = pd.read_csv(csv_tile) if os.path.exists(csv_tile) else None
df_scalar = pd.read_csv(csv_scalar) if os.path.exists(csv_scalar) else None

# Clean up duplicate SNR rows in tile csv if any (keep latest run per SNR)
if df_tile is not None:
    df_tile = df_tile.drop_duplicates(subset=['EbN0_dB'], keep='last').sort_values('EbN0_dB')

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

# Colors
c_tile_fer = '#d63031'    # Crimson
c_tile_ber = '#e17055'    # Coral/Orange
c_scalar_fer = '#0984e3'  # Blue
c_scalar_ber = '#74b9ff'  # Light Blue

# Panel 1: Waterfall Curves (FER & BER)
if df_scalar is not None:
    sub_scalar = df_scalar[(df_scalar['EbN0_dB'] >= 1.0) & (df_scalar['EbN0_dB'] <= 2.5)]
    ax1.semilogy(sub_scalar['EbN0_dB'], sub_scalar['FER'], 's--', color=c_scalar_fer, 
                 linewidth=2.0, markersize=6, label=r'Scalar Reference (440 Cores) - FER', alpha=0.85)
    ax1.semilogy(sub_scalar['EbN0_dB'], sub_scalar['BER'], 's:', color=c_scalar_ber, 
                 linewidth=1.5, markersize=5, label=r'Scalar Reference (440 Cores) - BER', alpha=0.75)

if df_tile is not None:
    valid_tile_fer = df_tile[df_tile['FER'] > 0]
    valid_tile_ber = df_tile[df_tile['BER'] > 0]
    ax1.semilogy(valid_tile_fer['EbN0_dB'], valid_tile_fer['FER'], 'D-', color=c_tile_fer, 
                 linewidth=2.4, markersize=7, label=r'Tile Matrix 4-CW (1 Core) - FER')
    ax1.semilogy(valid_tile_ber['EbN0_dB'], valid_tile_ber['BER'], '^--', color=c_tile_ber, 
                 linewidth=1.8, markersize=6, label=r'Tile Matrix 4-CW (1 Core) - BER')

ax1.set_xlabel(r'$E_b/N_0$ [dB]', fontsize=13, fontweight='bold')
ax1.set_ylabel(r'Error Rate (FER / BER)', fontsize=13, fontweight='bold')
ax1.set_title(r'CCSDS AR4JA Rate-1/2 ($N=2560, K=1024$)' + '\n' + r'Waterfall Comparison: Tile Matrix vs. Scalar Reference', 
              fontsize=14, fontweight='bold', pad=12)
ax1.set_xlim(0.9, 2.6)
ax1.set_ylim(1e-9, 1.0)
ax1.grid(True, which='both', linestyle='--', alpha=0.5)
ax1.legend(loc='lower left', fontsize=10, frameon=True, facecolor='white', framealpha=0.95)

# Panel 2: Mean Iterations
if df_tile is not None and 'Mean_Iterations' in df_tile.columns:
    ax2.plot(df_tile['EbN0_dB'], df_tile['Mean_Iterations'], 'D-', color='#6c5ce7', 
             linewidth=2.2, markersize=7, label='Tile Matrix Engine (Mean Iters)')
    for _, row in df_tile.iterrows():
        ax2.annotate(f"{row['Mean_Iterations']:.1f}", 
                     (row['EbN0_dB'], row['Mean_Iterations']),
                     textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9, fontweight='bold', color='#2d3436')

ax2.set_xlabel(r'$E_b/N_0$ [dB]', fontsize=13, fontweight='bold')
ax2.set_ylabel(r'Average Iterations to Convergence', fontsize=13, fontweight='bold')
ax2.set_title(r'Early Parity Termination Convergence Speed' + '\n' + r'Average Iterations per Codeword ($I_{\max}=200$)', 
              fontsize=14, fontweight='bold', pad=12)
ax2.set_xlim(1.4, 2.6)
ax2.set_ylim(0, 75)
ax2.grid(True, which='both', linestyle='--', alpha=0.5)
ax2.legend(loc='upper right', fontsize=10, frameon=True, facecolor='white', framealpha=0.95)

plt.tight_layout()
os.makedirs(os.path.dirname(output_plot), exist_ok=True)
plt.savefig(output_plot, dpi=300)
print(f"[SUCCESS] Plot saved to: {output_plot}")
