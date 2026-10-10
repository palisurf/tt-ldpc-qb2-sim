import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_path = "Results/results_AR4JA_r45_4c_128c_r12_amin_ebn0_0.00_3.00_step0.10_max2275000_min25_iter16_440cores.csv"
log_path = "Results/run_overnight.log"

df = pd.read_csv(csv_path)

# Check if there is an in-progress point in the log
latest_pt = None
if os.path.exists(log_path):
    with open(log_path, 'r') as f:
        for line in f:
            if line.strip().startswith("[") and "dB]" in line and "Blocks:" in line:
                latest_pt = line.strip()

in_progress_point = None
if latest_pt:
    try:
        # e.g.: [2.80 dB] Blocks: 668,800,000/1,001,000,000 ( 66.8%) | FE: 0/25 | FER: 0.00e+00 | BER: 0.00e+00 | 55.3 Msps | 24787.9s
        ebn0_str = latest_pt.split("[")[1].split("dB")[0].strip()
        ebn0_val = float(ebn0_str)
        blocks_str = latest_pt.split("Blocks:")[1].split("/")[0].replace(",", "").strip()
        blocks_val = int(blocks_str)
        fe_str = latest_pt.split("FE:")[1].split("/")[0].strip()
        fe_val = int(fe_str)
        fer_str = latest_pt.split("FER:")[1].split("|")[0].strip()
        fer_val = float(fer_str)
        ber_str = latest_pt.split("BER:")[1].split("|")[0].strip()
        ber_val = float(ber_str)

        if ebn0_val not in df['EbN0_dB'].values:
            in_progress_point = {
                'ebn0': ebn0_val,
                'blocks': blocks_val,
                'fe': fe_val,
                'fer': fer_val,
                'ber': ber_val,
                'fer_bound': 1.0 / blocks_val if blocks_val > 0 else 1e-9,
                'ber_bound': 1.0 / (blocks_val * 2048) if blocks_val > 0 else 1e-12
            }
    except Exception as e:
        print(f"Note: Could not parse in-progress point: {e}")

df = df.sort_values(by='EbN0_dB').reset_index(drop=True)

# Create two-panel figure
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)

c_fer = '#1f77b4'
c_ber = '#d62728'

# --- Subplot 1: Full Range 0.0 - 2.8 dB ---
ax1.semilogy(df['EbN0_dB'], df['FER'], 'o-', color=c_fer, linewidth=2, markersize=5, label='Frame Error Rate (FER)')
ax1.semilogy(df['EbN0_dB'], df['BER'], 's--', color=c_ber, linewidth=1.8, markersize=4.5, label='Bit Error Rate (BER)')

if in_progress_point and in_progress_point['fe'] == 0:
    ax1.plot(in_progress_point['ebn0'], in_progress_point['fer_bound'], 'v', color=c_fer, markersize=8, label='FER Upper Bound (0 errors)')
    ax1.annotate(
        f"2.8 dB: 0 errors\n(< {in_progress_point['fer_bound']:.1e})",
        xy=(in_progress_point['ebn0'], in_progress_point['fer_bound']),
        xytext=(in_progress_point['ebn0'] - 0.55, in_progress_point['fer_bound'] * 5),
        arrowprops=dict(facecolor=c_fer, shrink=0.1, width=1, headwidth=5),
        fontsize=8.5, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="#e8f4f8", ec=c_fer, alpha=0.9)
    )

ax1.set_title("Complete Sweep ($0.0 - 2.8$ dB)\nRoll-off & Transition Regime", fontsize=13, fontweight='bold', pad=10)
ax1.set_xlabel("$E_b / N_0$ (dB)", fontsize=12, fontweight='bold')
ax1.set_ylabel("Error Probability", fontsize=12, fontweight='bold')
ax1.set_ylim(1e-10, 1.5)
ax1.set_xlim(-0.1, 2.95)
ax1.grid(True, which="both", ls=":", alpha=0.5)
ax1.legend(fontsize=10, loc='lower left', frameon=True)

# --- Subplot 2: Waterfall Zoom 1.4 - 2.85 dB ---
waterfall_df = df[df['EbN0_dB'] >= 1.4]
ax2.semilogy(waterfall_df['EbN0_dB'], waterfall_df['FER'], 'o-', color=c_fer, linewidth=2.5, markersize=7, label='FER (Approximate-Min*)')
ax2.semilogy(waterfall_df['EbN0_dB'], waterfall_df['BER'], 's--', color=c_ber, linewidth=2.0, markersize=6, label='BER (Approximate-Min*)')

# 2.70 dB annotation
pt_27 = df[df['EbN0_dB'] == 2.70].iloc[0]
ax2.annotate(
    f"2.7 dB: FER = {pt_27['FER']:.2e}\n({int(pt_27['Total_Blocks']):,} blks, 28 FE)",
    xy=(pt_27['EbN0_dB'], pt_27['FER']),
    xytext=(pt_27['EbN0_dB'] - 0.52, pt_27['FER'] * 4.0),
    arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=5),
    fontsize=9, fontweight='bold',
    bbox=dict(boxstyle="round,pad=0.35", fc="#ffffcc", ec="black", alpha=0.9)
)

# 2.80 dB Upper bound marker if zero errors
if in_progress_point and in_progress_point['fe'] == 0:
    ax2.plot(in_progress_point['ebn0'], in_progress_point['fer_bound'], 'v', color='darkblue', markersize=10, markeredgecolor='black', label='2.8 dB Limit (0 errors)')
    ax2.annotate(
        f"2.8 dB: 0 Errors in {in_progress_point['blocks']:,} blks\nFER < {in_progress_point['fer_bound']:.2e} (running)",
        xy=(in_progress_point['ebn0'], in_progress_point['fer_bound']),
        xytext=(in_progress_point['ebn0'] - 0.65, in_progress_point['fer_bound'] * 0.2),
        arrowprops=dict(facecolor='darkblue', shrink=0.08, width=1.5, headwidth=6),
        fontsize=9, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.35", fc="#d4edda", ec="#28a745", alpha=0.95)
    )

ax2.set_title("Deep Waterfall & Error Floor ($1.4 - 2.85$ dB)\nPenetrating into Sub-$10^{-8}$ Regime", fontsize=13, fontweight='bold', pad=10)
ax2.set_xlabel("$E_b / N_0$ (dB)", fontsize=12, fontweight='bold')
ax2.set_ylabel("Error Probability", fontsize=12, fontweight='bold')
ax2.set_ylim(5e-11, 0.5)
ax2.set_xlim(1.35, 2.92)
ax2.grid(True, which="both", ls=":", alpha=0.5)
ax2.legend(fontsize=10, loc='lower left', frameon=True)

fig.suptitle(
    "Tenstorrent QB2 (440 Tensix Cores) | AR4JA Rate-1/2 ($N=2048, K=1024$)\nApproximate-Min* (Christopher Jones MILCOM 2003, BF16)",
    fontsize=14, fontweight='bold', y=0.99
)

plt.tight_layout()
os.makedirs("Results", exist_ok=True)
out_png = "Results/waterfall_plot.png"
plt.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"Plot successfully saved to {out_png}")
