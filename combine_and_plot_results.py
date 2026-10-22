#!/usr/bin/env python3
"""
combine_and_plot_results.py

Combines historical JPL AR4JA Rate-1/2 LDPC simulation benchmark (2005)
with live Tenstorrent QuietBox 2 (440 Tensix cores) 200-iteration simulation results.

Computes 95% Poisson confidence intervals (error bars) for both simulations
and generates:
  1. Combined CSV results file: Results/combined_ar4ja_jpl_qb2_iter200.csv
  2. Publication-quality comparison plot: Results/combined_ar4ja_r12_waterfall_iter200.png
"""

import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure tools/ directory is available for import_jpl_results
tools_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools")
if tools_dir not in sys.path:
    sys.path.insert(0, tools_dir)
from import_jpl_results import load_jpl_results, poisson_ci_95


def get_live_qb2_point(log_path: str = "Results/simulation_full_1.0_2.5dB_iter200.log"):
    """Extracts latest in-progress telemetry point from QB2 simulation log if available."""
    if not os.path.exists(log_path):
        return None
    latest_line = None
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.strip().startswith("[") and "dB]" in line and "Blocks:" in line and "FE:" in line:
                latest_line = line.strip()
    if not latest_line:
        return None

    try:
        ebn0_m = re.search(r'\[\s*([-+]?\d*\.?\d+)\s*dB\]', latest_line)
        blocks_m = re.search(r'Blocks:\s*([\d,]+)/', latest_line)
        fe_m = re.search(r'FE:\s*(\d+)/', latest_line)
        fer_m = re.search(r'FER:\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?)', latest_line, re.IGNORECASE)
        ber_m = re.search(r'BER:\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?)', latest_line, re.IGNORECASE)

        if ebn0_m and blocks_m and fe_m and fer_m and ber_m:
            ebn0 = float(ebn0_m.group(1))
            blocks = int(blocks_m.group(1).replace(",", ""))
            fe = int(fe_m.group(1))
            fer = float(fer_m.group(1))
            raw_ber = float(ber_m.group(1))
            bit_errors = int(round(raw_ber * blocks * 1024))
            ser = bit_errors / (blocks * 2560.0) if blocks > 0 else 0.0
            return {
                'EbN0_dB': ebn0,
                'Total_Blocks': blocks,
                'Total_Bit_Errors': bit_errors,
                'Total_Block_Errors': fe,
                'SER': ser,
                'FER': fer,
                'Is_Live': True
            }
    except Exception as e:
        print(f"Note: Error parsing live log line: {e}")
    return None


def add_confidence_intervals_qb2(df: pd.DataFrame) -> pd.DataFrame:
    """Computes exact 95% Poisson confidence intervals for QB2 simulation data scaled to N=2560."""
    fer_ci_low = []
    fer_ci_high = []
    ser_ci_low = []
    ser_ci_high = []

    for _, r in df.iterrows():
        k = r['Total_Block_Errors']
        n = r['Total_Blocks']
        fer = r['FER']
        ser = r['SER']
        if n > 0 and k > 0:
            k_l, k_h = poisson_ci_95(k)
            fer_ci_low.append(k_l / n)
            fer_ci_high.append(k_h / n)
            ser_ci_low.append(ser * (k_l / k))
            ser_ci_high.append(ser * (k_h / k))
        else:
            fer_ci_low.append(0.0)
            fer_ci_high.append(0.0)
            ser_ci_low.append(0.0)
            ser_ci_high.append(0.0)

    df['FER_ci_lower'] = fer_ci_low
    df['FER_ci_upper'] = fer_ci_high
    df['SER_ci_lower'] = ser_ci_low
    df['SER_ci_upper'] = ser_ci_high

    df['FER_err_minus'] = np.maximum(0.0, df['FER'] - df['FER_ci_lower'])
    df['FER_err_plus'] = np.maximum(0.0, df['FER_ci_upper'] - df['FER'])
    df['SER_err_minus'] = np.maximum(0.0, df['SER'] - df['SER_ci_lower'])
    df['SER_err_plus'] = np.maximum(0.0, df['SER_ci_upper'] - df['SER'])
    return df


def main():
    jpl_file = "Results/Scalar_5w/JPL Results - Read Only/AR4JA_r45_4c_128c_r12.txt"
    if not os.path.exists(jpl_file):
        jpl_file = "Results/JPL Results - Read Only/AR4JA_r45_4c_128c_r12.txt"

    qb2_csv = "Results/waterfall_0.0_2.6_10B_iter200.csv"
    if not os.path.exists(qb2_csv):
        qb2_csv = "Results/results_AR4JA_r45_4c_128c_r12_amin_ebn0_1.00_2.50_step0.10_max22727272_min25_iter200_440cores_unclipped.csv"

    output_csv = "Results/combined_ar4ja_jpl_qb2_iter200.csv"
    output_unclipped_csv = "Results/combined_ar4ja_jpl_qb2_iter200_unclipped.csv"
    output_png = "Results/combined_ar4ja_r12_waterfall_iter200.png"
    scalar_folder_png = "Results/Scalar_5w/combined_ar4ja_r12_waterfall_iter200.png"
    jpl_folder_png = "Results/Scalar_5w/JPL Results - Read Only/combined_ar4ja_r12_waterfall_iter200.png"
    doc_fig_png = "docs/figures/combined_ar4ja_r12_waterfall_iter200.png"

    # 1. Load JPL historical results (24 active points, excluding 3 '%' lines)
    # Note: JPL BER was already computed with blocklength N=2560 -> assign directly to SER
    df_jpl = load_jpl_results(jpl_file, code_rate=0.5, block_length=2560, include_commented=False)
    df_jpl['Source'] = 'JPL 2005 (Flood Decoder)'
    df_jpl['Is_Live'] = False
    df_jpl['SER'] = df_jpl['BER']
    df_jpl['SER_ci_lower'] = df_jpl['BER_ci_lower']
    df_jpl['SER_ci_upper'] = df_jpl['BER_ci_upper']
    df_jpl['SER_err_minus'] = df_jpl['BER_err_minus']
    df_jpl['SER_err_plus'] = df_jpl['BER_err_plus']

    # 2. Load QB2 results and scale to N=2560
    df_qb2 = pd.read_csv(qb2_csv)
    df_qb2['Source'] = 'QuietBox 2 440 Tensix Cores (Christopher R. Jones, PhD, 2026 - Layered Decoder)'
    df_qb2['Is_Live'] = False
    # Scale QB2 SER to N=2560: Total_Bit_Errors / (Total_Blocks * 2560)
    df_qb2['SER'] = df_qb2['Total_Bit_Errors'] / (df_qb2['Total_Blocks'] * 2560.0)

    # 2b. Load QB2 L_chan clipped results (L_max = 6.5) if available
    qb2_clip_csv = "Results/Scalar_5w/results_AR4JA_r45_4c_128c_r12_amin_clipL6.5R0.0_ebn0_2.50_2.50_step0.10_max22727272_min25_iter200_440cores.csv"
    if not os.path.exists(qb2_clip_csv):
        qb2_clip_csv = "Results/results_AR4JA_r45_4c_128c_r12_amin_clipL6.5R0.0_ebn0_2.50_2.50_step0.10_max22727272_min25_iter200_440cores.csv"

    df_qb2_clip = None
    if os.path.exists(qb2_clip_csv):
        df_qb2_clip = pd.read_csv(qb2_clip_csv)
        df_qb2_clip['Source'] = 'QuietBox 2 440 Tensix Cores (Christopher R. Jones, PhD, 2026 - Layered, L_chan Clipped L_max=6.5)'
        df_qb2_clip['Is_Live'] = False
        df_qb2_clip['SER'] = df_qb2_clip['Total_Bit_Errors'] / (df_qb2_clip['Total_Blocks'] * 2560.0)
        df_qb2_clip = add_confidence_intervals_qb2(df_qb2_clip)
        print(f"[OK] Loaded L_chan clipped data from {qb2_clip_csv}")

    df_qb2 = add_confidence_intervals_qb2(df_qb2)

    # 3. Create unified Combined DataFrame
    # Format standardized columns
    jpl_std = pd.DataFrame({
        'Source': df_jpl['Source'],
        'EbN0_dB': df_jpl['EbN0_dB'].round(2),
        'Total_Blocks': df_jpl['Blocks_Run'],
        'Total_Symbol_Errors': df_jpl['Raw_Symbol_Errors'],
        'Total_Block_Errors': df_jpl['Raw_Frame_Errors'],
        'SER': df_jpl['SER'],
        'SER_ci_lower': df_jpl['SER_ci_lower'],
        'SER_ci_upper': df_jpl['SER_ci_upper'],
        'FER': df_jpl['FER_calc'],
        'FER_ci_lower': df_jpl['FER_ci_lower'],
        'FER_ci_upper': df_jpl['FER_ci_upper'],
        'Is_Live': df_jpl['Is_Live']
    })

    qb2_std = pd.DataFrame({
        'Source': df_qb2['Source'],
        'EbN0_dB': df_qb2['EbN0_dB'].round(2),
        'Total_Blocks': df_qb2['Total_Blocks'],
        'Total_Symbol_Errors': df_qb2['Total_Bit_Errors'],
        'Total_Block_Errors': df_qb2['Total_Block_Errors'],
        'SER': df_qb2['SER'],
        'SER_ci_lower': df_qb2['SER_ci_lower'],
        'SER_ci_upper': df_qb2['SER_ci_upper'],
        'FER': df_qb2['FER'],
        'FER_ci_lower': df_qb2['FER_ci_lower'],
        'FER_ci_upper': df_qb2['FER_ci_upper'],
        'Is_Live': df_qb2['Is_Live']
    })

    df_combined = pd.concat([jpl_std, qb2_std], ignore_index=True)
    df_combined = df_combined.sort_values(by=['EbN0_dB', 'Source']).reset_index(drop=True)

    # Save Combined CSV
    df_combined.to_csv(output_csv, index=False)
    df_combined.to_csv(output_unclipped_csv, index=False)
    if os.path.exists("Results/Scalar_5w"):
        df_combined.to_csv("Results/Scalar_5w/combined_ar4ja_jpl_qb2_iter200.csv", index=False)
    print(f"[OK] Saved combined simulation data to: {output_csv}")
    print(f"[OK] Saved duplicate to: {output_unclipped_csv}")

    # ================= 4. PLOTTING =================
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 8), dpi=300)

    # Styling Palette
    c_jpl_fer = '#1b4f72'    # Navy
    c_jpl_ser = '#78281f'    # Mahogany / Burgundy
    c_qb2_fer = '#1e8449'    # Emerald Green
    c_qb2_ser = '#d35400'    # Amber / Rust
    c_clip_fer = '#8e44ad'   # Royal Purple
    c_clip_ser = '#c0392b'   # Crimson

    # Filter non-zero points for log plot
    jpl_valid_fer = df_jpl[df_jpl['FER_calc'] > 0]
    jpl_valid_ser = df_jpl[df_jpl['SER'] > 0]
    qb2_valid_fer = df_qb2[df_qb2['FER'] > 0]
    qb2_valid_ser = df_qb2[df_qb2['SER'] > 0]

    for ax in (ax1, ax2):
        # 1. Historical JPL Simulation (Flood Decoder)
        ax.errorbar(
            jpl_valid_fer['EbN0_dB'], jpl_valid_fer['FER_calc'],
            yerr=[jpl_valid_fer['FER_err_minus'], jpl_valid_fer['FER_err_plus']],
            fmt='-o', color=c_jpl_fer, ecolor=c_jpl_fer, elinewidth=1.2,
            capsize=3.5, capthick=1.0,
            markerfacecolor='#2980b9', markeredgecolor='white',
            markeredgewidth=0.8, linewidth=2.0, markersize=6.5,
            label=r'$\mathbf{JPL\ 2005\ [Flood\ Decoder]\ FER}\ (\pm 95\%\ \mathrm{CI})$', zorder=4
        )
        ax.errorbar(
            jpl_valid_ser['EbN0_dB'], jpl_valid_ser['SER'],
            yerr=[jpl_valid_ser['SER_err_minus'], jpl_valid_ser['SER_err_plus']],
            fmt='--s', color=c_jpl_ser, ecolor=c_jpl_ser, elinewidth=1.1,
            capsize=3.0, capthick=0.9,
            markerfacecolor='#c0392b', markeredgecolor='white',
            markeredgewidth=0.8, linewidth=1.8, markersize=5.5,
            label=r'$\mathbf{JPL\ 2005\ [Flood\ Decoder]\ SER}\ (\pm 95\%\ \mathrm{CI})$', zorder=4
        )

        # 2. QuietBox 2 Simulation (Layered Decoder - Unclipped)
        ax.errorbar(
            qb2_valid_fer['EbN0_dB'], qb2_valid_fer['FER'],
            yerr=[qb2_valid_fer['FER_err_minus'], qb2_valid_fer['FER_err_plus']],
            fmt='-D', color=c_qb2_fer, ecolor=c_qb2_fer, elinewidth=1.3,
            capsize=3.5, capthick=1.0,
            markerfacecolor='#2ecc71', markeredgecolor='white',
            markeredgewidth=0.8, linewidth=2.2, markersize=6.5,
            label=r'$\mathbf{QB2\ [Layered,\ bfloat16,\ Unclipped]\ FER}\ (\pm 95\%\ \mathrm{CI})$', zorder=5
        )
        ax.errorbar(
            qb2_valid_ser['EbN0_dB'], qb2_valid_ser['SER'],
            yerr=[qb2_valid_ser['SER_err_minus'], qb2_valid_ser['SER_err_plus']],
            fmt='--^', color=c_qb2_ser, ecolor=c_qb2_ser, elinewidth=1.2,
            capsize=3.0, capthick=0.9,
            markerfacecolor='#e67e22', markeredgecolor='white',
            markeredgewidth=0.8, linewidth=1.8, markersize=6.0,
            label=r'$\mathbf{QB2\ [Layered,\ bfloat16,\ Unclipped]\ SER}\ (\pm 95\%\ \mathrm{CI})$', zorder=5
        )

        # 2b. QuietBox 2 Simulation (Layered Decoder - L_chan Clipped L_max=6.5)
        if df_qb2_clip is not None:
            clip_valid_fer = df_qb2_clip[df_qb2_clip['FER'] > 0]
            clip_valid_ser = df_qb2_clip[df_qb2_clip['SER'] > 0]
            ax.errorbar(
                clip_valid_fer['EbN0_dB'], clip_valid_fer['FER'],
                yerr=[clip_valid_fer['FER_err_minus'], clip_valid_fer['FER_err_plus']],
                fmt='X', color=c_clip_fer, ecolor=c_clip_fer, elinewidth=1.3,
                capsize=3.5, capthick=1.0,
                markerfacecolor='#af7ac5', markeredgecolor='white',
                markeredgewidth=0.8, linewidth=0.0, markersize=8.5,
                label=r'$\mathbf{QB2\ [Layered,\ L_{\mathrm{chan}}=6.5\ Clipped]\ FER}\ (\pm 95\%\ \mathrm{CI})$', zorder=6
            )
            ax.errorbar(
                clip_valid_ser['EbN0_dB'], clip_valid_ser['SER'],
                yerr=[clip_valid_ser['SER_err_minus'], clip_valid_ser['SER_err_plus']],
                fmt='v', color='#e74c3c', ecolor='#e74c3c', elinewidth=1.2,
                capsize=3.0, capthick=0.9,
                markerfacecolor='#f1948a', markeredgecolor='white',
                markeredgewidth=0.8, linewidth=0.0, markersize=7.5,
                label=r'$\mathbf{QB2\ [Layered,\ L_{\mathrm{chan}}=6.5\ Clipped]\ SER}\ (\pm 95\%\ \mathrm{CI})$', zorder=6
            )

        ax.set_yscale('log')
        ax.grid(True, which='major', linestyle='-', linewidth=0.75, color='#d0d7de', alpha=0.8)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='#e1e4e8', alpha=0.6)
        ax.set_ylabel('Error Probability (FER / SER)', fontsize=12, fontweight='bold', color='#24292f')

    # Panel 1: Full Waterfall (0.0 to 2.76 dB)
    ax1.set_title('Full Waterfall Comparison: Flood (JPL 2005) vs. Layered (QuietBox 2 2026)', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel(r'$E_b/N_0$ [dB]', fontsize=12, fontweight='bold', color='#24292f')
    ax1.set_xlim(-0.05, 2.76)
    ax1.set_ylim(1e-11, 2.0)
    ax1.legend(loc='lower left', fontsize=7.4, frameon=True, framealpha=0.95, facecolor='white', edgecolor='#d0d7de')

    info_box = (
        r"$\mathbf{Code}$: CCSDS AR4JA Rate-1/2 ($N=2560, K=1024$)" + "\n"
        r"$\mathbf{Decoder}$: Approximate-Min* (200 max iterations)" + "\n"
        r"$\mathbf{Authors}$: Christopher R. Jones, Dariush Divsalar, and" + "\n"
        r"            Richard D. Wesel (UCLA / NASA JPL)" + "\n"
        r"$\mathbf{JPL\ 2005\ Data}$: Flood Decoder (Parallel Schedule)" + "\n"
        r"$\mathbf{QB2\ 2026\ Data}$: Row-Layered Decoder (Circulant DMA Engine)" + "\n"
        r"$\mathbf{QB2\ Precision}$: Native 16-bit FP ($\mathtt{bfloat16}$ / $\mathtt{Float16\_b}$)" + "\n"
        r"$\mathbf{QB2\ Clipping}$: NO LLR Clipping (Unclipped Channel and Messages)" + "\n"
        r"$\mathbf{Hardware}$: Tenstorrent QuietBox 2 (440 Tensix Cores, 2x2 Mesh)" + "\n"
        r"$\mathbf{Throughput}$: $153\ \mathrm{Msps} \to 5.15\ \mathrm{Gsps}$ ($2.58\ \mathrm{Gbps}$ info rate)" + "\n"
        r"$\mathbf{Campaign}$: 29.5 Billion blocks simulated across 27 SNR points" + "\n"
        r"$\mathbf{SER\ Metric}$: Symbol Error Rate (All SERs scaled to $N=2560$)" + "\n"
        r"$\mathbf{Error\ Bars}$: Exact 95% Poisson/Binomial CI"
    )
    ax1.text(
        0.04, 0.49, info_box, transform=ax1.transAxes,
        fontsize=7.6, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.55', facecolor='#f8f9fa', edgecolor='#bdc3c7', alpha=0.95)
    )

    # Panel 2: Deep Floor Comparison (1.70 to 2.76 dB)
    ax2.set_title(r'Ultra-Deep Floor Agreement Down to $10^{-10}$ with 95% Error Bars', fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel(r'$E_b/N_0$ [dB]', fontsize=12, fontweight='bold', color='#24292f')
    ax2.set_xlim(1.70, 2.76)
    ax2.set_ylim(1.5e-11, 6e-4)
    ax2.legend(loc='upper right', fontsize=7.4, frameon=True, framealpha=0.95, facecolor='white', edgecolor='#d0d7de')

    # Annotate Deep Floor Points
    # 1. QB2 2.20 dB point - placed UP into open white corridor above curves
    pt_220 = qb2_valid_fer[qb2_valid_fer['EbN0_dB'] == 2.20]
    if not pt_220.empty:
        r22 = pt_220.iloc[0]
        ax2.annotate(
            f"QB2 2.20 dB: {int(r22['Total_Block_Errors'])} FE ({r22['Total_Blocks']/1e6:.0f}M blks)\nFER = {r22['FER']:.2e} [{r22['FER_ci_lower']*1e8:.1f}-{r22['FER_ci_upper']*1e8:.1f}]e-8",
            xy=(2.20, r22['FER']),
            xytext=(0, 82), textcoords='offset points',
            fontsize=7.1, fontweight='bold', color=c_qb2_fer, ha='center',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#f8f9fa', edgecolor='#1e8449', alpha=0.92, lw=0.8),
            arrowprops=dict(arrowstyle="->", color=c_qb2_fer, lw=1.2, shrinkA=3, shrinkB=3)
        )

    # 2. JPL 2.26 dB point - placed left and down into lower-left whitespace
    ax2.annotate(
        "JPL 2.26 dB: 3 FE (295M blks)\nFER = 1.02e-8",
        xy=(2.2603, 1.0186e-08),
        xytext=(-105, -26), textcoords='offset points',
        fontsize=7.2, fontweight='bold', color=c_jpl_fer, ha='right',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#f8f9fa', edgecolor='#1b4f72', alpha=0.92, lw=0.8),
        arrowprops=dict(arrowstyle="->", color=c_jpl_fer, lw=1.1, shrinkA=3, shrinkB=3)
    )

    # 3. QB2 2.30 dB point - placed UP above curve in open corridor
    pt_230 = qb2_valid_fer[qb2_valid_fer['EbN0_dB'] == 2.30]
    if not pt_230.empty:
        r23 = pt_230.iloc[0]
        ax2.annotate(
            f"QB2 2.30 dB: {int(r23['Total_Block_Errors'])} FE ({r23['Total_Blocks']/1e9:.2f}B blks)\nFER = {r23['FER']:.2e} [{r23['FER_ci_lower']*1e8:.1f}-{r23['FER_ci_upper']*1e8:.1f}]e-8\nSER = {r23['SER']:.2e}",
            xy=(2.30, r23['FER']),
            xytext=(0, 52), textcoords='offset points',
            fontsize=7.1, fontweight='bold', color=c_qb2_fer, ha='center',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#e8f8f5', edgecolor='#1e8449', alpha=0.92, lw=0.8),
            arrowprops=dict(arrowstyle="->", color=c_qb2_fer, lw=1.2, shrinkA=3, shrinkB=3)
        )

    # 4. QB2 2.40 dB point - placed UP in pristine whitespace
    pt_240 = qb2_valid_fer[qb2_valid_fer['EbN0_dB'] == 2.40]
    if not pt_240.empty:
        r24 = pt_240.iloc[0]
        ax2.annotate(
            f"QB2 2.40 dB: {int(r24['Total_Block_Errors'])} FE ({r24['Total_Blocks']/1e9:.2f}B blks)\nFER = {r24['FER']:.2e} [{r24['FER_ci_lower']*1e9:.1f}-{r24['FER_ci_upper']*1e9:.1f}]e-9\nSER = {r24['SER']:.2e}",
            xy=(2.40, r24['FER']),
            xytext=(0, 42), textcoords='offset points',
            fontsize=7.1, fontweight='bold', color=c_qb2_fer, ha='center',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#e8f8f5', edgecolor='#1e8449', alpha=0.92, lw=0.8),
            arrowprops=dict(arrowstyle="->", color=c_qb2_fer, lw=1.2, shrinkA=3, shrinkB=3)
        )

    # 5. JPL 2.39 dB point - placed down and left into pristine bottom whitespace (y ~ 5e-11)
    ax2.annotate(
        "JPL 2.39 dB: 2 FE (2.1B blks)\nFER = 9.52e-10\nSER = 1.74e-10",
        xy=(2.3853, 9.5238e-10),
        xytext=(-35, -55), textcoords='offset points',
        fontsize=7.2, fontweight='bold', color=c_jpl_fer, ha='right',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#f8f9fa', edgecolor='#1b4f72', alpha=0.92, lw=0.8),
        arrowprops=dict(arrowstyle="->", color=c_jpl_fer, lw=1.1, shrinkA=3, shrinkB=3)
    )

    # 6. QB2 2.50 dB point - L_chan Clipped (L_max = 6.5) - UPPER stack in upper-right corridor
    if df_qb2_clip is not None:
        pt_clip = df_qb2_clip[df_qb2_clip['EbN0_dB'] == 2.50]
        if not pt_clip.empty:
            rc = pt_clip.iloc[0]
            ax2.annotate(
                f"QB2 2.50 dB (L_chan=6.5):\n{int(rc['Total_Block_Errors'])} FE ({rc['Total_Blocks']/1e9:.2f}B blks)\nFER = {rc['FER']:.2e} [{rc['FER_ci_lower']*1e9:.1f}-{rc['FER_ci_upper']*1e9:.1f}]e-9\nSER = {rc['SER']:.2e}",
                xy=(2.50, rc['FER']),
                xytext=(30, 84), textcoords='offset points',
                fontsize=7.1, fontweight='bold', color='#512e5f', ha='left',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#f4ecf7', edgecolor='#8e44ad', alpha=0.92, lw=0.8),
                arrowprops=dict(arrowstyle="->", color='#8e44ad', lw=1.2, shrinkA=3, shrinkB=3)
            )

    # 7. QB2 2.50 dB point (Error Floor Emergence) - Unclipped - LOWER stack in upper-right corridor
    pt_250 = qb2_valid_fer[qb2_valid_fer['EbN0_dB'] == 2.50]
    if not pt_250.empty:
        r25 = pt_250.iloc[0]
        ax2.annotate(
            f"QB2 2.50 dB (Unclipped):\n{int(r25['Total_Block_Errors'])} FE ({r25['Total_Blocks']/1e9:.1f}B blks)\nFER = {r25['FER']:.2e} [{r25['FER_ci_lower']*1e9:.1f}-{r25['FER_ci_upper']*1e9:.1f}]e-9\nSER = {r25['SER']:.2e}",
            xy=(2.50, r25['FER']),
            xytext=(30, 36), textcoords='offset points',
            fontsize=7.1, fontweight='bold', color='#145a32', ha='left',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#e8f8f5', edgecolor='#1e8449', alpha=0.92, lw=0.8),
            arrowprops=dict(arrowstyle="->", color='#1e8449', lw=1.2, shrinkA=3, shrinkB=3)
        )

    # 8. QB2 2.60 dB point (Completed 10B Run) - placed in right margin
    pt_260 = qb2_valid_fer[qb2_valid_fer['EbN0_dB'] == 2.60]
    if not pt_260.empty:
        r26 = pt_260.iloc[0]
        ax2.annotate(
            f"QB2 2.60 dB (Unclipped):\n{int(r26['Total_Block_Errors'])} FE ({r26['Total_Blocks']/1e9:.1f}B blks, completed)\nFER = {r26['FER']:.2e} [{r26['FER_ci_lower']*1e10:.1f}-{r26['FER_ci_upper']*1e10:.1f}]e-10\nSER = {r26['SER']:.2e} (Trapping Set)",
            xy=(2.60, r26['FER']),
            xytext=(16, 10), textcoords='offset points',
            fontsize=7.1, fontweight='bold', color='#145a32', ha='left',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#e8f8f5', edgecolor='#1e8449', alpha=0.92, lw=0.8),
            arrowprops=dict(arrowstyle="->", color='#1e8449', lw=1.2, shrinkA=3, shrinkB=3)
        )

    plt.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved combined waterfall plot to: {output_png}")
    for extra_png in [scalar_folder_png, jpl_folder_png, doc_fig_png]:
        dir_name = os.path.dirname(extra_png)
        if os.path.exists(dir_name):
            fig.savefig(extra_png, dpi=300, bbox_inches='tight')
            print(f"[OK] Saved copy to: {extra_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
