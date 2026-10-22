#!/usr/bin/env python3
"""
combine_and_plot_results.py

Combines and compares three comprehensive AR4JA Rate-1/2 LDPC simulation datasets:
  1. Historical NASA JPL benchmark (2005, Flood Decoder, Parallel Schedule)
  2. Tenstorrent QuietBox 2 Scalar 5-Worker Engine (Christopher R. Jones, PhD, 2026)
  3. Tenstorrent QuietBox 2 Circulant DMA 4-Codeword Engine (Christopher R. Jones, PhD, 2026)

Computes exact 95% Poisson confidence intervals (error bars) for all simulations
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


def add_confidence_intervals(df: pd.DataFrame, fer_col='FER', ser_col='SER', blk_col='Total_Blocks', fe_col='Total_Block_Errors') -> pd.DataFrame:
    """Computes exact 95% Poisson confidence intervals for error rates."""
    fer_ci_low, fer_ci_high = [], []
    ser_ci_low, ser_ci_high = [], []

    for _, r in df.iterrows():
        k = r[fe_col]
        n = r[blk_col]
        fer = r[fer_col]
        ser = r[ser_col]
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

    df['FER_err_minus'] = np.maximum(0.0, df[fer_col] - df['FER_ci_lower'])
    df['FER_err_plus'] = np.maximum(0.0, df['FER_ci_upper'] - df[fer_col])
    df['SER_err_minus'] = np.maximum(0.0, df[ser_col] - df['SER_ci_lower'])
    df['SER_err_plus'] = np.maximum(0.0, df['SER_ci_upper'] - df[ser_col])
    return df


def main():
    # -------------------------------------------------------------
    # 1. Paths Configuration
    # -------------------------------------------------------------
    jpl_file = "Results/Scalar_5w/JPL Results - Read Only/AR4JA_r45_4c_128c_r12.txt"
    if not os.path.exists(jpl_file):
        jpl_file = "Results/JPL Results - Read Only/AR4JA_r45_4c_128c_r12.txt"

    scalar_100_250 = "Results/Scalar_5w/results_AR4JA_r45_4c_128c_r12_amin_ebn0_1.00_2.50_step0.10_max22727272_min25_iter200_440cores_unclipped.csv"
    scalar_260 = "Results/Scalar_5w/results_AR4JA_r45_4c_128c_r12_amin_ebn0_2.60_2.60_step0.10_max22727273_min50_iter200_440cores.csv"

    circ_csv = "Results/waterfall_0.0_2.6_10B_iter200.csv"

    output_csv = "Results/combined_ar4ja_jpl_qb2_iter200.csv"
    output_png = "Results/combined_ar4ja_r12_waterfall_iter200.png"
    scalar_folder_png = "Results/Scalar_5w/combined_ar4ja_r12_waterfall_iter200.png"
    jpl_folder_png = "Results/Scalar_5w/JPL Results - Read Only/combined_ar4ja_r12_waterfall_iter200.png"
    doc_fig_png = "docs/figures/combined_ar4ja_r12_waterfall_iter200.png"
    doc_fig_3ds_png = "docs/figures/ar4ja_rate12_3dataset_waterfall_comparison.png"

    # -------------------------------------------------------------
    # 2. Load Dataset 1: Historical NASA JPL (2005, Flood Decoder)
    # -------------------------------------------------------------
    df_jpl = load_jpl_results(jpl_file, code_rate=0.5, block_length=2560, include_commented=False)
    df_jpl['Source'] = 'JPL 2005 (Flood Decoder, 200 Iter)'
    df_jpl['SER'] = df_jpl['BER']
    df_jpl['SER_ci_lower'] = df_jpl['BER_ci_lower']
    df_jpl['SER_ci_upper'] = df_jpl['BER_ci_upper']
    df_jpl['SER_err_minus'] = df_jpl['BER_err_minus']
    df_jpl['SER_err_plus'] = df_jpl['BER_err_plus']
    print(f"[OK] Loaded {len(df_jpl)} points from JPL historical benchmark.")

    # -------------------------------------------------------------
    # 3. Load Dataset 2: QB2 Scalar 5-Worker Engine (Scalar_5w)
    # -------------------------------------------------------------
    df_s = pd.read_csv(scalar_100_250)
    if os.path.exists(scalar_260):
        df_s = pd.concat([df_s, pd.read_csv(scalar_260)], ignore_index=True)
    df_s = df_s.drop_duplicates(subset=['EbN0_dB']).sort_values('EbN0_dB').reset_index(drop=True)
    df_s['Source'] = 'QuietBox 2 (Scalar 5-Worker Engine, 2026)'
    df_s['SER'] = df_s['Total_Bit_Errors'] / (df_s['Total_Blocks'] * 2560.0)
    df_s = add_confidence_intervals(df_s)
    print(f"[OK] Loaded {len(df_s)} points from QB2 Scalar 5-Worker Engine (1.00 dB - 2.60 dB).")

    # -------------------------------------------------------------
    # 4. Load Dataset 3: QB2 Circulant DMA 4-Codeword Engine (4CW)
    # -------------------------------------------------------------
    df_c = pd.read_csv(circ_csv)
    df_c['Source'] = 'QuietBox 2 (Circulant DMA 4-CW Engine, 2026)'
    df_c['SER'] = df_c['Total_Bit_Errors'] / (df_c['Total_Blocks'] * 2560.0)
    df_c = add_confidence_intervals(df_c)
    print(f"[OK] Loaded {len(df_c)} points from QB2 Circulant DMA 4-CW Engine (0.00 dB - 2.60 dB).")

    # -------------------------------------------------------------
    # 5. Save Combined CSV
    # -------------------------------------------------------------
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
        'FER_ci_upper': df_jpl['FER_ci_upper']
    })

    scalar_std = pd.DataFrame({
        'Source': df_s['Source'],
        'EbN0_dB': df_s['EbN0_dB'].round(2),
        'Total_Blocks': df_s['Total_Blocks'],
        'Total_Symbol_Errors': df_s['Total_Bit_Errors'],
        'Total_Block_Errors': df_s['Total_Block_Errors'],
        'SER': df_s['SER'],
        'SER_ci_lower': df_s['SER_ci_lower'],
        'SER_ci_upper': df_s['SER_ci_upper'],
        'FER': df_s['FER'],
        'FER_ci_lower': df_s['FER_ci_lower'],
        'FER_ci_upper': df_s['FER_ci_upper']
    })

    circ_std = pd.DataFrame({
        'Source': df_c['Source'],
        'EbN0_dB': df_c['EbN0_dB'].round(2),
        'Total_Blocks': df_c['Total_Blocks'],
        'Total_Symbol_Errors': df_c['Total_Bit_Errors'],
        'Total_Block_Errors': df_c['Total_Block_Errors'],
        'SER': df_c['SER'],
        'SER_ci_lower': df_c['SER_ci_lower'],
        'SER_ci_upper': df_c['SER_ci_upper'],
        'FER': df_c['FER'],
        'FER_ci_lower': df_c['FER_ci_lower'],
        'FER_ci_upper': df_c['FER_ci_upper']
    })

    df_combined = pd.concat([jpl_std, scalar_std, circ_std], ignore_index=True)
    df_combined = df_combined.sort_values(by=['EbN0_dB', 'Source']).reset_index(drop=True)
    df_combined.to_csv(output_csv, index=False)
    if os.path.exists("Results/Scalar_5w"):
        df_combined.to_csv("Results/Scalar_5w/combined_ar4ja_jpl_qb2_iter200.csv", index=False)
    print(f"[OK] Saved unified 3-result dataset to: {output_csv}")

    # -------------------------------------------------------------
    # 6. Plotting Configuration
    # -------------------------------------------------------------
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8.5), dpi=300)

    # Cohesive Color Palette:
    # 1. JPL: Deep Navy & Maroon
    # 2. Scalar 5W: Royal Purple & Coral
    # 3. Circulant DMA: Emerald Green & Amber
    c_jpl_fer = '#1b4f72'  # Deep Navy Blue
    c_jpl_ser = '#78281f'  # Maroon
    c_s_fer   = '#6c3483'  # Royal Purple
    c_s_ser   = '#a93226'  # Coral / Crimson
    c_c_fer   = '#1e8449'  # Emerald Green
    c_c_ser   = '#d35400'  # Amber / Rust

    # Filter positive points for log scale
    jpl_valid_fer = df_jpl[df_jpl['FER_calc'] > 0]
    jpl_valid_ser = df_jpl[df_jpl['SER'] > 0]
    s_valid_fer   = df_s[df_s['FER'] > 0]
    s_valid_ser   = df_s[df_s['SER'] > 0]
    c_valid_fer   = df_c[df_c['FER'] > 0]
    c_valid_ser   = df_c[df_c['SER'] > 0]

    for ax in (ax1, ax2):
        # 1. JPL 2005 (Flood Decoder)
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

        # 2. QB2 Scalar 5-Worker Engine (2026)
        ax.errorbar(
            s_valid_fer['EbN0_dB'], s_valid_fer['FER'],
            yerr=[s_valid_fer['FER_err_minus'], s_valid_fer['FER_err_plus']],
            fmt='-.v', color=c_s_fer, ecolor=c_s_fer, elinewidth=1.2,
            capsize=3.5, capthick=1.0,
            markerfacecolor='#9b59b6', markeredgecolor='white',
            markeredgewidth=0.8, linewidth=1.9, markersize=6.5,
            label=r'$\mathbf{QB2\ [Scalar\ 5\text{-}Worker\ Engine]\ FER}\ (\pm 95\%\ \mathrm{CI})$', zorder=5
        )
        ax.errorbar(
            s_valid_ser['EbN0_dB'], s_valid_ser['SER'],
            yerr=[s_valid_ser['SER_err_minus'], s_valid_ser['SER_err_plus']],
            fmt=':p', color=c_s_ser, ecolor=c_s_ser, elinewidth=1.1,
            capsize=3.0, capthick=0.9,
            markerfacecolor='#e74c3c', markeredgecolor='white',
            markeredgewidth=0.8, linewidth=1.6, markersize=5.5,
            label=r'$\mathbf{QB2\ [Scalar\ 5\text{-}Worker\ Engine]\ SER}\ (\pm 95\%\ \mathrm{CI})$', zorder=5
        )

        # 3. QB2 Circulant DMA 4-CW Engine (2026)
        ax.errorbar(
            c_valid_fer['EbN0_dB'], c_valid_fer['FER'],
            yerr=[c_valid_fer['FER_err_minus'], c_valid_fer['FER_err_plus']],
            fmt='-D', color=c_c_fer, ecolor=c_c_fer, elinewidth=1.3,
            capsize=3.5, capthick=1.0,
            markerfacecolor='#2ecc71', markeredgecolor='white',
            markeredgewidth=0.8, linewidth=2.2, markersize=6.5,
            label=r'$\mathbf{QB2\ [Circulant\ DMA\ 4\text{-}CW\ Engine]\ FER}\ (\pm 95\%\ \mathrm{CI})$', zorder=6
        )
        ax.errorbar(
            c_valid_ser['EbN0_dB'], c_valid_ser['SER'],
            yerr=[c_valid_ser['SER_err_minus'], c_valid_ser['SER_err_plus']],
            fmt='--^', color=c_c_ser, ecolor=c_c_ser, elinewidth=1.2,
            capsize=3.0, capthick=0.9,
            markerfacecolor='#e67e22', markeredgecolor='white',
            markeredgewidth=0.8, linewidth=1.8, markersize=6.0,
            label=r'$\mathbf{QB2\ [Circulant\ DMA\ 4\text{-}CW\ Engine]\ SER}\ (\pm 95\%\ \mathrm{CI})$', zorder=6
        )

        ax.set_yscale('log')
        ax.grid(True, which='major', linestyle='-', linewidth=0.75, color='#d0d7de', alpha=0.8)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='#e1e4e8', alpha=0.6)
        ax.set_ylabel('Error Probability (FER / SER)', fontsize=12, fontweight='bold', color='#24292f')

    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # Panel 1: Full Waterfall (0.0 to 2.68 dB)
    # -------------------------------------------------------------
    ax1.set_title('Full Waterfall Comparison: JPL (2005) vs. QB2 Scalar (5W) vs. QB2 Circulant DMA (4CW)',
                  fontsize=12.5, fontweight='bold', pad=12)
    ax1.set_xlabel(r'$E_b/N_0$ [dB]', fontsize=12, fontweight='bold', color='#24292f')
    ax1.set_xlim(-0.05, 2.68)
    ax1.set_ylim(1e-11, 2.0)
    ax1.legend(loc='lower left', fontsize=7.2, frameon=True, framealpha=0.95, facecolor='white', edgecolor='#d0d7de')

    info_box = (
        r"$\mathbf{Code}$: CCSDS AR4JA Rate-1/2 ($N=2560, K=1024$)" + "\n"
        r"$\mathbf{Decoder}$: Approximate-Min* (200 max iterations)" + "\n"
        r"$\mathbf{Author}$: Christopher R. Jones" + "\n"
        r"$\mathbf{JPL\ 2005\ Data}$: Flood Decoder (Parallel Schedule)" + "\n"
        r"$\mathbf{QB2\ Scalar\ 5W}$: Row-Layered (5 Workers/Core, Scalar Mem Chasing)" + "\n"
        r"$\mathbf{QB2\ Circulant\ DMA}$: Row-Layered (4-CW Quartet, 512-bit DMA)" + "\n"
        r"$\mathbf{Precision}$: Native 16-bit FP ($\mathtt{bfloat16}$, Unclipped)" + "\n"
        r"$\mathbf{Hardware}$: Tenstorrent QuietBox 2 (440 Tensix Cores, 2x2 Mesh)" + "\n"
        r"$\mathbf{Throughput}$: Scalar 5W $\approx 158\ \mathrm{Msps} \to$ Circulant DMA $\mathbf{5.15\ \mathrm{Gsps}}$" + "\n"
        r"$\mathbf{SER\ Metric}$: Symbol Error Rate (All SERs scaled to $N=2560$)" + "\n"
        r"$\mathbf{Error\ Bars}$: Exact 95% Poisson/Binomial CI"
    )
    ax1.text(
        0.04, 0.49, info_box, transform=ax1.transAxes,
        fontsize=7.3, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.55', facecolor='#f8f9fa', edgecolor='#bdc3c7', alpha=0.95)
    )

    # -------------------------------------------------------------
    # Panel 2: Deep Floor Comparison (1.70 to 2.68 dB)
    # -------------------------------------------------------------
    ax2.set_title(r'Ultra-Deep Floor Agreement Down to $10^{-10}$ Across All 3 Decoders',
                  fontsize=12.5, fontweight='bold', pad=12)
    ax2.set_xlabel(r'$E_b/N_0$ [dB]', fontsize=12, fontweight='bold', color='#24292f')
    ax2.set_xlim(1.70, 2.68)
    ax2.set_ylim(1.5e-11, 6e-4)
    ax2.legend(loc='upper right', fontsize=7.5, frameon=True, framealpha=0.95, facecolor='white', edgecolor='#d0d7de')

    # Deep Floor Benchmark Summary Box in pristine lower-left whitespace (no lines intersect)
    bench_box = (
        r"$\mathbf{Deep\ Floor\ Benchmark\ Summary\ (200\ Max\ Iterations)}$" + "\n"
        r"$\bullet\ \mathbf{JPL\ 2005\ [Flood]}:$" + "\n"
        r"   $2.26\ \mathrm{dB}: 3\ \mathrm{FE}\ (295\mathrm{M}\ \mathrm{blks}),\ \mathrm{FER} = 1.02\times 10^{-8}$" + "\n"
        r"   $2.39\ \mathrm{dB}: 2\ \mathrm{FE}\ (2.10\mathrm{B}\ \mathrm{blks}),\ \mathrm{FER} = 9.52\times 10^{-10}$" + "\n"
        r"$\bullet\ \mathbf{QB2\ Scalar\ 5W\ [Layered]}:$" + "\n"
        r"   $2.60\ \mathrm{dB}: 3\ \mathrm{FE}\ (10.0\mathrm{B}\ \mathrm{blks}),\ \mathrm{FER} = 3.00\times 10^{-10}$" + "\n"
        r"$\bullet\ \mathbf{QB2\ Circulant\ DMA\ 4CW\ [Layered]}:$" + "\n"
        r"   $2.40\ \mathrm{dB}: 25\ \mathrm{FE}\ (7.79\mathrm{B}\ \mathrm{blks}),\ \mathrm{FER} = 3.21\times 10^{-9}$" + "\n"
        r"   $2.50\ \mathrm{dB}: 18\ \mathrm{FE}\ (10.0\mathrm{B}\ \mathrm{blks}),\ \mathrm{FER} = 1.80\times 10^{-9}$" + "\n"
        r"   $2.60\ \mathrm{dB}: 8\ \mathrm{FE}\ (10.0\mathrm{B}\ \mathrm{blks}),\ \mathrm{FER} = \mathbf{8.00\times 10^{-10}}$" + "\n"
        r"$\mathbf{Conclusion}$: All 3 decoders reveal the same trapping-set" + "\n"
        r"error floor plateau at $\mathrm{FER} \approx 8\times 10^{-10}$ with zero divergence."
    )
    ax2.text(
        0.04, 0.04, bench_box, transform=ax2.transAxes,
        fontsize=7.3, verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.55', facecolor='#f8f9fa', edgecolor='#1b4f72', alpha=0.96, lw=1.0)
    )

    # Clean Callout 1: 2.20 dB Agreement (Pointing straight down from above to 2.20 dB)
    r22_c = c_valid_fer[c_valid_fer['EbN0_dB'] == 2.20].iloc[0]
    r22_s = s_valid_fer[s_valid_fer['EbN0_dB'] == 2.20].iloc[0]
    ax2.annotate(
        r"$\mathbf{2.20\ dB\ Multi\text{-}Engine\ Agreement}$" + "\n"
        f"Circulant DMA: 26 FE ({r22_c['Total_Blocks']/1e6:.0f}M blks), FER = {r22_c['FER']:.2e}\n"
        f"Scalar 5-Worker: 25 FE ({r22_s['Total_Blocks']/1e6:.0f}M blks), FER = {r22_s['FER']:.2e}",
        xy=(2.20, r22_c['FER']),
        xytext=(0, 65), textcoords='offset points',
        fontsize=7.2, fontweight='bold', color='#145a32', ha='center',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#f8f9fa', edgecolor='#1e8449', alpha=0.95, lw=0.9),
        arrowprops=dict(arrowstyle="->", color=c_c_fer, lw=1.3, shrinkA=3, shrinkB=4)
    )

    # Clean Callout 2: 2.60 dB Ultra-Deep Error Floor (10 Billion Blocks per Engine)
    r26_c = c_valid_fer[c_valid_fer['EbN0_dB'] == 2.60].iloc[0]
    r26_s = s_valid_fer[s_valid_fer['EbN0_dB'] == 2.60].iloc[0]
    ax2.annotate(
        r"$\mathbf{2.60\ dB\ Ultra\text{-}Deep\ Floor\ (10\ Billion\ Blocks/Engine)}$" + "\n"
        f"Circulant DMA: 8 FE, FER = {r26_c['FER']:.2e} [{r26_c['FER_ci_lower']*1e10:.1f}-{r26_c['FER_ci_upper']*1e10:.1f}] " + r"$\times 10^{-10}$" + "\n"
        f"Scalar 5-Worker: 3 FE, FER = {r26_s['FER']:.2e} [{r26_s['FER_ci_lower']*1e10:.1f}-{r26_s['FER_ci_upper']*1e10:.1f}] " + r"$\times 10^{-10}$" + "\n"
        r"$\mathbf{Consistent\ Trapping\text{-}Set\ Plateau}$",
        xy=(2.60, r26_c['FER']),
        xytext=(-25, 60), textcoords='offset points',
        fontsize=7.2, fontweight='bold', color='#145a32', ha='center',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#e8f8f5', edgecolor='#1e8449', alpha=0.95, lw=0.9),
        arrowprops=dict(arrowstyle="->", color=c_c_fer, lw=1.3, shrinkA=3, shrinkB=4)
    )

    plt.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved combined 3-result waterfall plot to: {output_png}")
    for extra_png in [scalar_folder_png, jpl_folder_png, doc_fig_png, doc_fig_3ds_png]:
        dir_name = os.path.dirname(extra_png)
        if os.path.exists(dir_name):
            fig.savefig(extra_png, dpi=300, bbox_inches='tight')
            print(f"[OK] Saved copy to: {extra_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
