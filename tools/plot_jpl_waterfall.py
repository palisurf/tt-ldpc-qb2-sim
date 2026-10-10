#!/usr/bin/env python3
"""
plot_jpl_waterfall.py

Plots Computed BER and Computed FER vs Eb/N0 with 95% Poisson Confidence Intervals
(error bars) for the reprocessed historical JPL AR4JA LDPC benchmark
(AR4JA_r45_4c_128c_r12, 200 iterations, 2005.11.18).

Only active (uncommented) lines are included. Commented lines are excluded.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from import_jpl_results import load_jpl_results


def create_jpl_waterfall_plot(
    data_file: str = "Results/JPL Results - Read Only/AR4JA_r45_4c_128c_r12.txt",
    output_png: str = "Results/JPL Results - Read Only/jpl_ar4ja_r12_waterfall_iter200.png",
    block_length: int = 2560,
    code_rate: float = 0.5
):
    # Strictly load only uncommented lines (excludes all lines prefixed with '%')
    df = load_jpl_results(
        data_file,
        code_rate=code_rate,
        block_length=block_length,
        include_commented=False
    )

    print(f"Loaded {len(df)} active data points (all '%' commented lines excluded).")

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5), dpi=300)

    # Color Palette
    c_fer_line = '#1a5276'   # Deep Navy
    c_fer_mark = '#2980b9'   # Blue
    c_ber_line = '#900c3f'   # Deep Burgundy
    c_ber_mark = '#c0392b'   # Crimson

    # Filter points where error rate > 0 for log plot
    valid_fer = df[df['FER_calc'] > 0].copy()
    valid_ser = df[df['BER'] > 0].copy()

    for ax in (ax1, ax2):
        # 1. FER with 95% Poisson confidence error bars
        ax.errorbar(
            valid_fer['EbN0_dB'], valid_fer['FER_calc'],
            yerr=[valid_fer['FER_err_minus'], valid_fer['FER_err_plus']],
            fmt='-o', color=c_fer_line, ecolor=c_fer_mark, elinewidth=1.3,
            capsize=3.5, capthick=1.1,
            markerfacecolor=c_fer_mark, markeredgecolor='white',
            markeredgewidth=1.0, linewidth=2.2, markersize=7.0,
            label=r'$\mathbf{JPL\ [Flood\ Decoder]\ FER}\ (\pm 95\%\ \mathrm{CI})$', zorder=4
        )

        # 2. SER with 95% Poisson confidence error bars
        ax.errorbar(
            valid_ser['EbN0_dB'], valid_ser['BER'],
            yerr=[valid_ser['BER_err_minus'], valid_ser['BER_err_plus']],
            fmt='--s', color=c_ber_line, ecolor=c_ber_mark, elinewidth=1.2,
            capsize=3.0, capthick=1.0,
            markerfacecolor=c_ber_mark, markeredgecolor='white',
            markeredgewidth=1.0, linewidth=1.9, markersize=6.0,
            label=r'$\mathbf{JPL\ [Flood\ Decoder]\ SER}\ (\pm 95\%\ \mathrm{CI})$', zorder=4
        )

        ax.set_yscale('log')
        ax.grid(True, which='major', linestyle='-', linewidth=0.75, color='#d0d7de', alpha=0.8)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='#e1e4e8', alpha=0.6)
        ax.set_ylabel('Error Probability (FER / SER)', fontsize=12, fontweight='bold', color='#24292f')

    # ================= PANEL 1: Full Range (0.0 to 2.45 dB) =================
    ax1.set_title('JPL AR4JA Rate-1/2: Full Waterfall with 95% Confidence Bars', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel(r'$E_b/N_0$ [dB]', fontsize=12, fontweight='bold', color='#24292f')
    ax1.set_xlim(-0.05, 2.45)
    ax1.set_ylim(1e-11, 2.0)
    ax1.legend(loc='lower left', fontsize=9.0, frameon=True, framealpha=0.92, facecolor='white', edgecolor='#d0d7de')

    # Metadata callout box
    info_box = (
        r"$\mathbf{Code}$: CCSDS AR4JA Rate-1/2" + "\n"
        r"$N=2560, M=1536, K=1024$" + "\n"
        r"$\mathbf{Decoder}$: Approximate-Min* (Flood Schedule)" + "\n"
        r"$\mathbf{Max\ Iterations}$: 200" + "\n"
        r"$\mathbf{Active\ Points}$: 24 (Uncommented)" + "\n"
        r"$\mathbf{SER\ Metric}$: Symbol Error Rate (Scaled to $N=2560$)" + "\n"
        r"$\mathbf{Confidence}$: 95% Poisson / Binomial CI" + "\n"
        r"$\mathbf{Source}$: Dr. Christopher R. Jones (JPL)" + "\n"
        r"$\mathbf{Date}$: November 18, 2005"
    )
    ax1.text(
        0.04, 0.44, info_box, transform=ax1.transAxes,
        fontsize=8.8, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.55', facecolor='#f8f9fa', edgecolor='#bdc3c7', alpha=0.95)
    )

    # ================= PANEL 2: Zoomed Error Floor (1.2 to 2.50 dB) =================
    ax2.set_title(r'Ultra-Deep Floor Transition with Error Bars Down to $10^{-10}$', fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel(r'$E_b/N_0$ [dB]', fontsize=12, fontweight='bold', color='#24292f')
    ax2.set_xlim(1.20, 2.50)
    ax2.set_ylim(1e-11, 1e-1)
    ax2.legend(loc='upper right', fontsize=9.5, frameon=True, framealpha=0.92, facecolor='white', edgecolor='#d0d7de')

    # Annotate Deep Floor Points with error counts & confidence spans
    deep_points = [
        (2.2091, 3.00e-8, "2.21 dB: 3 FE (100M blks)\nFER: [0.62 - 8.77]e-8", (-155, 12), 'right'),
        (2.2603, 1.02e-8, "2.26 dB: 3 FE (295M blks)\nFER: [0.21 - 2.98]e-8", (18, 16), 'left'),
        (2.3228, 3.19e-9, "2.32 dB: 3 FE (940M blks)\nFER: [0.66 - 9.33]e-9", (-155, -20), 'right'),
        (2.3853, 9.52e-10, "2.39 dB: 2 FE (2.1B blks)\nFER = 9.52e-10 [0.12 - 3.44]e-9\nSER = 1.74e-10 [0.21 - 6.30]e-10", (16, 25), 'left'),
    ]

    for ebn0, fer, text, offset, align in deep_points:
        ax2.annotate(
            text,
            xy=(ebn0, fer),
            xytext=offset,
            textcoords='offset points',
            fontsize=8.2,
            fontweight='bold',
            color='#1a5276',
            ha=align,
            arrowprops=dict(arrowstyle="->", color='#2980b9', lw=1.2, shrinkA=3, shrinkB=3)
        )

    plt.tight_layout()

    # Save to requested destination directory and primary Results directory
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved waterfall plot with error bars to: {output_png}")

    alt_png = "Results/jpl_ar4ja_r12_waterfall_iter200.png"
    fig.savefig(alt_png, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved duplicate to: {alt_png}")

    plt.close(fig)


if __name__ == "__main__":
    create_jpl_waterfall_plot()
