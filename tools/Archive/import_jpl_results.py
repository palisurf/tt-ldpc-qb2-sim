#!/usr/bin/env python3
"""
import_jpl_results.py

Parser and import utility for historical JPL LDPC simulation result files
(e.g., Results/JPL Results - Read Only/AR4JA_r45_4c_128c_r12.txt).

File format specification:
  - Header / metadata lines are prefixed with '%'
  - Each data line contains 6 whitespace-delimited fields:
      1. SNR (dB)            : Channel Es/N0 in dB
      2. FER                 : Frame Error Rate recorded by simulator
      3. Raw Symbol Errs     : Accumulated symbol / bit errors
      4. Raw Frame Errors    : Accumulated block / frame errors
      5. Blocks Run          : Total codewords / blocks simulated
      6. Total Iterations    : Sum of decoding iterations across all blocks
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple


def parse_jpl_header(lines) -> Dict[str, Any]:
    """Extracts simulation parameters and metadata from header comment lines."""
    metadata: Dict[str, Any] = {}
    for line in lines:
        line_clean = line.strip()
        if not line_clean.startswith('%'):
            continue
        text = line_clean.lstrip('%').strip()
        if 'created' in text.lower():
            metadata['created'] = text
        elif 'code description in directory' in text.lower():
            metadata['code_directory'] = text
        elif 'number of iterations:' in text.lower():
            m = re.search(r'number of iterations:\s*(\d+)', text, re.IGNORECASE)
            if m:
                metadata['max_iterations'] = int(m.group(1))
        elif 'number of blocks run at each snr value:' in text.lower():
            m = re.search(r'number of blocks run at each snr value:\s*(\d+)', text, re.IGNORECASE)
            if m:
                metadata['max_blocks'] = int(m.group(1))
        elif 'blocklength:' in text.lower():
            m = re.search(r'blocklength:\s*(\d+)', text, re.IGNORECASE)
            if m:
                metadata['blocklength'] = int(m.group(1))
        elif 'desired snr low:' in text.lower():
            m = re.search(r'desired snr low:\s*([-+]?\d*\.?\d+)', text, re.IGNORECASE)
            if m:
                metadata['snr_low_db'] = float(m.group(1))
        elif 'desired snr high:' in text.lower():
            m = re.search(r'desired snr high:\s*([-+]?\d*\.?\d+)', text, re.IGNORECASE)
            if m:
                metadata['snr_high_db'] = float(m.group(1))
        elif 'desired snr stepsize' in text.lower():
            m = re.search(r'desired snr stepsize\s*:\s*([-+]?\d*\.?\d+)', text, re.IGNORECASE)
            if m:
                metadata['snr_step_db'] = float(m.group(1))
    return metadata


def poisson_ci_95(k: float) -> Tuple[float, float]:
    """
    Computes 95% confidence interval for Poisson event count k.
    Uses exact chi2 quantiles if scipy is available,
    otherwise uses exact precomputed table for common counts
    and Garwood approximation for arbitrary k.
    """
    try:
        from scipy.stats import chi2
        low = 0.5 * chi2.ppf(0.025, 2 * k) if k > 0 else 0.0
        high = 0.5 * chi2.ppf(0.975, 2 * (k + 1))
        return float(low), float(high)
    except ImportError:
        table = {
            1: (0.0253, 5.5716),
            2: (0.2422, 7.2247),
            3: (0.6187, 8.7673),
            4: (1.0899, 10.2415),
            5: (1.6235, 11.6683),
            10: (4.7954, 18.3904),
            12: (6.1996, 20.9616),
            20: (12.2166, 30.8884),
            25: (16.1824, 36.9042),
            50: (37.1135, 65.9180),
        }
        if int(k) in table:
            return table[int(k)]
        z = 1.95996
        low = max(0.0, k * (1.0 - 1.0 / (9.0 * k) - z / (3.0 * np.sqrt(k)))**3) if k > 0 else 0.0
        k_p = k + 1
        high = k_p * (1.0 - 1.0 / (9.0 * k_p) + z / (3.0 * np.sqrt(k_p)))**3
        return float(low), float(high)


def load_jpl_results(
    filepath: str,
    code_rate: float = 0.5,
    block_length: int = 2560,
    include_commented: bool = False,
    calculate_ebn0: bool = True
) -> pd.DataFrame:
    """
    Parses a JPL simulation results text file into a pandas DataFrame.

    Parameters:
      filepath (str): Path to the JPL text result file.
      code_rate (float): Code rate R (default: 0.5 for AR4JA Rate-1/2).
                         Used to convert Es/N0 to Eb/N0 via: Eb/N0 = Es/N0 + 10*log10(1/R).
      block_length (int): Total block length in symbols (default: 2560).
                          Used to compute BER = Raw_Symbol_Errors / (Blocks_Run * block_length).
      include_commented (bool): If True, also parses commented-out data rows
                                (e.g. %-2.9014 ...). Default: False.
      calculate_ebn0 (bool): If True, computes and includes an 'EbN0_dB' column.

    Returns:
      pd.DataFrame: Structured table with columns:
        - SNR_dB: Channel Es/N0 in dB
        - EbN0_dB: Converted Eb/N0 in dB (if calculate_ebn0=True)
        - BER: Bit Error Rate (Raw_Symbol_Errors / (Blocks_Run * block_length))
        - FER: Frame Error Rate from file
        - FER_calc: Exact computed FER (Raw_Frame_Errors / Blocks_Run)
        - Raw_Symbol_Errors: Total symbol/bit errors
        - Raw_Frame_Errors: Total frame/block errors
        - Blocks_Run: Total blocks simulated
        - Total_Iterations: Cumulative decoder iterations
        - Avg_Iterations: Average iterations per block (Total_Iterations / Blocks_Run)
        - Is_Commented: Boolean flag (True if row was commented out in original file)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"JPL results file not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    metadata = parse_jpl_header(lines)

    rows = []
    for line_idx, line in enumerate(lines, 1):
        line_s = line.strip()
        if not line_s:
            continue

        is_comment = line_s.startswith('%')

        if is_comment and not include_commented:
            continue

        token_source = line_s.lstrip('%').strip() if is_comment else line_s
        tokens = token_source.split()

        # We expect at least 6 numeric tokens for a data line
        if len(tokens) >= 6:
            try:
                snr = float(tokens[0])
                fer = float(tokens[1])
                sym_errs = float(tokens[2])
                frame_errs = float(tokens[3])
                blocks = float(tokens[4])
                iters = float(tokens[5])

                # Protect against lines that happened to have 6 non-numeric tokens
                rows.append({
                    'SNR_dB': snr,
                    'FER': fer,
                    'Raw_Symbol_Errors': int(sym_errs) if sym_errs.is_integer() else sym_errs,
                    'Raw_Frame_Errors': int(frame_errs) if frame_errs.is_integer() else frame_errs,
                    'Blocks_Run': int(blocks) if blocks.is_integer() else blocks,
                    'Total_Iterations': int(iters) if iters.is_integer() else iters,
                    'Is_Commented': is_comment,
                    'Line_Number': line_idx,
                })
            except ValueError:
                # Not a numeric data line (e.g. comment text with >=6 words)
                continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Exact computed FER: preserves precision when file recorded 0 for small FERs
    df['FER_calc'] = np.where(df['Blocks_Run'] > 0, df['Raw_Frame_Errors'] / df['Blocks_Run'], 0.0)

    # Computed BER: Raw Symbol Errors / (Blocks Run * block_length)
    df['BER'] = np.where(
        (df['Blocks_Run'] > 0) & (block_length > 0),
        df['Raw_Symbol_Errors'] / (df['Blocks_Run'] * block_length),
        0.0
    )

    # Compute 95% Confidence Intervals (Poisson / Clopper-Pearson)
    fer_ci_low = []
    fer_ci_high = []
    ber_ci_low = []
    ber_ci_high = []

    for _, r in df.iterrows():
        k = r['Raw_Frame_Errors']
        n = r['Blocks_Run']
        fer = r['FER_calc']
        ber = r['BER']
        if n > 0 and k > 0:
            k_l, k_h = poisson_ci_95(k)
            fer_ci_low.append(k_l / n)
            fer_ci_high.append(k_h / n)
            ber_ci_low.append(ber * (k_l / k))
            ber_ci_high.append(ber * (k_h / k))
        else:
            fer_ci_low.append(0.0)
            fer_ci_high.append(0.0)
            ber_ci_low.append(0.0)
            ber_ci_high.append(0.0)

    df['FER_ci_lower'] = fer_ci_low
    df['FER_ci_upper'] = fer_ci_high
    df['BER_ci_lower'] = ber_ci_low
    df['BER_ci_upper'] = ber_ci_high

    df['FER_err_minus'] = np.maximum(0.0, df['FER_calc'] - df['FER_ci_lower'])
    df['FER_err_plus'] = np.maximum(0.0, df['FER_ci_upper'] - df['FER_calc'])
    df['BER_err_minus'] = np.maximum(0.0, df['BER'] - df['BER_ci_lower'])
    df['BER_err_plus'] = np.maximum(0.0, df['BER_ci_upper'] - df['BER'])

    # Average iterations per codeword
    df['Avg_Iterations'] = np.where(df['Blocks_Run'] > 0, df['Total_Iterations'] / df['Blocks_Run'], 0.0)

    # Convert Es/N0 (dB) to Eb/N0 (dB): Eb/N0 = Es/N0 + 10*log10(1/R)
    if calculate_ebn0 and code_rate > 0:
        es_to_eb_offset = 10.0 * np.log10(1.0 / code_rate)
        df['EbN0_dB'] = df['SNR_dB'] + es_to_eb_offset

    # Reorder columns logically
    cols = ['SNR_dB']
    if 'EbN0_dB' in df.columns:
        cols.append('EbN0_dB')
    cols += [
        'BER', 'BER_ci_lower', 'BER_ci_upper', 'BER_err_minus', 'BER_err_plus',
        'FER', 'FER_calc', 'FER_ci_lower', 'FER_ci_upper', 'FER_err_minus', 'FER_err_plus',
        'Raw_Symbol_Errors', 'Raw_Frame_Errors', 'Blocks_Run',
        'Total_Iterations', 'Avg_Iterations'
    ]
    if include_commented:
        cols.append('Is_Commented')

    df = df[cols].sort_values(by='SNR_dB').reset_index(drop=True)
    df.attrs['metadata'] = metadata
    return df


def to_qb2_csv(df: pd.DataFrame, output_csv: str) -> None:
    """Exports DataFrame to QB2 simulation CSV format."""
    out_df = pd.DataFrame({
        'EbN0_dB': df['EbN0_dB'].round(2) if 'EbN0_dB' in df.columns else df['SNR_dB'].round(2),
        'Total_Blocks': df['Blocks_Run'],
        'Total_Bit_Errors': df['Raw_Symbol_Errors'],
        'Total_Block_Errors': df['Raw_Frame_Errors'],
        'BER': df['BER'],
        'FER': df['FER_calc'],
    })
    out_df.to_csv(output_csv, index=False)
    print(f"[OK] Exported QB2-compatible CSV to: {output_csv}")


def main():
    default_path = os.path.join("Results", "JPL Results - Read Only", "AR4JA_r45_4c_128c_r12.txt")
    parser = argparse.ArgumentParser(
        description="Import and parse historical JPL LDPC simulation result files."
    )
    parser.add_argument(
        "file",
        nargs="?",
        default=default_path,
        help=f"Path to JPL results text file (default: {default_path})"
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=0.5,
        help="Code rate R for Es/N0 -> Eb/N0 conversion (default: 0.5 for Rate-1/2)"
    )
    parser.add_argument(
        "--block-length",
        type=int,
        default=2560,
        help="Total block length in symbols for BER calculation (default: 2560)"
    )
    parser.add_argument(
        "--include-commented",
        action="store_true",
        help="Include commented-out data rows prefixed with '%%'"
    )
    parser.add_argument(
        "--to-csv",
        type=str,
        default=None,
        help="Save parsed data to specified CSV file"
    )
    parser.add_argument(
        "--qb2-format",
        action="store_true",
        help="Format exported CSV to match QB2 simulation output format"
    )

    args = parser.parse_args()

    print(f"Loading JPL simulation data from: {args.file}")
    df = load_jpl_results(
        args.file,
        code_rate=args.rate,
        block_length=args.block_length,
        include_commented=args.include_commented
    )

    meta = df.attrs.get('metadata', {})
    if meta:
        print("\n--- Header Metadata ---")
        for k, v in meta.items():
            print(f"  {k}: {v}")

    print(f"\n--- Parsed Data ({len(df)} points, block_length={args.block_length}) ---")
    pd.set_option('display.max_columns', 12)
    pd.set_option('display.width', 140)
    pd.set_option('display.float_format', lambda x: f"{x:.4e}" if (abs(x) < 1e-3 and x != 0) else f"{x:.4f}")
    print(df.to_string(index=False))

    if args.to_csv:
        if args.qb2_format:
            to_qb2_csv(df, args.to_csv)
        else:
            df.to_csv(args.to_csv, index=False)
            print(f"\n[OK] Saved DataFrame to: {args.to_csv}")


if __name__ == "__main__":
    main()
