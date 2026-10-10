#!/usr/bin/env python3
import subprocess
import json
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

def main():
    binary = "./ldpc_sim"
    matrix = "matrices/AR4JA_r45_4c_128c_r12.chinn.out"
    ebn0 = 2.00
    punctured = 512
    min_errors = 30
    batch_size = 100
    max_blocks_per_core = 20000
    max_iter = 200

    gamma_values = [1.00, 1.33, 1.67, 2.00, 2.50, 3.00, 0.00] # 0.00 = unclipped

    output_csv = "Results/gamma_clipping_sweep_2.00dB.csv"
    output_png = "Results/gamma_clipping_sweep_2.00dB.png"
    os.makedirs("Results", exist_ok=True)

    print("==========================================================")
    print("  TENSIORRENT QB2: PROPORTIONAL GAMMA CLIPPING SWEEP      ")
    print(f"  Eb/N0: {ebn0:.2f} dB | Max Iter: {max_iter} | Min Errors: {min_errors}")
    print(f"  Gamma values: {gamma_values}")
    print("==========================================================\n")

    results = []

    for gamma in gamma_values:
        gamma_label = f"{gamma:.2f}" if gamma > 0.0 else "Unclipped"
        print(f"\n>>> Running Gamma = {gamma_label} ...")

        cmd = [
            binary,
            "-c", matrix,
            "-e", str(ebn0),
            "-p", str(punctured),
            "-t", str(min_errors),
            "-b", str(batch_size),
            "-k", str(max_blocks_per_core),
            "-i", str(max_iter)
        ]
        if gamma > 0.0:
            cmd.extend(["-G", str(gamma)])
        else:
            cmd.extend(["-L", "0.0"])

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        final_data = None
        l_max_val = 0.0

        for raw_line in proc.stdout:
            line = raw_line.strip()
            if not line:
                continue
            if "[CONFIG] LLR Clipping" in line:
                print(f"  {line}")
                if "L_ch_max=" in line:
                    try:
                        l_max_val = float(line.split("L_ch_max=")[1].split(",")[0].strip())
                    except:
                        pass
            if line.startswith("{") and "ebn0" in line:
                try:
                    data = json.loads(line)
                    if data.get("progress"):
                        pct = (data['blocks'] / data['max_blocks'] * 100.0) if data.get('max_blocks', 0) > 0 else 0.0
                        sys.stdout.write(
                            f"\r  [Gamma {gamma_label:>9}] Blocks: {data['blocks']:,}/{data['max_blocks']:,} ({pct:5.1f}%) | "
                            f"FE: {data['frame_errors']}/{data['target_errors']} | FER: {data['fer']:.2e} | "
                            f"BER: {data['ber']:.2e} | {data.get('throughput_msps', 0.0):.1f} Msps"
                        )
                        sys.stdout.flush()
                    else:
                        final_data = data
                except json.JSONDecodeError:
                    pass

        proc.wait()
        print()

        if final_data:
            blocks = final_data.get('blocks', 0)
            be = final_data.get('bit_errors', 0)
            fe = final_data.get('frame_errors', 0)
            fer = fe / blocks if blocks > 0 else 0.0
            ber = be / (blocks * 2048) if blocks > 0 else 0.0
            elap = final_data.get('elapsed_sec', 0.0)
            msps = (blocks * 2048 / elap / 1e6) if elap > 0 else 0.0

            results.append({
                'Gamma': gamma,
                'Gamma_Label': gamma_label,
                'L_max': l_max_val,
                'Total_Blocks': blocks,
                'Bit_Errors': be,
                'Block_Errors': fe,
                'BER': ber,
                'FER': fer,
                'Throughput_Msps': msps,
                'Elapsed_Sec': elap
            })
            print(f"  -> Result: FER={fer:.3e}, BER={ber:.3e}, Blocks={blocks:,}, FE={fe}, Time={elap:.1f}s")
        else:
            print(f"  [ERROR] No final data parsed for Gamma={gamma_label}")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n[DONE] CSV saved to {output_csv}")

    # Plot results
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Convert gamma to categorical labels for plotting
    labels = df['Gamma_Label'].tolist()
    x = range(len(labels))

    # Panel 1: FER and BER vs Gamma
    ax1.semilogy(x, df['FER'], 'o-', color='#8e44ad', linewidth=2.2, markersize=8, label='FER (Frame Error Rate)')
    ax1.semilogy(x, df['BER'], 's--', color='#2980b9', linewidth=1.8, markersize=7, label='BER (Bit Error Rate)')
    for i, txt in enumerate(df['FER']):
        ax1.annotate(f"{txt:.2e}", (x[i], df['FER'].iloc[i]), xytext=(0, 8), textcoords="offset points", ha='center', fontsize=9, fontweight='bold', color='#8e44ad')

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax1.set_xlabel(r'Proportional Clipping Ratio $\gamma = L_{\max} / \mu_{\mathrm{LLR}}$', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Error Rate (Log Scale)', fontsize=11, fontweight='bold')
    ax1.set_title(f'CCSDS AR4JA Rate-1/2: Error Rate vs. Clipping Ratio $\gamma$\n($E_b/N_0 = 2.00$ dB, Max Iter = {max_iter}, 440 Tensix Cores)', fontsize=12, fontweight='bold')
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    ax1.legend(fontsize=10, loc='best')

    # Panel 2: Blocks Decoded to Target Errors / Convergence Speed
    ax2.bar(x, df['Total_Blocks'] / 1e6, color='#27ae60', alpha=0.7, edgecolor='black', width=0.5)
    for i, val in enumerate(df['Total_Blocks']):
        ax2.annotate(f"{val/1e6:.2f}M", (x[i], val / 1e6), xytext=(0, 5), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax2.set_xlabel(r'Proportional Clipping Ratio $\gamma = L_{\max} / \mu_{\mathrm{LLR}}$', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Total Blocks Evaluated (Millions)', fontsize=11, fontweight='bold')
    ax2.set_title(f'Blocks Decoded to Collect {min_errors} Errors (Higher = Lower Error Rate)\n(At $E_b/N_0 = 2.00$ dB)', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"[PLOT] Figure saved to {output_png}")

if __name__ == "__main__":
    main()
