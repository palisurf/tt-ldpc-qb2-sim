#!/usr/bin/env python3
import argparse
import subprocess
import json
import os
import sys
import numpy as np

def parse_matrix_header(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Matrix file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
    if not lines:
        raise ValueError(f"Matrix file is empty: {filepath}")
        
    first_tokens = lines[0].split()
    if len(first_tokens) == 3:
        # .chinn.out format:
        # The last row specifies the total matrix dimensions: M (rows/checks), N (cols/variables)
        last_tokens = lines[-1].split()
        M = int(last_tokens[0])
        N = int(last_tokens[1])
        
        # Calculate degrees
        check_deg = [0] * M
        var_deg = [0] * N
        for line in lines[:-1]:
            parts = line.split()
            if len(parts) >= 3 and float(parts[2]) != 0.0:
                r = int(parts[0]) - 1
                c = int(parts[1]) - 1
                if 0 <= r < M:
                    check_deg[r] += 1
                if 0 <= c < N:
                    var_deg[c] += 1
                    
        max_check_deg = max(check_deg) if check_deg else 0
        max_var_deg = max(var_deg) if var_deg else 0
        return N, M, max_var_deg, max_check_deg
    else:
        # Standard .chinn format
        header = first_tokens
        deg_header = lines[1].split()
        N = int(header[0])
        M = int(header[1])
        if len(header) >= 4:
            max_var_deg = int(header[2])
            max_check_deg = int(header[3])
        else:
            max_var_deg = int(deg_header[0])
            max_check_deg = int(deg_header[1])
        return N, M, max_var_deg, max_check_deg

parse_chinn_header = parse_matrix_header

def main():
    parser = argparse.ArgumentParser(description="TT-Metalium LDPC Monte Carlo Simulator Driver")
    parser.add_argument("--chinn", "--matrix", dest="chinn", type=str, required=True, help="Path to matrix file (.chinn or .chinn.out)")
    parser.add_argument("--ebn0_start", type=float, required=True, help="Start Eb/N0 in dB")
    parser.add_argument("--ebn0_end", type=float, required=True, help="End Eb/N0 in dB")
    parser.add_argument("--ebn0_step", type=float, default=0.25, help="Eb/N0 step size in dB")
    parser.add_argument("--punctured", type=int, default=0, help="Number of punctured variable nodes")
    parser.add_argument("--min_errors", "--target_errors", dest="min_errors", type=int, default=50, help="Minimum errored blocks to find per SNR point (default: 50)")
    parser.add_argument("--max_blocks_per_core", "--blocks_per_core", dest="max_blocks_per_core", type=int, default=100000, help="Maximum blocks to simulate per core (default: 100000)")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size per core between error checks (default: 1000)")
    parser.add_argument("--max_iter", "--max_iterations", dest="max_iter", type=int, default=16, help="Maximum decoder iterations per codeword (default: 16)")
    parser.add_argument("--binary", type=str, default="./ldpc_sim", help="Path to compiled C++ binary")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file path (defaults to auto-generated from parameters)")
    parser.add_argument("--single_core", action="store_true", help="Run on a single Tensix core only")
    parser.add_argument("--nms", action="store_true", help="Use Normalized Min-Sum (alpha=0.75) instead of default Approximate-Min*")
    
    args = parser.parse_args()

    N, M, max_vdeg, max_cdeg = parse_matrix_header(args.chinn)
    unpunctured_N = N - args.punctured
    # True punctured code rate
    rate = (N - M) / unpunctured_N

    # Results directory
    results_dir = "Results"
    os.makedirs(results_dir, exist_ok=True)

    # Auto-generate descriptive CSV filename if not explicitly provided
    algo_str = "nms" if args.nms else "amin"
    algo_desc = "Normalized Min-Sum (alpha=0.75)" if args.nms else "Approximate-Min* (Christopher Jones MILCOM 2003)"

    if args.output:
        if os.path.dirname(args.output):
            output_filename = args.output
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        else:
            output_filename = os.path.join(results_dir, args.output)
    else:
        matrix_base = os.path.basename(args.chinn)
        for ext in [".chinn.out", ".chinn", ".out"]:
            if matrix_base.endswith(ext):
                matrix_base = matrix_base[:-len(ext)]
                break
        core_str = "1core" if args.single_core else "440cores"
        output_filename = os.path.join(
            results_dir,
            f"results_{matrix_base}"
            f"_{algo_str}"
            f"_ebn0_{args.ebn0_start:.2f}_{args.ebn0_end:.2f}_step{args.ebn0_step:.2f}"
            f"_max{args.max_blocks_per_core}"
            f"_min{args.min_errors}"
            f"_iter{args.max_iter}"
            f"_{core_str}.csv"
        )

    print("==========================================================")
    print("      Tenstorrent QB2 LDPC Monte Carlo Simulation         ")
    print("==========================================================")
    print(f"Matrix File        : {args.chinn}")
    print(f"Decoder Algorithm  : {algo_desc}")
    print(f"Code Dimensions    : N={N}, M={M} (Rate = {rate:.4f})")
    print(f"Punctured Nodes    : {args.punctured} (Transmitted N = {unpunctured_N} symbols/code)")
    print(f"Eb/N0 Range        : {args.ebn0_start:.2f} dB to {args.ebn0_end:.2f} dB (step {args.ebn0_step:.2f} dB)")
    print(f"Stopping Criteria  : Min {args.min_errors} block errors OR max {args.max_blocks_per_core} blocks/core")
    print(f"Batch Size/Core    : {args.batch_size} blocks per core per evaluation step")
    print(f"Max Decoder Iters  : {args.max_iter}")
    print(f"Output File        : {output_filename}")
    print("==========================================================\n")

    ebn0_points = np.arange(args.ebn0_start, args.ebn0_end + args.ebn0_step / 2.0, args.ebn0_step)

    with open(output_filename, 'w') as out_file:
        out_file.write("EbN0_dB,Total_Blocks,Total_Bit_Errors,Total_Block_Errors,BER,FER\n")

    print(f"{'Eb/N0 (dB)':<10} | {'Blocks Run':<12} | {'Bit Errors':<12} | {'Block Errors':<14} | {'BER':<12} | {'FER':<12} | {'Throughput (Msps)':<18}")
    print("-" * 104)

    for ebn0 in ebn0_points:
        cmd = [
            args.binary,
            "-c", args.chinn,
            "-e", str(ebn0),
            "-p", str(args.punctured),
            "-t", str(args.min_errors),
            "-k", str(args.max_blocks_per_core),
            "-i", str(args.max_iter)
        ]
        if args.single_core:
            cmd.append("-s")
        if args.nms:
            cmd.append("-n")
        if args.batch_size is not None:
            cmd.extend(["-b", str(args.batch_size)])

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            is_tty = sys.stdout.isatty()
            for raw_line in proc.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("{") and "ebn0" in line:
                    try:
                        data = json.loads(line)
                        if data.get("progress"):
                            pct = (data['blocks'] / data['max_blocks'] * 100.0) if data.get('max_blocks', 0) > 0 else 0.0
                            progress_str = (
                                f"  [{ebn0:.2f} dB] Blocks: {data['blocks']:,}/{data['max_blocks']:,} ({pct:5.1f}%) "
                                f"| FE: {data['frame_errors']}/{data['target_errors']} "
                                f"| FER: {data['fer']:.2e} "
                                f"| BER: {data['ber']:.2e} "
                                f"| {data['throughput_msps']:.1f} Msps "
                                f"| {data['elapsed_sec']:.1f}s"
                            )
                            if is_tty:
                                print(f"\r{progress_str:<104}", end="", flush=True)
                            else:
                                print(progress_str, flush=True)
                        else:
                            final_data = data
                    except Exception:
                        pass
                else:
                    if is_tty:
                        print(f"\n{line}")
                    else:
                        print(line)

            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"Simulation process exited with code {proc.returncode}")
            if final_data is None:
                raise RuntimeError("Could not parse final simulation output JSON.")

            blocks = final_data["blocks"]
            bit_errors = final_data["bit_errors"]
            frame_errors = final_data["frame_errors"]
            elapsed_sec = final_data.get("elapsed_sec", 0.0)

            total_symbols = blocks * unpunctured_N
            ber = bit_errors / total_symbols if total_symbols > 0 else 0.0
            fer = frame_errors / blocks if blocks > 0 else 0.0
            throughput_msps = (total_symbols / elapsed_sec) / 1e6 if elapsed_sec > 0.0 else 0.0

            summary_row = f"{ebn0:<10.2f} | {blocks:<12d} | {bit_errors:<12d} | {frame_errors:<14d} | {ber:<12.4e} | {fer:<12.4e} | {throughput_msps:<18.2f}"
            if is_tty:
                print(f"\r{summary_row:<104}")
            else:
                print(f"-> Final: {summary_row}")

            # Note: Throughput is printed to console only, not stored in the CSV file
            with open(output_filename, 'a') as out_file:
                out_file.write(f"{ebn0:.2f},{blocks},{bit_errors},{frame_errors},{ber:.6e},{fer:.6e}\n")
                out_file.flush()

        except Exception as e:
            print(f"\nError simulating Eb/N0 = {ebn0:.2f} dB: {e}", file=sys.stderr)
            break

    print(f"\nSimulation complete. Output saved to: {output_filename}")

if __name__ == "__main__":
    main()