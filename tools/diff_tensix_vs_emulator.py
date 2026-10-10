#!/usr/bin/env python3
"""
Tenstorrent QuietBox 2: Differential Co-Verification Analyzer
Compares binary dumps between Golden CPU Emulator and Tensix Hardware L1 SRAM.
"""

import sys
import struct
import numpy as np

def bf16_to_fp32(val_u16):
    """Convert 16-bit bfloat16 representation to 32-bit float."""
    u32 = int(val_u16) << 16
    return struct.unpack('>f', struct.pack('>I', u32))[0]

def analyze_vnode_diff(golden_path, tensix_path, num_vnodes=2560):
    print("=" * 80)
    print(" TENSIX TTSIM vs GOLDEN EMULATOR: VARIABLE NODE (CB_VNODE_A) DIFFERENTIAL")
    print(f" Golden File: {golden_path}")
    print(f" Tensix File: {tensix_path}")
    print("=" * 80)

    with open(golden_path, "rb") as f:
        golden_bytes = f.read()
    with open(tensix_path, "rb") as f:
        tensix_bytes = f.read()

    expected_len = num_vnodes * 4 * 2 # 20,480 bytes
    if len(golden_bytes) < expected_len or len(tensix_bytes) < expected_len:
        print(f"[ERROR] Incomplete file: Golden={len(golden_bytes)}B, Tensix={len(tensix_bytes)}B (expected {expected_len}B)")
        return False

    golden_words = np.frombuffer(golden_bytes[:expected_len], dtype=np.uint16)
    tensix_words = np.frombuffer(tensix_bytes[:expected_len], dtype=np.uint16)

    mismatches = np.where(golden_words != tensix_words)[0]
    total_words = len(golden_words)

    if len(mismatches) == 0:
        print(f"\n>>> [100% BIT-EXACT MATCH] All {total_words} bfloat16 words match perfectly between CPU and Silicon!")
        return True

    print(f"\n>>> [MISMATCH DETECTED] {len(mismatches)} / {total_words} words differ ({len(mismatches)/total_words*100:.2f}% divergence)!\n")
    print(f"{'Offset':<8} | {'Var (v)':<8} | {'CW':<4} | {'Golden (Hex)':<12} | {'Golden (Val)':<14} | {'Tensix (Hex)':<12} | {'Tensix (Val)':<14} | {'Diff'}")
    print("-" * 96)

    shown = 0
    max_diff = 0.0
    for idx in mismatches:
        byte_off = idx * 2
        t = byte_off // 2048
        f = (byte_off % 2048) // 512
        r = (byte_off % 512) // 32
        grp = (byte_off % 32) // 8
        cw = (byte_off % 8) // 2
        
        row_global = r + (16 if f >= 2 else 0)
        k_global = grp + (4 if (f % 2 == 1) else 0)
        v = t * 256 + row_global * 8 + k_global

        g_u16 = golden_words[idx]
        t_u16 = tensix_words[idx]
        g_flt = bf16_to_fp32(g_u16)
        t_flt = bf16_to_fp32(t_u16)
        diff = abs(g_flt - t_flt)
        if diff > max_diff:
            max_diff = diff

        if shown < 25:
            print(f"0x{idx*2:04X}   | {v:<8} | {cw:<4} | 0x{g_u16:04X}       | {g_flt:<+14.5f} | 0x{t_u16:04X}       | {t_flt:<+14.5f} | {diff:+.5f}")
            shown += 1

    print("-" * 96)
    print(f"Summary: Total Mismatches = {len(mismatches)} | Max Absolute Difference = {max_diff:.5f}")
    return False

def analyze_tile_diff(golden_path, tensix_path, tile_name="CB_SCRATCH_GATHER"):
    print("=" * 80)
    print(f" TENSIX SILICON vs GOLDEN EMULATOR: 32x32 TILE ({tile_name}) DIFFERENTIAL")
    print(f" Golden File: {golden_path}")
    print(f" Tensix File: {tensix_path}")
    print("=" * 80)

    with open(golden_path, "rb") as f:
        golden_bytes = f.read()
    with open(tensix_path, "rb") as f:
        tensix_bytes = f.read()

    expected_len = 32 * 32 * 2 # 2048 bytes
    if len(golden_bytes) < expected_len or len(tensix_bytes) < expected_len:
        print(f"[ERROR] Incomplete tile: Golden={len(golden_bytes)}B, Tensix={len(tensix_bytes)}B (expected {expected_len}B)")
        return False

    golden_words = np.frombuffer(golden_bytes[:expected_len], dtype=np.uint16)
    tensix_words = np.frombuffer(tensix_bytes[:expected_len], dtype=np.uint16)

    mismatches = np.where(golden_words != tensix_words)[0]
    total_words = len(golden_words)

    if len(mismatches) == 0:
        print(f"\n>>> [100% BIT-EXACT MATCH] All {total_words} tile elements match perfectly!")
        return True

    print(f"\n>>> [MISMATCH DETECTED] {len(mismatches)} / {total_words} tile words differ!\n")
    print(f"{'Row':<6} | {'Col':<6} | {'Golden (Hex)':<12} | {'Golden (Val)':<14} | {'Tensix (Hex)':<12} | {'Tensix (Val)':<14} | {'Diff'}")
    print("-" * 84)

    shown = 0
    for idx in mismatches:
        row = idx // 32
        col = idx % 32
        g_u16 = golden_words[idx]
        t_u16 = tensix_words[idx]
        g_flt = bf16_to_fp32(g_u16)
        t_flt = bf16_to_fp32(t_u16)
        diff = abs(g_flt - t_flt)

        if shown < 32:
            print(f"{row:<6} | {col:<6} | 0x{g_u16:04X}       | {g_flt:<+14.5f} | 0x{t_u16:04X}       | {t_flt:<+14.5f} | {diff:+.5f}")
            shown += 1

    print("-" * 84)
    return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 diff_tensix_vs_emulator.py <golden.bin> <tensix.bin> [--tile <name>]")
        sys.exit(1)

    golden_f = sys.argv[1]
    tensix_f = sys.argv[2]
    is_tile = "--tile" in sys.argv

    if is_tile:
        tile_name = sys.argv[sys.argv.index("--tile") + 1] if len(sys.argv) > sys.argv.index("--tile") + 1 else "TILE"
        analyze_tile_diff(golden_f, tensix_f, tile_name)
    else:
        analyze_vnode_diff(golden_f, tensix_f)
