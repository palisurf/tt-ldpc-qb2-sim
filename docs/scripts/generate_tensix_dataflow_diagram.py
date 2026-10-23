#!/usr/bin/env python3
"""
generate_tensix_dataflow_diagram.py

Generates a publication-quality data flow block diagram of the Tenstorrent QuietBox 2
Circulant DMA 4-Codeword LDPC Decoder Engine for a single Tensix Core.

Architectural Structure:
  VERTICAL LAYERS (Top to Bottom Stack):
    Layer 1 (Top):    L1 SRAM Persistent Storage (CB_VNODE, CB_R_MSG, CB_TELEMETRY)
    Layer 2:          Gather / Scatter & Data Movement Plane (BRISC, NCRISC, 512-Bit NOC DMA Engine)
    Layer 3:          Scratch Pads & Staging Circular Buffers (CB_SCRATCH_GATHER, CB_SCRATCH_SCATTER, Lockstep Monitor)
    Layer 4:          Unpack / Pack Orchestration (Worker 2: TRISC0 Unpack, Worker 4: TRISC2 Pack)
    Layer 5 (Bottom): Compute Engine & Tensix DEST Registers (Worker 3: TRISC1 Math, 64 FP16 DEST Regs)

Data Flow Directions:
  LEFT SIDE:  DOWNWARD flow (Ingest & Staging)
              - Vnode Data (LLR beliefs) flows down from CB_VNODE -> Gather -> Scratch Gather -> Unpack -> Compute
              - Edge Data (R_old checks) flows down from CB_R_MSG -> Unpack -> Compute
  RIGHT SIDE: UPWARD flow (Pack, Commit & Writeback)
              - Vnode Data (L_new beliefs) flows up from Compute -> Pack -> Scratch Scatter -> Scatter DMA -> CB_VNODE
              - Edge Data (R_new checks) flows up from Compute -> Pack -> CB_R_MSG FIFO
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def create_diagram():
    # High-resolution widescreen 16:9 ratio
    fig = plt.figure(figsize=(19, 11.4), dpi=300, facecolor='#0b1120')
    ax = fig.add_axes([0, 0, 1, 1], facecolor='#0b1120')
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 116)
    ax.axis('off')

    # Color Palette (Dark Mode / Engineering Blueprint)
    bg_box = '#131e36'
    
    # Theme colors
    col_l1 = '#0284c7'      # L1 Storage: Blue / Cyan
    col_courier = '#3b82f6' # Couriers: Blue
    col_scratch = '#10b981' # Scratch Pads: Emerald Green
    col_trisc = '#8b5cf6'   # Unpack / Pack TRISCs: Purple
    col_compute = '#ec4899' # SFPU Compute: Rose / Pink
    
    # Bus Colors
    col_vnode_down = '#38bdf8' # Vnode Down: Bright Cyan
    col_edge_down = '#818cf8'  # Edge Down: Indigo
    col_vnode_up = '#34d399'   # Vnode Up: Bright Green
    col_edge_up = '#f472b6'    # Edge Up: Pink / Magenta

    # -------------------------------------------------------------
    # TOP HEADER BANNER (y: 105.8 to 114.8, h: 9.0)
    # -------------------------------------------------------------
    header_box = patches.FancyBboxPatch(
        (5, 105.8), 190, 9.0, boxstyle="round,pad=0.3,rounding_size=0.8",
        facecolor='#0f172a', edgecolor='#38bdf8', linewidth=2.0
    )
    ax.add_patch(header_box)
    
    ax.text(100, 112.4, "TENSIX CORE LDPC DATAFLOW & MEMORY ARCHITECTURE",
            ha='center', va='center', fontsize=16.0, fontweight='bold', color='#ffffff',
            family='sans-serif')
    ax.text(100, 109.5, "QuietBox 2 Circulant DMA 4-Codeword Engine • × 440 Copies Across Full Mesh (4 Chips × 110 Cores = 1,760 Concurrent Codewords)",
            ha='center', va='center', fontsize=9.2, fontweight='bold', color='#fde047',
            family='sans-serif')
    ax.text(100, 107.2, "Vertical Layered Stack: L1 Persistent SRAM → Gather/Scatter Couriers → Staging Scratchpads → Unpack/Pack → Compute Engine",
            ha='center', va='center', fontsize=8.2, fontweight='bold', color='#93c5fd',
            family='sans-serif')

    # Stack X-Bounds
    sx0 = 42
    sx1 = 158
    sw = sx1 - sx0  # 116

    # =============================================================
    # LAYER 1 (TOP): L1 SRAM PERSISTENT STORAGE (y: 89.0 to 103.5, h: 14.5)
    # =============================================================
    l1_box = patches.FancyBboxPatch(
        (sx0, 89.0), sw, 14.5, boxstyle="round,pad=0.25,rounding_size=0.8",
        facecolor=bg_box, edgecolor=col_l1, linewidth=2.0
    )
    ax.add_patch(l1_box)
    ax.text(sx0 + 3, 101.5, "LAYER 1: L1 SRAM PERSISTENT STORAGE",
            ha='left', va='center', fontsize=10.5, fontweight='bold', color='#38bdf8')
    ax.text(sx1 - 3, 101.5, "Replicated × 440 Tensix Cores Across 2×2 Blackhole Mesh • 1,760 Codewords Parallel",
            ha='right', va='center', fontsize=7.6, color='#cbd5e1', fontweight='bold')

    # 1.1 CB_VNODE (x: 44 to 79, w: 35, y: 90.5 to 100.0)
    vnode_box = patches.FancyBboxPatch(
        (44, 90.5), 35, 9.5, boxstyle="round,pad=0.15,rounding_size=0.4",
        facecolor='#1e293b', edgecolor='#0ea5e9', linewidth=1.5
    )
    ax.add_patch(vnode_box)
    ax.text(61.5, 98.2, "CB_VNODE (CB 0) • Vnode Data", ha='center', va='center',
            fontsize=8.8, fontweight='bold', color='#38bdf8')
    ax.text(61.5, 95.8, "Variable Node Posterior LLR Beliefs", ha='center', va='center',
            fontsize=7.2, color='#e2e8f0', fontweight='bold')
    ax.text(61.5, 93.6, "• Size: 20 KB (2,560 Nodes × 4 Codewords)", ha='center', va='center',
            fontsize=6.8, color='#cbd5e1')
    ax.text(61.5, 91.6, "• In-Place Evolving: Lv = Lv_ch + Σ R", ha='center', va='center',
            fontsize=6.8, color='#fde047', fontweight='bold')

    # 1.2 CB_R_MSG (x: 82 to 120, w: 38, y: 90.5 to 100.0)
    rmsg_box = patches.FancyBboxPatch(
        (82, 90.5), 38, 9.5, boxstyle="round,pad=0.15,rounding_size=0.4",
        facecolor='#1e293b', edgecolor='#6366f1', linewidth=1.5
    )
    ax.add_patch(rmsg_box)
    ax.text(101.0, 98.2, "CB_R_MSG (CB 1) • Edge Data", ha='center', va='center',
            fontsize=8.8, fontweight='bold', color='#818cf8')
    ax.text(101.0, 95.8, "40-Tile Check Message Circular FIFO", ha='center', va='center',
            fontsize=7.2, color='#e2e8f0', fontweight='bold')
    ax.text(101.0, 93.6, "• Size: 80 KB (7,680 Edges × 4 CWs, bfloat16)", ha='center', va='center',
            fontsize=6.8, color='#cbd5e1')
    ax.text(101.0, 91.6, "• R_old Unpack Source  |  R_new Commit Dest", ha='center', va='center',
            fontsize=6.8, color='#c7d2fe', fontweight='bold')

    # 1.3 CB_TELEMETRY (x: 123 to 156, w: 33, y: 90.5 to 100.0)
    telem_box = patches.FancyBboxPatch(
        (123, 90.5), 33, 9.5, boxstyle="round,pad=0.15,rounding_size=0.4",
        facecolor='#1e293b', edgecolor='#a855f7', linewidth=1.5
    )
    ax.add_patch(telem_box)
    ax.text(139.5, 98.2, "CB_TELEMETRY (CB 24)", ha='center', va='center',
            fontsize=8.8, fontweight='bold', color='#c084fc')
    ax.text(139.5, 95.8, "Mailboxes & Convergence Latch", ha='center', va='center',
            fontsize=7.2, color='#e2e8f0', fontweight='bold')
    ax.text(139.5, 93.6, "• active_mask[3:0] & cw_iters[0..3]", ha='center', va='center',
            fontsize=6.8, color='#fde047', fontweight='bold')
    ax.text(139.5, 91.6, "• Zero post-convergence error count", ha='center', va='center',
            fontsize=6.8, color='#34d399', fontweight='bold')

    # =============================================================
    # LAYER 2: GATHER / SCATTER & DATA MOVEMENT (y: 68.0 to 82.8, h: 14.8)
    # Note: Gap between Layer 1 (bottom 89.0) and Layer 2 (top 82.8) is 6.2 units (y: 82.8 to 89.0)
    # =============================================================
    l2_box = patches.FancyBboxPatch(
        (sx0, 68.0), sw, 14.8, boxstyle="round,pad=0.25,rounding_size=0.8",
        facecolor=bg_box, edgecolor=col_courier, linewidth=2.0
    )
    ax.add_patch(l2_box)
    ax.text(sx0 + 3, 81.0, "LAYER 2: GATHER / SCATTER COURIERS",
            ha='left', va='center', fontsize=10.0, fontweight='bold', color='#60a5fa')
    ax.text(sx1 - 3, 81.0, "Workers 0 & 1 • 512-Bit NOC DMA Bursts (2c / 128-Byte Slice)",
            ha='right', va='center', fontsize=7.6, color='#94a3b8')

    # 2.1 Worker 0 (BRISC) - (x: 44 to 77, w: 33, y: 69.5 to 79.2)
    brisc_box = patches.FancyBboxPatch(
        (44, 69.5), 33, 9.7, boxstyle="round,pad=0.15,rounding_size=0.4",
        facecolor='#1e293b', edgecolor='#3b82f6', linewidth=1.5
    )
    ax.add_patch(brisc_box)
    ax.text(60.5, 77.6, "WORKER 0: BRISC (Faces 0 & 2)", ha='center', va='center',
            fontsize=8.8, fontweight='bold', color='#93c5fd')
    ax.text(60.5, 75.3, "• Checks 0..3 (16 Rows), Lanes 0..15", ha='center', va='center',
            fontsize=6.8, color='#cbd5e1')
    ax.text(60.5, 73.3, "• Stage 1 Gather: 384 B to Face 0/2", ha='center', va='center',
            fontsize=6.8, color='#38bdf8', fontweight='bold')
    ax.text(60.5, 71.3, "• Stage 5 Scatter: Blind DMA Commit", ha='center', va='center',
            fontsize=6.8, color='#f59e0b', fontweight='bold')

    # 2.2 512-Bit DMA Engine Badge (x: 80 to 120, w: 40, y: 69.5 to 79.2)
    dma_box = patches.FancyBboxPatch(
        (80, 69.5), 40, 9.7, boxstyle="round,pad=0.15,rounding_size=0.4",
        facecolor='#0c4a6e', edgecolor='#38bdf8', linewidth=1.5
    )
    ax.add_patch(dma_box)
    ax.text(100, 77.6, "512-BIT L1 NOC DMA ENGINE", ha='center', va='center',
            fontsize=9.0, fontweight='bold', color='#ffffff')
    ax.text(100, 75.3, "Circulant Contiguity: Δv/Δc = +1", ha='center', va='center',
            fontsize=7.2, color='#67e8f9', fontweight='bold')
    ax.text(100, 73.3, "128-Byte Slice ÷ 64 B/Cyc = 2 Cycles!", ha='center', va='center',
            fontsize=7.2, color='#fde047', fontweight='bold')
    ax.text(100, 71.3, "Collapses Gather & Scatter to 12c", ha='center', va='center',
            fontsize=6.8, color='#cbd5e1')

    # 2.3 Worker 1 (NCRISC) - (x: 123 to 156, w: 33, y: 69.5 to 79.2)
    ncrisc_box = patches.FancyBboxPatch(
        (123, 69.5), 33, 9.7, boxstyle="round,pad=0.15,rounding_size=0.4",
        facecolor='#1e293b', edgecolor='#3b82f6', linewidth=1.5
    )
    ax.add_patch(ncrisc_box)
    ax.text(139.5, 77.6, "WORKER 1: NCRISC (Faces 1 & 3)", ha='center', va='center',
            fontsize=8.8, fontweight='bold', color='#93c5fd')
    ax.text(139.5, 75.3, "• Checks 4..7 (16 Rows), Lanes 16..31", ha='center', va='center',
            fontsize=6.8, color='#cbd5e1')
    ax.text(139.5, 73.3, "• Stage 1 Gather: 384 B to Face 1/3", ha='center', va='center',
            fontsize=6.8, color='#38bdf8', fontweight='bold')
    ax.text(139.5, 71.3, "• Stage 5 Scatter: Blind DMA Commit", ha='center', va='center',
            fontsize=6.8, color='#f59e0b', fontweight='bold')

    # =============================================================
    # LAYER 3: SCRATCH PADS & STAGING CIRCULAR BUFFERS (y: 50.0 to 65.5, h: 15.5)
    # =============================================================
    l3_box = patches.FancyBboxPatch(
        (sx0, 50.0), sw, 15.5, boxstyle="round,pad=0.25,rounding_size=0.8",
        facecolor=bg_box, edgecolor=col_scratch, linewidth=2.0
    )
    ax.add_patch(l3_box)
    ax.text(sx0 + 3, 63.5, "LAYER 3: SCRATCH PADS & STAGING CIRCULAR BUFFERS",
            ha='left', va='center', fontsize=10.5, fontweight='bold', color='#34d399')
    ax.text(sx1 - 3, 63.5, "Double-Buffered Tile Pages (2 KB/Page) • Read-After-Write Hazard Elimination",
            ha='right', va='center', fontsize=7.6, color='#94a3b8')

    # 3.1 CB_SCRATCH_GATHER (x: 44 to 78, w: 34, y: 51.5 to 61.5)
    sg_box = patches.FancyBboxPatch(
        (44, 51.5), 34, 10.0, boxstyle="round,pad=0.15,rounding_size=0.4",
        facecolor='#1e293b', edgecolor='#10b981', linewidth=1.5
    )
    ax.add_patch(sg_box)
    ax.text(61, 59.8, "CB_SCRATCH_GATHER (CB 2)", ha='center', va='center',
            fontsize=8.8, fontweight='bold', color='#34d399')
    ax.text(61, 57.5, "Stage 1 Gather Staging Buffer (4 KB)", ha='center', va='center',
            fontsize=7.2, color='#e2e8f0', fontweight='bold')
    ax.text(61, 55.3, "• Double-Buffered: 2 KB / Page", ha='center', va='center',
            fontsize=6.8, color='#cbd5e1')
    ax.text(61, 53.3, "• Faces 0..3 in Vertical SIMD Orientation", ha='center', va='center',
            fontsize=6.8, color='#6ee7b7', fontweight='bold')

    # 3.2 Lockstep Invariant Table (x: 81 to 120, w: 39, y: 51.5 to 61.5)
    ls_box = patches.FancyBboxPatch(
        (81, 51.5), 39, 10.0, boxstyle="round,pad=0.15,rounding_size=0.4",
        facecolor='#0f172a', edgecolor='#10b981', linewidth=1.5
    )
    ax.add_patch(ls_box)
    ax.text(100.5, 59.8, "MANDATORY LOCKSTEP INVARIANT", ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='#34d399')
    ax.text(100.5, 57.5, "Gather(12c) → Unpack(16c) → Compute(912c) →", ha='center', va='center',
            fontsize=6.8, color='#cbd5e1')
    ax.text(100.5, 55.5, "Pack(16c) → Scatter(12c) = 968c Turnaround", ha='center', va='center',
            fontsize=6.8, color='#cbd5e1')
    ax.text(100.5, 53.3, "94.2% Compute Duty Cycle (912c ÷ 968c)!", ha='center', va='center',
            fontsize=7.2, color='#fde047', fontweight='bold')

    # 3.3 CB_SCRATCH_SCATTER (x: 123 to 156, w: 33, y: 51.5 to 61.5)
    ss_box = patches.FancyBboxPatch(
        (123, 51.5), 33, 10.0, boxstyle="round,pad=0.15,rounding_size=0.4",
        facecolor='#1e293b', edgecolor='#f59e0b', linewidth=1.5
    )
    ax.add_patch(ss_box)
    ax.text(139.5, 59.8, "CB_SCRATCH_SCATTER (CB 3)", ha='center', va='center',
            fontsize=8.8, fontweight='bold', color='#fbbf24')
    ax.text(139.5, 57.5, "Stage 5 Scatter Staging Buffer (4 KB)", ha='center', va='center',
            fontsize=7.2, color='#e2e8f0', fontweight='bold')
    ax.text(139.5, 55.3, "• Receives Fresh L_new from TRISC2", ha='center', va='center',
            fontsize=6.8, color='#cbd5e1')
    ax.text(139.5, 53.3, "• Read by Workers 0 & 1 for DMA Commit", ha='center', va='center',
            fontsize=6.8, color='#fcd34d', fontweight='bold')

    # =============================================================
    # LAYER 4: UNPACK / PACK ORCHESTRATION (y: 33.5 to 47.5, h: 14)
    # =============================================================
    l4_box = patches.FancyBboxPatch(
        (sx0, 33.5), sw, 14, boxstyle="round,pad=0.25,rounding_size=0.8",
        facecolor=bg_box, edgecolor=col_trisc, linewidth=2.0
    )
    ax.add_patch(l4_box)
    ax.text(sx0 + 3, 45.5, "LAYER 4: UNPACK / PACK ORCHESTRATION (WORKERS 2 & 4)",
            ha='left', va='center', fontsize=10.5, fontweight='bold', color='#c084fc')
    ax.text(sx1 - 3, 45.5, "Zero-Transpose Memory Mapping • Direct DEST Register Interconnect",
            ha='right', va='center', fontsize=7.6, color='#94a3b8')

    # 4.1 Worker 2: TRISC0 (UNPACK) - (x: 44 to 98, w: 54, y: 35 to 43.5)
    t0_box = patches.FancyBboxPatch(
        (44, 35), 54, 8.5, boxstyle="round,pad=0.15,rounding_size=0.4",
        facecolor='#2e1065', edgecolor='#a855f7', linewidth=1.5
    )
    ax.add_patch(t0_box)
    ax.text(71, 41.5, "WORKER 2: TRISC0 (UNPACK ENGINE)", ha='center', va='center',
            fontsize=8.8, fontweight='bold', color='#f3e8ff')
    ax.text(71, 39.2, "• Unpacks Vnode Beliefs (CB_SCRATCH_GATHER) → Tile 1 (dst[32..63])", ha='center', va='center',
            fontsize=6.8, color='#38bdf8', fontweight='bold')
    ax.text(71, 37.0, "• Unpacks Edge Messages (CB_R_MSG FIFO) → Tile 0 (dst[0..31])", ha='center', va='center',
            fontsize=6.8, color='#818cf8', fontweight='bold')

    # 4.2 Worker 4: TRISC2 (PACK) - (x: 102 to 156, w: 54, y: 35 to 43.5)
    t2_box = patches.FancyBboxPatch(
        (102, 35), 54, 8.5, boxstyle="round,pad=0.15,rounding_size=0.4",
        facecolor='#2e1065', edgecolor='#a855f7', linewidth=1.5
    )
    ax.add_patch(t2_box)
    ax.text(129, 41.5, "WORKER 4: TRISC2 (PACK ENGINE)", ha='center', va='center',
            fontsize=8.8, fontweight='bold', color='#f3e8ff')
    ax.text(129, 39.2, "• Packs Fresh Beliefs L_new (Tile 1) → CB_SCRATCH_SCATTER", ha='center', va='center',
            fontsize=6.8, color='#34d399', fontweight='bold')
    ax.text(129, 37.0, "• Packs Fresh Checks R_new (Tile 0) → CB_R_MSG FIFO Commit", ha='center', va='center',
            fontsize=6.8, color='#f472b6', fontweight='bold')

    # =============================================================
    # LAYER 5 (BOTTOM): COMPUTE ENGINE & DEST REGISTERS (y: 5 to 30.5, h: 25.5)
    # =============================================================
    l5_box = patches.FancyBboxPatch(
        (sx0, 5), sw, 25.5, boxstyle="round,pad=0.25,rounding_size=0.8",
        facecolor=bg_box, edgecolor=col_compute, linewidth=2.0
    )
    ax.add_patch(l5_box)
    ax.text(sx0 + 3, 28.5, "LAYER 5: COMPUTE ENGINE & TENSIX DEST REGISTERS (WORKER 3: TRISC1)",
            ha='left', va='center', fontsize=10.5, fontweight='bold', color='#f472b6')
    ax.text(sx1 - 3, 28.5, "32-Lane SFPU Vector Engine • Vectorized True Recursive Approximate-Min*",
            ha='right', va='center', fontsize=7.6, color='#94a3b8')

    # 5.1 TENSIX DEST REGISTERS (x: 44 to 98, w: 54, y: 7 to 26.5)
    dest_box = patches.FancyBboxPatch(
        (44, 7), 54, 19.5, boxstyle="round,pad=0.15,rounding_size=0.4",
        facecolor='#14532d', edgecolor='#4ade80', linewidth=1.5
    )
    ax.add_patch(dest_box)
    ax.text(71, 24.5, "TENSIX DEST REGISTERS (64 × 16-BIT FP)", ha='center', va='center',
            fontsize=9.0, fontweight='bold', color='#86efac')
    ax.text(71, 22.0, "Tile 0: dst_reg[0..31] → Outgoing Checks R_new (Edge Data)", ha='center', va='center',
            fontsize=7.4, color='#bbf7d0', fontweight='bold')
    ax.text(71, 19.8, "  F0 [0..7] | F1 [8..15] | F2 [16..23] | F3 [24..31]", ha='center', va='center',
            fontsize=6.8, color='#cbd5e1')
    ax.text(71, 17.0, "Tile 1: dst_reg[32..63] → Updated Beliefs L_new (Vnode Data)", ha='center', va='center',
            fontsize=7.4, color='#fef08a', fontweight='bold')
    ax.text(71, 14.8, "  F0 [32..39] | F1 [40..47] | F2 [48..55] | F3 [56..63]", ha='center', va='center',
            fontsize=6.8, color='#cbd5e1')
    ax.text(71, 12.0, "• Round-to-Nearest-Even (RNE) Mantissa Protection", ha='center', va='center',
            fontsize=7.0, color='#86efac', fontweight='bold')
    ax.text(71, 9.4, "• Damped Checks: R_new = R_mag · 0.9375 · sgn", ha='center', va='center',
            fontsize=7.0, color='#34d399', fontweight='bold')

    # 5.2 SFPU ARITHMETIC PIPELINE (x: 102 to 156, w: 54, y: 7 to 26.5)
    math_box = patches.FancyBboxPatch(
        (102, 7), 54, 19.5, boxstyle="round,pad=0.15,rounding_size=0.4",
        facecolor='#1e1b4b', edgecolor='#ec4899', linewidth=1.5
    )
    ax.add_patch(math_box)
    ax.text(129, 24.5, "SFPU PARALLEL TREE APPROX-MIN* (32 LANES)", ha='center', va='center',
            fontsize=8.8, fontweight='bold', color='#f472b6')
    ax.text(129, 22.2, "1. Prior Subtraction: q_d = Lv - R_old (in-place Tile 0)", ha='center', va='center',
            fontsize=7.1, color='#cbd5e1')
    ax.text(129, 19.8, "2. Subvector Transpose: swap_subvec16 (pairwise align)", ha='center', va='center',
            fontsize=7.1, color='#cbd5e1')
    ax.text(129, 17.4, "3. Parallel Tree: Pairwise M01, M23, M45 (FP32 LRegs)", ha='center', va='center',
            fontsize=7.1, color='#cbd5e1')
    ax.text(129, 15.0, "4. Full Extrinsics: R_mag(d) strictly excludes edge d", ha='center', va='center',
            fontsize=7.2, color='#fde047', fontweight='bold')
    ax.text(129, 12.6, "5. Belief Damping (β = 0.9375): Trapping set suppression", ha='center', va='center',
            fontsize=7.2, color='#34d399', fontweight='bold')
    ax.text(129, 10.2, "6. Posterior Update: L_new = q_d + R_new (Tile 1 RNE)", ha='center', va='center',
            fontsize=7.2, color='#a5f3fc', fontweight='bold')
    ax.text(129, 8.1, "Full Extrinsics • Zero Sorting • Intrinsic Damping (β = 15/16)", ha='center', va='center',
            fontsize=6.6, color='#fbcfe8')

    # =============================================================
    # LEFT SIDE: DOWNWARD DATA FLOW (VNODE DATA & EDGE DATA)
    # =============================================================
    # --- LANE L1: VNODE DATA FLOW DOWN (x = 30) ---
    # Originates from CB_VNODE (x=44, y=94.5)
    vnode_down_pts = [(44, 94.5), (30, 94.5), (30, 16.5), (44, 16.5)]
    ax.plot([p[0] for p in vnode_down_pts], [p[1] for p in vnode_down_pts],
            color=col_vnode_down, linewidth=3.0, linestyle='-')
    
    # Downward Arrowheads along Lane L1
    ax.annotate('', xy=(30, 85), xytext=(30, 88),
                arrowprops=dict(arrowstyle="->", color=col_vnode_down, lw=2.5))
    ax.annotate('', xy=(30, 65), xytext=(30, 68),
                arrowprops=dict(arrowstyle="->", color=col_vnode_down, lw=2.5))
    ax.annotate('', xy=(30, 46), xytext=(30, 49),
                arrowprops=dict(arrowstyle="->", color=col_vnode_down, lw=2.5))
    ax.annotate('', xy=(44, 16.5), xytext=(40, 16.5),
                arrowprops=dict(arrowstyle="->", color=col_vnode_down, lw=2.5))

    # Taps into Layer 2 (Gather Couriers), Layer 3 (Scratch Gather), Layer 4 (TRISC0 Unpack)
    ax.annotate('', xy=(44, 74.5), xytext=(30, 74.5),
                arrowprops=dict(arrowstyle="->", color=col_vnode_down, lw=2.0))
    ax.annotate('', xy=(44, 56.5), xytext=(30, 56.5),
                arrowprops=dict(arrowstyle="->", color=col_vnode_down, lw=2.0))
    ax.annotate('', xy=(44, 40.5), xytext=(30, 40.5),
                arrowprops=dict(arrowstyle="->", color=col_vnode_down, lw=2.0))

    # Lane L1 Labels (placed cleanly to avoid line crossings)
    ax.text(30, 78.5, "512b DMA\nGather", ha='right', va='center', fontsize=6.8, color=col_vnode_down, fontweight='bold')
    ax.text(30, 60.5, "Stage 1\nGather", ha='right', va='center', fontsize=6.8, color=col_vnode_down, fontweight='bold')
    ax.text(30, 43.5, "Stage 2\nUnpack", ha='right', va='center', fontsize=6.8, color=col_vnode_down, fontweight='bold')
    ax.text(37, 18.5, "Vnode In (Tile 1)", ha='center', va='bottom', fontsize=6.8, color=col_vnode_down, fontweight='bold')

    # Badge for Vnode Downward Bus
    ax.text(30, 102.5, "VNODE DATA (DOWN)\nLv Beliefs Highway", ha='center', va='center',
            fontsize=7.2, fontweight='bold', color='#ffffff',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#0284c7', edgecolor='#38bdf8', alpha=0.95))

    # --- LANE L2: EDGE DATA FLOW DOWN (x = 13) ---
    # Originates from top of CB_R_MSG at (95, 100.0) -> UP to (95, 104.8) -> LEFT to (13, 104.8) -> DOWN along x=13
    edge_down_pts = [(95, 100.0), (95, 104.8), (13, 104.8), (13, 21.5), (44, 21.5)]
    ax.plot([p[0] for p in edge_down_pts], [p[1] for p in edge_down_pts],
            color=col_edge_down, linewidth=2.2, linestyle='--')
    
    # Downward Arrowheads along Lane L2
    ax.annotate('', xy=(13, 75), xytext=(13, 80),
                arrowprops=dict(arrowstyle="->", color=col_edge_down, lw=2.2))
    ax.annotate('', xy=(13, 55), xytext=(13, 60),
                arrowprops=dict(arrowstyle="->", color=col_edge_down, lw=2.2))
    ax.annotate('', xy=(13, 37), xytext=(13, 42),
                arrowprops=dict(arrowstyle="->", color=col_edge_down, lw=2.2))
    ax.annotate('', xy=(44, 21.5), xytext=(38, 21.5),
                arrowprops=dict(arrowstyle="->", color=col_edge_down, lw=2.2))
    
    # Tap to TRISC0 Unpack at (13, 37.0) -> (44, 37.0)
    ax.annotate('', xy=(44, 37.0), xytext=(13, 37.0),
                arrowprops=dict(arrowstyle="->", color=col_edge_down, lw=2.0))

    # Lane L2 Labels
    ax.text(13, 50, "Bypasses L2/L3 Staging\nDirect FIFO Unpack", ha='center', va='center',
            fontsize=6.5, color='#a5b4fc', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='#0f172a', edgecolor=col_edge_down, alpha=0.9))
    ax.text(37, 23.5, "Edge In (Tile 0)", ha='center', va='bottom', fontsize=6.8, color=col_edge_down, fontweight='bold')

    # Badge for Edge Downward Bus
    ax.text(13, 102.5, "EDGE DATA (DOWN)\nR_old Check FIFO", ha='center', va='center',
            fontsize=7.2, fontweight='bold', color='#ffffff',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#3730a3', edgecolor='#818cf8', alpha=0.95))

    # =============================================================
    # RIGHT SIDE: UPWARD DATA FLOW (VNODE COMMIT & EDGE COMMIT)
    # =============================================================
    # --- LANE R1: VNODE DATA COMMIT UP (x = 170) ---
    # Originates from TRISC2 Pack & CB_SCRATCH_SCATTER (x=156, y=56.5)
    # Path: (156, 56.5) -> (170, 56.5) -> UP to (170, 87.2) -> LEFT along open L1-L2 corridor to (61.5, 87.2) -> UP to (61.5, 90.5)
    vnode_up_pts = [(156, 56.5), (170, 56.5), (170, 87.2), (61.5, 87.2), (61.5, 90.5)]
    ax.plot([p[0] for p in vnode_up_pts], [p[1] for p in vnode_up_pts],
            color=col_vnode_up, linewidth=3.0, linestyle='-')
    
    # Upward Arrowheads along Lane R1
    ax.annotate('', xy=(170, 68), xytext=(170, 64),
                arrowprops=dict(arrowstyle="->", color=col_vnode_up, lw=2.5))
    ax.annotate('', xy=(170, 80), xytext=(170, 76),
                arrowprops=dict(arrowstyle="->", color=col_vnode_up, lw=2.5))
    ax.annotate('', xy=(61.5, 90.5), xytext=(61.5, 88.2),
                arrowprops=dict(arrowstyle="->", color=col_vnode_up, lw=2.5))

    # Tap from Scatter Courier in Layer 2: Worker 1 drives 512b DMA writeback
    ax.annotate('', xy=(170, 74.5), xytext=(156, 74.5),
                arrowprops=dict(arrowstyle="->", color=col_vnode_up, lw=2.0))
    # Tap from TRISC2 Pack in Layer 4: freshly packed L_new up into CB_SCRATCH_SCATTER
    ax.annotate('', xy=(156, 53.5), xytext=(156, 42.0),
                arrowprops=dict(arrowstyle="->", color=col_vnode_up, lw=2.2))
    ax.text(158, 47.5, "Pack L_new", ha='left', va='center', fontsize=6.8, color=col_vnode_up, fontweight='bold')

    # Lane R1 Labels
    ax.text(170, 61.5, "Stage 5\nScatter", ha='left', va='center', fontsize=6.8, color=col_vnode_up, fontweight='bold')
    ax.text(170, 78.5, "512b DMA\nCommit", ha='left', va='center', fontsize=6.8, color=col_vnode_up, fontweight='bold')
    ax.text(125, 87.2, "In-Place Posterior Writeback: Lv ← L_new (384 B / face)", ha='center', va='center',
            fontsize=6.8, color=col_vnode_up, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='#0b1120', edgecolor=col_vnode_up, alpha=0.9))

    # Badge for Vnode Upward Bus
    ax.text(170, 102.5, "VNODE COMMIT (UP)\nL_new Writeback", ha='center', va='center',
            fontsize=7.2, fontweight='bold', color='#ffffff',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#065f46', edgecolor='#34d399', alpha=0.95))

    # --- LANE R2: EDGE DATA COMMIT UP (x = 187) ---
    # Originates from TRISC2 Pack at (156, 37.0) -> (187, 37.0) -> straight UP to (187, 104.8) -> LEFT to (105, 104.8) -> DOWN to (105, 100.0)
    edge_up_pts = [(156, 37.0), (187, 37.0), (187, 104.8), (105, 104.8), (105, 100.0)]
    ax.plot([p[0] for p in edge_up_pts], [p[1] for p in edge_up_pts],
            color=col_edge_up, linewidth=2.2, linestyle='--')
    
    # Upward Arrowheads along Lane R2
    ax.annotate('', xy=(187, 55), xytext=(187, 50),
                arrowprops=dict(arrowstyle="->", color=col_edge_up, lw=2.2))
    ax.annotate('', xy=(187, 75), xytext=(187, 70),
                arrowprops=dict(arrowstyle="->", color=col_edge_up, lw=2.2))
    ax.annotate('', xy=(187, 92), xytext=(187, 87),
                arrowprops=dict(arrowstyle="->", color=col_edge_up, lw=2.2))
    ax.annotate('', xy=(105, 100.0), xytext=(105, 103.0),
                arrowprops=dict(arrowstyle="->", color=col_edge_up, lw=2.2))

    # Lane R2 Labels
    ax.text(187, 65, "Bypasses L3/L2 Couriers\nDirect FIFO Commit", ha='center', va='center',
            fontsize=6.5, color='#fbcfe8', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='#0f172a', edgecolor=col_edge_up, alpha=0.9))
    ax.text(187, 41.5, "Stage 4 Pack\nR_new", ha='left', va='center', fontsize=6.8, color=col_edge_up, fontweight='bold')

    # Badge for Edge Upward Bus
    ax.text(187, 102.5, "EDGE COMMIT (UP)\nR_new to FIFO", ha='center', va='center',
            fontsize=7.2, fontweight='bold', color='#ffffff',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#831843', edgecolor='#f472b6', alpha=0.95))

    # --- INTERNAL VERTICAL COUPLING (L4 <-> L5) ---
    # Downward arrow from TRISC0 to DEST: Write initial unpack
    ax.annotate('', xy=(71, 26.5), xytext=(71, 35.0),
                arrowprops=dict(arrowstyle="->", color='#c084fc', lw=2.0))
    ax.text(71, 31.0, "Stage 2 Unpack → DEST", ha='center', va='center', fontsize=6.8, color='#c084fc', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.1', facecolor='#0b1120', edgecolor='#a855f7', alpha=0.9))

    # Upward arrow from DEST to TRISC2: Read results to pack
    ax.annotate('', xy=(129, 35.0), xytext=(129, 26.5),
                arrowprops=dict(arrowstyle="->", color='#ec4899', lw=2.0))
    ax.text(129, 31.0, "Stage 4 Pack ← DEST", ha='center', va='center', fontsize=6.8, color='#ec4899', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.1', facecolor='#0b1120', edgecolor='#ec4899', alpha=0.9))

    # Parity check update to CB_TELEMETRY via corridor y=84.6 (in the open gap between L1 and L2)
    # Exits BRISC at (77, 74.5) -> (78.5, 74.5) -> up to y=84.6 -> across to x=139.5 -> up into CB_TELEMETRY (139.5, 90.5)
    ax.plot([77, 78.5, 78.5, 139.5, 139.5], [74.5, 74.5, 84.6, 84.6, 90.5], color='#fde047', linewidth=1.8, linestyle=':')
    ax.annotate('', xy=(139.5, 90.5), xytext=(139.5, 87.5),
                arrowprops=dict(arrowstyle="->", color='#fde047', lw=1.8))
    ax.text(92, 84.6, "Tile 39 Parity Scan (H · c^T = 0) → Convergence Latch", ha='center', va='center',
            fontsize=6.5, color='#fde047', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.12', facecolor='#0b1120', edgecolor='#fde047', alpha=0.9))

    # Save output files
    output_png = Path("docs/figures/tensix_ldpc_dataflow_block_diagram.png")
    output_svg = Path("docs/figures/tensix_ldpc_dataflow_block_diagram.svg")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_png, dpi=300, facecolor='#0b1120', edgecolor='none', bbox_inches='tight')
    plt.savefig(output_svg, facecolor='#0b1120', edgecolor='none', bbox_inches='tight')
    print(f"[SUCCESS] Saved vertical-layer block diagram to: {output_png} and {output_svg}")
    plt.close()

if __name__ == "__main__":
    create_diagram()
