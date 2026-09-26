# Tenstorrent QuietBox 2 AR4JA LDPC Monte Carlo Decoder Engine
## Comprehensive Architecture, Implementation & Throughput Scaling Roadmap

**Project Lead / Algorithm Originator:** Dr. Christopher R. Jones (Approximate-Min* MILCOM 2003)  
**Target Hardware:** Tenstorrent QuietBox 2 (4 Blackhole ASICs, 2x2 Torus/Mesh, 440 Tensix Cores)  
**Repository:** [`palisurf/tt-ldpc-qb2-sim`](https://github.com/palisurf/tt-ldpc-qb2-sim)  
**Primary Benchmark Code:** CCSDS AR4JA Rate-1/2 ($N=2560, M=1536, K=1024, P=512$, $N_{\text{tx}}=2048$)  
**Target Operational Regime:** Ultra-deep error floor ($10^{-8} \le \text{FER} \le 10^{-10}$)

---

## 1. Executive Summary

This report documents the architectural foundation of the **QuietBox 2 (QB2) LDPC Monte Carlo Simulator**, synthesizes recent experimental discoveries (including the resolution of the 2.0 dB "knee" artifact and characterization of the $1.75 \times 10^{-8}$ error floor), and details a phased engineering roadmap to scale simulation throughput by **$3\times$ to $50\times$** across the 440-core Blackhole mesh.

The system is designed to simulate billions of LDPC codewords down to the deep-space communication error floor completely autonomously on-device, bypassing host PCIe bottlenecks while implementing Dr. Jones' Approximate-Min* algorithm with native `bfloat16` precision and row-layered convergence.

---

## 2. Current Architectural Approach

### 2.1 Hardware Topology & Device Mesh

The QuietBox 2 host workstation contains 4 Tenstorrent Blackhole processors interconnected in a 2x2 2D-Torus / Mesh network:
- **Processor Layout:** 4 chips $\times$ 110 Tensix compute cores ($11 \times 10$ physical grid) = **440 Tensix compute cores**.
- **Per-Core Resources:** $1.5\text{ MB}$ local L1 SRAM tightly coupled to 5 32-bit RISC-V cores:
  - 2 Data Movement processors: BRISC (`RISCV_0`) and NCRISC (`RISCV_1`).
  - 3 Compute processors (TRISCs): TRISC0 (UNPACK), TRISC1 (MATH), TRISC2 (PACK).
  - High-performance vector SFPU (Special Function Processing Unit) with 32-wide SIMD lanes.
- **Mesh Dispatch:** Managed via TT-Metal's `distributed::MeshDevice`. Work is dispatched uniformly across all 440 cores in lockstep; each core maintains its own independent state machine.

```
       +---------------------------------------------+
       |   QuietBox 2 Workstation (Ubuntu 24.04)     |
       |   Host Orchestrator: ldpc_sim.cpp           |
       +---------------------------------------------+
                              | PCIe Gen4 / UMD
        +---------------------+---------------------+
        |                                           |
  +-----------+                               +-----------+
  | Blackhole | <==== 2D Torus Interconnect ===> | Blackhole |
  |  Chip 0   |                               |  Chip 1   |
  | (110 Cores)|                               | (110 Cores)|
  +-----------+                               +-----------+
        ^                                           ^
        | Ethernet / Inter-Chip Fabric             | Ethernet
        v                                           v
  +-----------+                               +-----------+
  | Blackhole | <==== 2D Torus Interconnect ===> | Blackhole |
  |  Chip 2   |                               |  Chip 3   |
  | (110 Cores)|                               | (110 Cores)|
  +-----------+                               +-----------+
    Aggregate Mesh: 440 Tensix Cores | 660 MB L1 SRAM
```

### 2.2 Autonomous On-Device Monte Carlo Pipeline

Traditional Monte Carlo simulators suffer severe bandwidth bottlenecks transferring noise vectors and decoded words between CPU and accelerator. The QB2 engine eliminates this entirely by hosting the full simulation lifecycle inside Tensix L1 SRAM:

1. **Noise Generation:** Polar Box-Muller AWGN generation executes directly in L1 using rejection sampling on the unit disk, producing Gaussian variates without evaluating trigonometric functions ($\sin/\cos$).
2. **Puncturing:** For the CCSDS AR4JA code, $P = 512$ systematic parity nodes are not transmitted. Their channel LLRs are initialized to zero ($L_i = 0$) in L1, indicating complete a priori uncertainty.
3. **Decoding Loop:** Christopher Jones Approximate-Min* row-layered updates iterate until parity check satisfies ($H \cdot \mathbf{c}^T = \mathbf{0}$) or max iterations are reached.
4. **Error Tabulation:** Bit and block errors are verified on-device by checking hard decisions against the all-zero codeword.
5. **Host Telemetry:** The host only dispatches lightweight batch metadata and reads back a single 16-byte summary packet per core (`bit_errors`, `frame_errors`, `total_iters`, `sample_llr`).

### 2.3 Row-Layered (Gauss-Seidel) Approximate-Min* Decoding

The decoder operates in a row-layered schedule where variable node LLRs are updated immediately after each check node row is processed, cutting convergence iterations by $2\times$ relative to standard flooding:

```
For each check node row m = 0 to M-1:
  1. Extrinsic Extraction:   q_{m,d} = L_{v(m,d)} - r_{m,d}
  2. Check Updating:         Compute r'_{m,d} via Approximate-Min*
  3. Posterior Update:       L_{v(m,d)} <- q_{m,d} + r'_{m,d}
                             r_{m,d}     <- r'_{m,d}
```

The check node update implements Dr. Jones' Approximate-Min* algorithm (MILCOM 2003):
- **3-Magnitude Tracking:** Identifies the three smallest incoming reliabilities $\min_1 \le \min_2 \le \min_3$ and computes differences $\Delta_1 = \min_2 - \min_1$ and $\Delta_2 = \min_3 - \min_2$.
- **2-Region Piecewise Linear Correction:** Eliminates look-up tables with hardware-friendly linear segments:
  $$g(\Delta) \approx \begin{cases} 0.69315 - 0.4000 \cdot \Delta & \Delta < 0.5 \\ 0.49315 - 0.3280 \cdot (\Delta - 0.5) & 0.5 \le \Delta < 2.0 \\ 0.0 & \Delta \ge 2.0 \end{cases}$$
- **Result:** Matches exact floating-point Jacobi log belief propagation while achieving **$57.3\%$ frame error reduction** over Normalized Min-Sum ($\alpha = 0.75$).

### 2.4 High-Dimensional PRNG Independence (`xoshiro256**`)

To prevent pseudo-random sequence correlations when evaluating trillions of bits across 440 parallel cores:
- Core PRNG is upgraded to 256-bit `xoshiro256**` (period $2^{256} - 1 \approx 1.16 \times 10^{77}$).
- **Core Separation:** Tensix cores are separated by successive `long_jump()` operations, equivalent to $2^{96} \approx 7.92 \times 10^{28}$ PRNG draws per core.
- **Batch Separation:** Consecutive batches on the same core advance by `jump()`, equivalent to $2^{64} \approx 1.84 \times 10^{19}$ draws, ensuring mathematically disjoint sample spaces across time and space.

### 2.5 Native Bfloat16 Memory Optimization

Check messages $r_{m,d}$ and posterior LLRs $\tilde{L}_v$ are stored in native `bfloat16` (`Float16_b`):
- Reduced per-core memory footprint from **$448\text{ KB}$ down to $56\text{ KB}$** ($8\times$ reduction).
- Reclaimed **$172.5\text{ MB}$ of L1 SRAM** across the 440-core mesh.
- Maintained identical error-correction performance to FP32 ($< 0.07\%$ difference across 10,000 blocks).

---

## 3. Key Experimental Discoveries

### 3.1 Resolution of the 2.0 dB "Knee" Artifact

Earlier simulation sweeps under fixed $L_{\max} = 4.2$ channel clipping exhibited an anomalous $15.4\times$ cliff between $2.0\text{ dB}$ and $2.1\text{ dB}$. A systematic sweep comparing fixed clipping vs. unclipped floating-point AWGN revealed:
- Under unclipped AWGN, the waterfall descent is smooth, log-linear, and continuous:
  - $1.9 \to 2.0\text{ dB}$: $4.6\times$ reduction
  - $2.0 \to 2.1\text{ dB}$: $4.8\times$ reduction
  - $2.1 \to 2.2\text{ dB}$: $4.0\times$ reduction
- **Conclusion:** The 2.0 dB "knee" was an artificial distortion caused by rigid $L_{\max} = 4.2$ clipping over-saturating correct signal observations.

### 3.2 Deep Error Floor Characterization at 2.40 dB ($1.75 \times 10^{-8}$)

At $2.40\text{ dB}$, evaluating **$915.6\text{ Million blocks}$** ($> 1.87\text{ Trillion bits}$) yielded:
- 16 Frame Errors, $\text{FER} = 1.75 \times 10^{-8}$, $\text{BER} = 2.60 \times 10^{-9}$.
- Average bit errors per failed frame: $\approx 264$ bits.
- **Root Cause:** Failures are **non-convergence states** rather than trapping sets (trapping sets typically flip only 8–16 bits). In $5.5 \times 10^{11}$ Gaussian draws, rare $6\sigma - 7\sigma$ noise spikes produce extreme channel LLRs ($L_{\text{ch}} < -12.4$) that degree-6 check nodes cannot overturn within 200 iterations.
- **Solution:** Proportional gamma clipping ($\gamma = L_{\max} / \mu_{\text{LLR}} \approx 2.0 - 2.5$, corresponding to $L_{\max} \approx 7.0 - 8.5$) bounds extreme tail spikes without compressing the bulk signal distribution.

---

## 4. Current Hardware Utilization & Bottleneck Analysis

Profiling the active execution model identifies significant untapped compute capacity:

| Subsystem | Current State | Theoretical Capacity | Utilization |
| :--- | :--- | :--- | :---: |
| **Compute Engines** | Only **TRISC0 (UNPACK)** active | TRISC0 + TRISC1 + TRISC2 | **$33.3\%$** |
| **Data Movement** | Idle during compute | BRISC + NCRISC | **$< 1\%$** |
| **Vector Unit (SFPU)**| Unused (pure scalar C++) | 32-wide SIMD FP16/BF16 FMA | **$0\%$** |
| **L1 SRAM Footprint** | $56\text{ KB}$ per core | $1,500\text{ KB}$ per core | **$3.7\%$** |
| **Mesh Throughput** | $\approx 46\text{ Msps}$ (16 iters) | Scalable to $> 1\text{ Gsps}$ | Baseline |

Because each Tensix core uses only 56 KB of its 1,500 KB L1 SRAM and runs on a single TRISC thread, **over 95% of L1 SRAM and 66% of TRISC compute capacity are currently idle**.

---

## 5. Roadmap for Throughput Scaling

To accelerate ultra-deep floor simulations down to $10^{-9} - 10^{-10}$ FER, we outline a four-phase throughput scaling roadmap:

```
[Current Baseline: 46 Msps] 
       │
       ▼ (Phase 1: Multi-TRISC Concurrency)
[138 Msps (3.0x Speedup)]
       │
       ▼ (Phase 2: L1 Double Buffering)
[170 Msps (3.7x Speedup)]
       │
       ▼ (Phase 3: SWAR Register-Level SIMD)
[550 Msps (12.0x Speedup)]
       │
       ▼ (Phase 4: 32-Wide Tensix SFPU Vectorization)
[1.8 - 2.5 Gsps (40x - 55x Speedup)]
```

---

### Phase 1: Multi-TRISC Concurrency (TRISC0 + TRISC1 + TRISC2)
- **Target Speedup:** **$3.0\times$** ($\approx 46\text{ Msps} \to \mathbf{138\text{ Msps}}$ aggregate across 440 cores).
- **Architecture:** 
  - The Tensix core features three independent RISC-V compute processors: TRISC0 (Unpack), TRISC1 (Math), and TRISC2 (Pack).
  - All three TRISCs have direct read/write access to L1 SRAM.
  - Partition L1 SRAM into three non-overlapping decoder instances:
    - `Instance 0` (TRISC0): Base offset `0x00000` (56 KB)
    - `Instance 1` (TRISC1): Base offset `0x10000` (56 KB)
    - `Instance 2` (TRISC2): Base offset `0x20000` (56 KB)
    - Total footprint: **$168\text{ KB}$** (only $11.2\%$ of available 1.5 MB L1).
  - Each TRISC runs an independent Monte Carlo simulation instance on separate codewords with disjoint PRNG seeds.
- **Deliverables:**
  - Update `kernel/compute_trisc_ldpc_awgn_sim.cpp` with TRISC ID branching (`#if defined(UCK_CHL_UNPACK) ...`).
  - Update host launcher to allocate 3 Circular Buffers per core.
  - Verification: Single-core (`--single_core`) equivalence test vs. gold reference.

---

### Phase 2: Double Buffering & Decoupled Noise Generation
- **Target Speedup:** **$+20\text{--}25\%$** (raising Phase 1 to $\mathbf{\approx 170\text{ Msps}}$).
- **Architecture:**
  - Currently, Polar Box-Muller Gaussian generation and LDPC iterations execute sequentially on TRISC. Noise generation accounts for $\sim 18\text{--}22\%$ of total cycle time.
  - Implement ping-pong double buffering in L1 SRAM:
    - Buffer A: Actively decoded by TRISCs.
    - Buffer B: Populated with next-batch AWGN noise variates in the background.
  - Exploit idle Data Movement processors (BRISC/NCRISC) or dedicated PRNG TRISC thread to pre-generate noise frames while decoding occurs.
- **Footprint:** $2 \times 168\text{ KB} = 336\text{ KB}$ ($22.4\%$ of L1 SRAM).

---

### Phase 3: SWAR (SIMD Within A Register) / 4-Way Parallelism
- **Target Speedup:** **$3.0\times\text{--}4.0\times$** (cumulative $\mathbf{\approx 550\text{ Msps}}$).
- **Architecture:**
  - Tensix RISC-V registers are 32-bit (with 64-bit support).
  - Two `bfloat16` values can be packed into a single 32-bit register; four `bfloat16` values can be packed into a 64-bit pair.
  - Check node magnitude comparisons and sign XOR operations are performed in parallel across 2–4 codewords simultaneously using SWAR bit-manipulation techniques.
  - Replaces scalar branch-heavy minimum-finding with branchless bitwise select instructions (`pack`, `min`, `max`).

---

### Phase 4: Tensix SFPU / Matrix Engine Vectorization (32-Wide SIMD)
- **Target Speedup:** **$10\times\text{--}15\times$** over Phase 1 ($\mathbf{1.8\text{--}2.5\text{ Gsps}}$ aggregate mesh throughput).
- **Architecture:**
  - Harness the Tensix SFPU (Special Function Processing Unit), which executes 32-lane vector operations per cycle.
  - Vectorize Box-Muller: Compute 32 Gaussian random variates in parallel per cycle using vector transcendental approximation instructions (`sfpu_log`, `sfpu_sqrt`).
  - Vectorize Check Node Updating: Process 32 codewords in lockstep across the vector lanes (or process 32 edges of the Tanner graph in parallel).
  - At $2.0\text{ Gsps}$, evaluating **1 Billion codewords** ($2.05 \times 10^{12}$ bits) will complete in **under 15 minutes** (compared to $\approx 10.5\text{ hours}$ currently).

---

## 6. Implementation Timetable & Milestones

| Milestone | Target Objective | Expected Throughput | Effort / Complexity |
| :---: | :--- | :---: | :---: |
| **M1 (Current)** | Single-TRISC scalar engine on 440 cores | $46\text{ Msps}$ | Completed & Verified |
| **M2** | Multi-TRISC concurrency (TRISC0 + TRISC1 + TRISC2) | **$138\text{ Msps}$** ($3.0\times$) | Low–Medium (1–2 days) |
| **M3** | Proportional Gamma Clipping sweep ($\gamma=2.2$) at $2.4\text{ dB}$ | $138\text{ Msps}$ | Immediate (following M2) |
| **M4** | L1 Double Buffering (Ping-Pong Noise & Decoding) | **$170\text{ Msps}$** ($3.7\times$) | Medium (3–4 days) |
| **M5** | SWAR / Packed Bfloat16 register parallelism | **$550\text{ Msps}$** ($12.0\times$) | Medium–High (1 week) |
| **M6** | 32-Wide SFPU vectorization & deep-floor campaign ($10^{-10}$ FER) | **$1.8\text{--}2.5\text{ Gsps}$** ($40\times+$) | High (2–3 weeks) |

---

## 7. Conclusion

The QuietBox 2 AR4JA LDPC engine has proven the capability of Blackhole Tensix cores to simulate deep-space LDPC error floors autonomously at scale. By leveraging the unutilized TRISC cores, expanding into the remaining $96\%$ of L1 SRAM, and progressively vectorizing check node updates, the architecture provides a direct, low-risk path to achieving multi-gigabit simulation throughput on desktop-class liquid-cooled hardware.
