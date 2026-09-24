# Tenstorrent QuietBox 2 (QB2) LDPC Monte Carlo Simulator

High-performance, multi-chip Monte Carlo LDPC (Low-Density Parity-Check) simulation framework targeting the **Tenstorrent QuietBox 2 (QB2)** equipped with **4 Blackhole processors** in a 2x2 mesh topology (**440 Tensix compute cores**).

The simulator features a hardware-optimized implementation of the **Approximate-Min\*** constraint node updating algorithm developed by **Christopher R. Jones et al.**, delivering full Belief Propagation (BP) error-correction performance at the execution speed of Min-Sum.

---

## Key Features

- **Massive Multi-Core Parallelism**: Lockstep and independent Monte Carlo dispatch across **440 Tensix cores** on a 2x2 Blackhole mesh via `distributed::MeshDevice`.
- **Approximate-Min\* Constraint Updating (Default)**: Hardware LUT-free dual-magnitude check node processing with piecewise-linear correction, providing a **57% reduction in frame errors** over Normalized Min-Sum.
- **Normalized Min-Sum (Compile-Time / CLI Option)**: Normalized Min-Sum ($\alpha = 0.75$) retained as an opt-in toggle (`-n` / `--nms`) via TT-Metal JIT preprocessor defines.
- **Arbitrary Graph Parity-Check Decoder**: Support for CCSDS deep-space standards (including **AR4JA Rate-1/2**, $N=2560, M=1536, P=512$) with irregular degrees and systematic puncturing.
- **On-Device AWGN Noise Generation**: High-throughput Polar Box-Muller Gaussian PRNG using 128-bit `Xoshiro128+` implemented directly on TRISC compute threads (period $2^{128}-1$).
- **Native Bfloat16 Precision**: Check messages and LLR buffers execute in `bfloat16`, reducing L1 SRAM footprint from 448 KB to 56 KB per core with zero loss in error-correction performance.
- **Early Syndrome Termination**: Fast inline parity check ($H \cdot \mathbf{c}^T = \mathbf{0}$) terminating valid codewords early to maximize throughput.

---

## Constraint Node Updating Techniques

The simulator implements and benchmarks three distinct check node processing techniques on the CCSDS AR4JA rate-1/2 code:

### 1. Normalized Min-Sum (NMS)
In standard Min-Sum, check node outgoing messages approximate the log-MAP operation by finding the minimum incoming magnitude. Because Min-Sum systematically overestimates message reliability, an empirical attenuation factor $\alpha \approx 0.75\text{--}0.80$ is applied:
$$r_{m \to n} = \alpha \cdot \left( \prod_{n' \in \mathcal{N}(m) \setminus \{n\}} \text{sgn}(q_{n' \to m}) \right) \cdot \min_{n' \in \mathcal{N}(m) \setminus \{n\}} |q_{n' \to m}|$$
- **Complexity**: Tracks the two smallest magnitudes ($\min_1, \min_2$). Requires 1 floating-point multiply per outgoing edge.
- **Limitation**: Suffers a $0.2\text{--}0.3\text{ dB}$ waterfall penalty relative to belief propagation.

### 2. Approximate-Min\* (Christopher Jones PWL)
Exact belief propagation in the log-domain relies on the Jacobi logarithm:
$$\min^*(a, b) = \ln(e^{-a} + e^{-b}) = \min(a, b) - \ln(1 + e^{-|a - b|}) = \min(a, b) - g(|a - b|)$$

As established by **Christopher Jones et al. (MILCOM 2003)**, full check node updating does **not** require evaluating pairwise Jacobi trees for every outgoing edge. Instead:
1. Only the **three smallest incoming magnitudes** ($\min_1 \le \min_2 \le \min_3$) are tracked.
2. Only two difference terms are computed: $\Delta_1 = \min_2 - \min_1$ and $\Delta_2 = \min_3 - \min_2$.
3. Exactly **two outgoing magnitudes** are generated for the entire check node row:
   $$r_{\text{mag, all}} = \max\bigl(0, \; \min_1 - g(\Delta_1)\bigr) \quad (\text{for all non-minimum edges } d \ne d_{\min})$$
   $$r_{\text{mag, min}} = \max\bigl(0, \; \min_2 - g(\Delta_2)\bigr) \quad (\text{for the minimum edge } d = d_{\min})$$
4. The transcendental correction $g(\Delta) = \ln(1 + e^{-\Delta})$ is replaced by a hardware shift-add / FMA friendly **piecewise-linear (PWL)** approximation requiring **zero lookup tables (LUTs)**:
   $$g(\Delta) \approx \begin{cases} 0.69315 - 0.40 \cdot \Delta, & \Delta < 0.5 \\ 0.49315 - 0.328 \cdot (\Delta - 0.5), & 0.5 \le \Delta < 2.0 \\ 0, & \Delta \ge 2.0 \end{cases}$$

### 3. Exact Jacobi Log $\min^*$ (Full Belief Propagation)
Evaluates $g(\Delta) = \ln(1 + e^{-\Delta})$ using exact transcendental logarithm and exponential functions. While representing the theoretical optimum BP baseline, transcendental evaluation in software on embedded RISC-V cores incurs substantial cycle overhead.

---

## Experimental Benchmark & Comparison

### 1. Head-to-Head Algorithmic Verification (10,000 Codewords, $E_b/N_0 = 1.6\text{ dB}$)
Tested on the CCSDS AR4JA rate-1/2 code ($N=2560, M=1536, P=512$, 2048 transmitted symbols) in the waterfall transition region:

| Algorithm | Frame Errors | FER | Bit Errors | BER | Avg Iters | Error Reduction vs NMS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normalized Min-Sum ($\alpha = 0.75$)** | 1,092 | $0.1092$ | 117,529 | $5.74 \times 10^{-3}$ | 11.44 | Baseline |
| **Normalized Min-Sum ($\alpha = 0.80$)** | 938 | $0.0938$ | 101,659 | $4.96 \times 10^{-3}$ | 11.19 | $+14.1\%$ |
| **Approximate-Min\* (Jones PWL)** | **460** | **$0.0460$** | **49,319** | **$2.41 \times 10^{-3}$** | **10.10** | **$+57.9\%$** |
| **Exact Jacobi Log $\min^*$** | 445 | $0.0445$ | 46,432 | $2.27 \times 10^{-3}$ | 10.03 | $+59.2\%$ |

> **Key Result:** The LUT-free piecewise-linear Approximate-Min\* achieves **$97.8\%$ of the exact Jacobi log coding gain** (460 vs 445 frame errors), while cutting frame errors by more than half compared to Normalized Min-Sum and converging in fewer iterations.

---

### 2. Multi-Core Hardware Execution on QuietBox 2 (440 Tensix Cores)
Full hardware simulation across 440 Tensix cores ($22,000$ codewords, $45.056\text{M}$ transmitted symbols) at $E_b/N_0 = 1.6\text{ dB}$:

| Hardware Metric | Normalized Min-Sum (`--nms`) | Approximate-Min\* (Default) | Hardware Impact |
| :--- | :---: | :---: | :--- |
| **Codewords Simulated** | 22,000 | 22,000 | Identical workload |
| **Frame Errors** | 2,556 | **1,096** | **$57.1\%$ reduction** |
| **Frame Error Rate (FER)** | $0.11618$ | **$0.04982$** | **$2.33\times$ lower FER** |
| **Bit Errors** | 279,164 | **124,544** | **$55.4\%$ reduction** |
| **Bit Error Rate (BER)** | $6.1959 \times 10^{-3}$ | **$2.7642 \times 10^{-3}$** | **$2.24\times$ lower BER** |
| **Hardware Throughput** | **$29.06\text{ Msps}$** | **$28.17\text{ Msps}$** | **$< 3\%$ cycle difference** |

---

### 3. QuietBox 2 Microarchitecture & Throughput Analysis

On Tenstorrent's Tensix processors (TRISC RISC-V cores), transcendental functions have no direct hardware ALU instruction:
- **Exact Jacobi Log $\min^*$** requires polynomial approximations for both $\exp(-\Delta)$ and $\ln(1+x)$, totaling $\approx 65\text{--}70\text{ cycles}$ per evaluation ($\approx 135\text{ cycles}$ per check row).
- **Approximate-Min\* (Jones PWL)** executes in **$\approx 3\text{--}4\text{ cycles}$** using register comparisons and a single Fused Multiply-Add (`fmadd.s` / `fmsub.s`), requiring only $\approx 7\text{ cycles}$ per check row.

| Metric (AR4JA Rate-1/2, 10 iters) | Exact Jacobi $\min^*$ | Approximate-Min\* (PWL) | Speedup / Advantage |
| :--- | :---: | :---: | :--- |
| **Correction Evaluation Latency** | $\approx 65\text{--}70\text{ cycles}$ | **$\approx 3\text{--}4\text{ cycles}$** | **$18\times\text{--}20\times$ faster** |
| **Total Check Node Cost** | $\approx 205\text{ cycles}$ / row | **$\approx 77\text{ cycles}$** / row | **$2.66\times$ faster check node** |
| **Estimated Throughput (440 Cores)** | $\approx 12.3\text{ Msps}$ | **$28.17\text{ Msps}$** | **$+129\%$ ($2.3\times$) Throughput Advantage** |
| **vs Classical Full-Tree Jacobi** | $< 3.2\text{ Msps}$ | **$28.17\text{ Msps}$** | **$8.8\times$ Throughput Advantage** |

Furthermore, because Approximate-Min* operates entirely within the floating-point register file, it avoids consuming precious L1 SRAM for lookup tables (LUTs) and eliminates cache bank conflicts.

---

### 4. Bfloat16 Precision Validation & Memory Optimization

To maximize L1 SRAM utilization on Tenstorrent's native `bfloat16` architecture, message storage (`r_msg` and `channel_llrs`) is mapped to native `Float16_b` representation:

| Algorithm | Precision | Frame Errors (10k CW) | FER (1.6 dB) | L1 Scratchpad / Core |
| :--- | :---: | :---: | :---: | :---: |
| **Normalized Min-Sum** | Float32 | 1,145 | $0.1145$ | $448\text{ KB}$ |
| **Normalized Min-Sum** | **Bfloat16** | **1,138** | **$0.1138$** | **$56\text{ KB}$** |
| **Approximate-Min\*** | Float32 | 499 | $0.0499$ | $448\text{ KB}$ |
| **Approximate-Min\*** | **Bfloat16** | **506** | **$0.0506$** | **$56\text{ KB}$** |

- **Zero Decoding Loss**: Both algorithms retain numerical parity with IEEE FP32 within $\pm 0.07\%$ FER.
- **$8\times$ Memory Footprint Reduction**: Allocations drop from $448\text{ KB}$ to $56\text{ KB}$ per core, freeing $172.5\text{ MB}$ of high-speed SRAM across the 440-core mesh for multi-codeword concurrency.

---

## Directory Structure

```text
├── kernel/
│   ├── compute_trisc_ldpc_awgn_sim.cpp  # On-device decoder (Approx-Min* & NMS, bfloat16) & AWGN generator
│   ├── reader_ldpc.cpp                  # L1 SRAM parity matrix reader kernel
│   └── writer_ldpc.cpp                  # L1 to DRAM statistics writer kernel
├── matrices/
│   ├── AR4JA_r45_4c_128c_r12.chinn.out  # CCSDS AR4JA rate-1/2 matrix (N=2560, M=1536)
│   ├── sample_16k.chinn                 # Rate-1/2 16k matrix
│   └── test_code.chinn                  # Small validation matrix
├── tests/
│   ├── test_amin_star.cpp               # Comparative test harness (NMS vs A-Min* vs Exact Jacobi)
│   ├── test_bf16_amin.cpp               # FP32 vs Bfloat16 precision benchmark harness
│   ├── test_ar4ja_host.cpp              # Zero-dependency host C++ reference decoder
│   ├── test_kernel_host.cpp             # Bit-level math verification test harness
│   ├── test_kernel_ttsim.cpp            # TT-Metal emulator test
│   └── test_unit_single_core.cpp        # Single-core hardware verification harness
├── CallSim.sh                           # Top-level sweep script with NMS/A-Min* flags
├── run_simulation.py                    # Multi-core orchestrator and CSV logger
├── Makefile                             # Build rules for host and device binaries
└── ldpc_sim.cpp                         # Multi-chip 2x2 mesh C++ coordinator
```

---

## Quick Start

### 1. Requirements
- Ubuntu 22.04 / 24.04 LTS
- Tenstorrent QuietBox 2 (4 Blackhole processors in 2x2 mesh)
- TT-Metalium SDK (`export TT_METAL_RUNTIME_ROOT=/home/ttuser/tt-metal`)
- GCC 11+ with C++20 support

### 2. Compilation
```bash
make ldpc_sim -j
```

### 3. Running Simulations

#### Approximate-Min\* (Default):
```bash
# Sweep Eb/N0 from 1.6 to 2.4 dB with 50 blocks/core batch size
./CallSim.sh 1.6:2.4:0.2 100000 50 5000
```

#### Normalized Min-Sum (NMS Fallback):
To run with Normalized Min-Sum ($\alpha = 0.75$), append `--nms` or `-n`:
```bash
./CallSim.sh 1.6:2.4:0.2 100000 50 5000 --nms
```

---

## References

1. **C. Jones, E. Valles, M. Smith, and J. Villasenor**, *"Approximate-Min\* constraint node updating for LDPC and turbo decoding,"* in *Proceedings of the IEEE Military Communications Conference (MILCOM)*, Boston, MA, USA, Oct. 2003, vol. 1, pp. 157–162. doi: [10.1109/MILCOM.2003.1290100](https://doi.org/10.1109/MILCOM.2003.1290100).
2. **W. E. Ryan, S. Lin, and S. G. Wilson**, *Channel Codes: Classical and Modern*, Cambridge University Press, 2009. (Detailed coverage of dual-min and Approximate-Min\* check node implementations).
3. **CCSDS 131.0-B-5**, *"TM Synchronization and Channel Coding,"* Blue Book, Consultative Committee for Space Data Systems, Washington, D.C., USA. (AR4JA deep-space LDPC code specifications).
