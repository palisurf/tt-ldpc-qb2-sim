# Tenstorrent QuietBox 2 (QB2) LDPC Monte Carlo Simulator

High-performance Monte Carlo LDPC (Low-Density Parity-Check) decoder simulation framework targeting the **Tenstorrent QuietBox 2 (QB2)** equipped with **4 Blackhole processors** configured in a 2x2 mesh topology (440 Tensix compute cores).

---

## Features

- **Hardware Multi-Core Scaling**: Concurrent execution across **440 Tensix cores** on a 2x2 Blackhole mesh via `distributed::MeshDevice`.
- **Arbitrary Graph Parity-Check Decoder**: Support for CCSDS standard codes (such as **AR4JA Rate-1/2**, $N=2560, M=1536$) with irregular degrees and systematic puncturing ($P=512$).
- **On-Device AWGN Noise Generation**: High-throughput Polar Box-Muller Gaussian PRNG implemented directly on the Tensix compute threads.
- **Normalized Min-Sum Decoding**: Floating-point Min-Sum algorithm with scale factor $\alpha = 0.8$ and early syndrome termination ($H \cdot c^T = 0$).
- **High Throughput**: Evaluates up to **50 Msps** (~25 Mbps information rate) across the 440-core mesh.

---

## Directory Structure

```text
├── kernel/
│   ├── compute_trisc_ldpc_awgn_sim.cpp  # On-device decoder & AWGN generator (TRISC compute)
│   ├── reader_ldpc.cpp                  # L1 SRAM input data movement kernel
│   └── writer_ldpc.cpp                  # L1 to DRAM statistics writer kernel
├── matrices/
│   ├── AR4JA_r45_4c_128c_r12.chinn.out  # CCSDS AR4JA rate-1/2 matrix
│   ├── sample_16k.chinn                 # Rate-1/2 16k matrix
│   └── test_code.chinn                  # Small validation matrix
├── tests/
│   ├── test_ar4ja_host.cpp              # Zero-dependency C++ host decoder reference
│   ├── test_kernel_host.cpp             # Bit-level verification test harness
│   ├── test_kernel_ttsim.cpp            # TT-Metal ttsim emulator test
│   └── test_unit_single_core.cpp        # Single-core hardware verification harness
├── CallSim.sh                           # Top-level SNR sweep execution script
├── run_simulation.py                    # Multi-core orchestrator and CSV logger
├── Makefile                             # Build configurations
└── ldpc_sim.cpp                         # Multi-chip 2x2 mesh C++ coordinator
```

---

## Quick Start

### 1. Requirements
- Ubuntu 22.04 / 24.04 LTS
- Tenstorrent QuietBox 2 (Blackhole 2x2 mesh)
- TT-Metalium SDK (`TT_METAL_HOME=/home/ttuser/tt-metal`)
- GCC 11+ with C++20 support

### 2. Compilation
```bash
make ldpc_sim -j
```

### 3. Execution
Run a full SNR sweep using the provided wrapper:
```bash
./CallSim.sh 1.0:2.4:0.2 500 50 50
```
Arguments:
1. `1.0:2.4:0.2` : $E_b/N_0$ range `Start:Stop:Step` in dB
2. `500`         : Maximum blocks per core (up to 220,000 blocks per SNR point)
3. `50`          : Target frame errors for early stopping
4. `50`          : Batch size per core per evaluation step

---

## Verification Results

### CCSDS AR4JA Rate-1/2 ($N=2560, M=1536, P=512$)

| $E_b/N_0$ (dB) | Total Blocks | Bit Errors | Block Errors | BER | FER | Throughput (Msps) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1.00** | 22,000 | 4,328,432 | 19,468 | $9.61 \times 10^{-2}$ | $8.85 \times 10^{-1}$ | 23.18 |
| **1.20** | 22,000 | 2,546,652 | 14,612 | $5.65 \times 10^{-2}$ | $6.64 \times 10^{-1}$ | 23.75 |
| **1.40** | 22,000 | 1,030,340 | 7,656 | $2.29 \times 10^{-2}$ | $3.48 \times 10^{-1}$ | 25.54 |
| **1.60** | 22,000 | 281,152 | 2,528 | $6.24 \times 10^{-3}$ | $1.15 \times 10^{-1}$ | 29.11 |
| **1.80** | 22,000 | 45,032 | 508 | $1.00 \times 10^{-3}$ | $2.31 \times 10^{-2}$ | 33.43 |
| **2.00** | 440,000 | 93,256 | 1,168 | $1.03 \times 10^{-4}$ | $2.65 \times 10^{-3}$ | 41.58 |
| **2.20** | 440,000 | 4,372 | 84 | $4.85 \times 10^{-6}$ | $1.91 \times 10^{-4}$ | 46.55 |
| **2.40** | 6,600,000 | 3,216 | 32 | $2.38 \times 10^{-7}$ | $4.85 \times 10^{-6}$ | 51.08 |
| **2.60** | 11,000,000 | 4 | 4 | $1.78 \times 10^{-10}$ | $3.64 \times 10^{-7}$ | 55.20 |
| **2.80** | 11,000,000 | 0 | 0 | $< 4.4 \times 10^{-11}$ | $< 9.1 \times 10^{-8}$ | **59.07** |
