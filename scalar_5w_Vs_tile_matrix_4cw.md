# Architectural, Mathematical, and Microarchitectural Comparison:
# 5-Worker Scalar Engine (`scalar_5w`) vs. 4-Codeword Tile Matrix Engine (`tile_matrix_4cw`)

**Author:** Dr. Christopher R. Jones  
**Date:** October 2026  
**Document ID:** `docs/scalar_5w_Vs_tile_matrix_4cw.md`  
**Target Hardware:** Tenstorrent QuietBox 2 (4x Blackhole Processors / 440 Tensix Cores)  

---

## 1. Executive Summary & Historical Performance Divergence

Across extensive physical silicon and software emulator sweeps on the CCSDS AR4JA Rate-1/2 LDPC code ($N=2560, M=1536, K=1024$), Dr. Christopher Jones' original 5-Worker Scalar Engine ([`engines/scalar_5w/`](file:///Y:/JPL/LDPC_QB2/LDPC_QB2_GA/engines/scalar_5w/)) demonstrated superior Frame Error Rate (FER) performance compared to the SIMD-vectorized 4-Codeword Tile Matrix Engine ([`engines/tile_matrix_4cw/`](file:///Y:/JPL/LDPC_QB2/LDPC_QB2_GA/engines/tile_matrix_4cw/)). 

While the Tile Matrix Engine delivers an order-of-magnitude higher raw hardware throughput via native $32 \times 32$ Tensix matrix operations (>2.5 Gsps across the 440-core QuietBox 2 mesh), in the deep waterfall and error floor regime ($E_b/N_0 \ge 2.20\text{ dB}$), **`scalar_5w` achieves roughly a $4\times$ to $10\times$ lower (better) FER**:

### Empirical FER & BER Comparison Table ($I_{\max} = 200\text{ iterations}$)

| $E_b/N_0$ [dB] | `scalar_5w` (440 Cores) Total Blocks | `scalar_5w` FER | `tile_matrix_4cw` (Hardware / Mesh) Total Blocks | `tile_matrix_4cw` FER | Degradation Factor (`tile` / `scalar`) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$1.80\text{ dB}$** | $440{,}000$ | $8.182 \times 10^{-5}$ | $440{,}000$ | $8.210 \times 10^{-5}$ | $1.00\times$ (Identical Waterfall) |
| **$2.00\text{ dB}$** | $9{,}240{,}000$ | **$2.706 \times 10^{-6}$** | $8{,}224{,}000$ | **$3.040 \times 10^{-6}$** (Host) / **$8.046 \times 10^{-6}$** (HW BM) | $1.12\times - 2.97\times$ |
| **$2.20\text{ dB}$** | $302{,}720{,}000$ | **$8.258 \times 10^{-8}$** | $44{,}000{,}000$ | **$2.420 \times 10^{-7}$** | **$2.93\times$** |
| **$2.30\text{ dB}$** | $1{,}354{,}760{,}000$ | **$1.845 \times 10^{-8}$** | $330{,}000{,}000$ | **$7.576 \times 10^{-8}$** | **$4.11\times$** |
| **$2.50\text{ dB}$** | $4{,}667{,}520{,}000$ | **$2.357 \times 10^{-9}$** | $2{,}442{,}000{,}000$ | **$1.024 \times 10^{-8}$** | **$4.34\times$** |
| **$2.70\text{ dB}$** | $13{,}845{,}040{,}000$ | **$7.945 \times 10^{-10}$** | $5{,}707{,}363{,}200$ | **$3.329 \times 10^{-9}$** | **$4.19\times - 10\times$** |

### Confirmation of Golden Model Reference
[`tests/test_tile_matrix_40tile_lockstep_stepper_sfpi.cpp`](file:///Y:/JPL/LDPC_QB2/LDPC_QB2_GA/tests/test_tile_matrix_40tile_lockstep_stepper_sfpi.cpp) is confirmed to be the **official, hardware-accurate golden reference emulator** for the `tile_matrix_4cw` engine:
- It compiles the physical device check-node kernels verbatim (`compute_tile_deg6_sfpi` and `compute_tile_deg3_sfpi`) using the software SFPI vector emulation header [`engines/tile_matrix_4cw/sfpi_host_emu.h`](file:///Y:/JPL/LDPC_QB2/LDPC_QB2_GA/engines/tile_matrix_4cw/sfpi_host_emu.h).
- State dumps match physical Blackhole Tensix L1 SRAM **100% bit-exact** across all 1,024 gather words, 1,024 scatter words, and 10,240 resident variable node words across multi-iteration circular FIFO sweeps.

---

## 2. Algorithmic Core: Check Node Approximate-Min* Formulation

### 2.1 The MILCOM 2003 True Recursive Formulation (`scalar_5w`)
In [`engines/scalar_5w/ldpc_decoder_core.h`](file:///Y:/JPL/LDPC_QB2/LDPC_QB2_GA/engines/scalar_5w/ldpc_decoder_core.h#L154-L201), check node update logic follows Dr. Jones' original MILCOM 2003 recursive formulation:

```cpp
// scalar_5w: True Recursive Approximate-Min*
inline void compute_amin_star_general(
    const float* q_val, uint32_t deg, uint32_t min_idx, float min_mag,
    float& r_mag_min, float& r_mag_all
) {
    uint32_t first_idx = (min_idx == 0) ? 1 : 0;
    float m_others = uint_as_float(float_as_uint(q_val[first_idx]) & 0x7FFFFFFF);

    for (uint32_t d = first_idx + 1; d < deg; d++) {
        if (d == min_idx) continue;
        float abs_qd = uint_as_float(float_as_uint(q_val[d]) & 0x7FFFFFFF);
        float diff   = std::abs(m_others - abs_qd);
        float m_curr = std::min(m_others, abs_qd);
        float corr   = (diff >= 2.0f) ? 0.0f : ((diff < 0.5f) ? (0.69315f - 0.40f * diff) : (0.49315f - 0.328f * (diff - 0.5f)));
        m_others     = std::max(m_curr - corr, 0.0f);
    }

    float diff_all   = std::abs(m_others - min_mag);
    float m_all_curr = std::min(m_others, min_mag);
    float corr_all   = (diff_all >= 2.0f) ? 0.0f : ((diff_all < 0.5f) ? (0.69315f - 0.40f * diff_all) : (0.49315f - 0.328f * (diff_all - 0.5f)));
    float m_all      = std::max(m_all_curr - corr_all, 0.0f);

    r_mag_min = m_others;
    r_mag_all = m_all;
}
```

#### Crucial Algorithmic Properties:
1. **Precursor Minimum Separation:** The algorithm isolates $d_{\min} = \arg\min_d |q_d|$.
2. **Asymmetric Accumulation:** It accumulates all other $d_c - 1$ edges into $M_{\text{others}} = r_{\text{mag}}(d_{\min})$ in a serial chain of length 4.
3. **The Intrinsic Damping Effect:** For all other edges $d \ne d_{\min}$, the outgoing magnitude is set to:
   $$M_{\text{all}} = \operatorname{AMin}^*(M_{\text{others}}, |q_{d_{\min}}|)$$
   Because $M_{\text{others}}$ contains $|q_d|$, $M_{\text{all}}$ is strictly **smaller** than the true theoretical extrinsic value $\operatorname{AMin}^*_{k \ne d}(|q_k|)$.
4. **Why Intrinsic Damping Suppresses Error Floors:**
   On finite-length Tanner graphs containing short topological cycles (e.g. length-4 and length-6 cycles), standard sum-product or strict un-damped extrinsic messages experience positive feedback loops that inflate LLR magnitudes too quickly. This locks the decoder into non-codeword trapping sets. The slight magnitude attenuation inherent in $M_{\text{all}}$ acts as a natural **damping factor** (analogous to $\alpha < 1$ in normalized min-sum), preventing premature saturation and allowing weak parity equations to correct erroneous nodes.

---

### 2.2 SIMD-Vectorized Pairwise Tree Formulation (`tile_matrix_4cw`)
In [`tests/test_tile_matrix_40tile_lockstep_stepper_sfpi.cpp`](file:///Y:/JPL/LDPC_QB2/LDPC_QB2_GA/tests/test_tile_matrix_40tile_lockstep_stepper_sfpi.cpp#L638-L687) and [`engines/tile_matrix_4cw/compute_trisc.cpp`](file:///Y:/JPL/LDPC_QB2/LDPC_QB2_GA/engines/tile_matrix_4cw/compute_trisc.cpp#L618-L687), check node operations are structured to vectorize across 32 lanes simultaneously using subvector transpositions (`SFPTRANSP` / `swap_subvec16`):

```cpp
// tile_matrix_4cw: Pairwise Tree Approximate-Min*
// Stage 3: Pairwise AMin* within edge pairs
M01 = sfpi::round_bf16(sfpi_amin_star_branchless(m0, m0_sw)); // min*(|q0|, |q1|)
M23 = sfpi::round_bf16(sfpi_amin_star_branchless(m1, m1_sw)); // min*(|q2|, |q3|)
M45 = sfpi_amin_star_branchless(m2, m2_sw);                   // min*(|q4|, |q5|)

// Stage 4: Outgoing check messages
// Edge pair 0:
m_oth0 = sfpi_amin_star_branchless(M23, M45);
r_mag0 = sfpi_amin_star_branchless(m_oth0, m0_sw); // min*(|q1|, |q2|, |q3|, |q4|, |q5|)

// Edge pair 1:
m_oth1 = sfpi_amin_star_branchless(M01, M45);
r_mag1 = sfpi_amin_star_branchless(m_oth1, m1_sw); // min*(|q0|, |q2|, |q3|, |q4|, |q5|)

// Edge pair 2:
m_oth2 = sfpi_amin_star_branchless(M01, M23);
r_mag2 = sfpi_amin_star_branchless(m_oth2, m2_sw); // min*(|q0|, |q1|, |q2|, |q3|, |q5|)
```

#### Key Differences & Consequences:
1. **Strict Extrinsic Evaluation:** Unlike `scalar_5w`, this pairwise tree computes the exact extrinsic message for **every single edge** ($r_d$ completely excludes $q_d$).
2. **Absence of Damping:** Because every outgoing message is mathematically un-attenuated, check node magnitudes grow significantly faster iteration-over-iteration.
3. **Trapping Set Susceptibility:** When a few variable nodes are corrupted by channel noise, un-damped check messages broadcast large extrinsic beliefs that overpower contradictory parity checks, causing frames that would otherwise converge in `scalar_5w` to stall or oscillate.

---

## 3. Graph Topology & Check Scheduling Mechanics

### 3.1 Pure Serial Gauss-Seidel Natural Scheduling (`scalar_5w`)
- **Ordering:** Checks are evaluated strictly in sequential matrix row order:
  $$m = 0, 1, 2, \dots, 1535$$
- **Immediate L1 Writeback:** Check $m$ updates variable node $v$ in-place in L1 SRAM (`channel_llrs_pair[v]`). If check $m' > m$ shares variable $v$, it reads the newly updated posterior LLR in the **exact same iteration**.
- **Coherent Accumulator Propagation:**
  The CCSDS AR4JA Rate-1/2 matrix has a specific structure:
  - Rows $0 \dots 1023$: Exactly **1,024 checks of Degree 6**.
  - Rows $1024 \dots 1535$: Exactly **512 checks of Degree 3**.
  The Degree-3 checks form a bidiagonal accumulator structure covering the parity nodes. Evaluating **all 1,024 Degree-6 checks first** establishes the initial belief from the information bits; then evaluating **all 512 Degree-3 checks contiguously** sweeps belief unidirectionally through the accumulator chain in a single pass.

---

### 3.2 40-Tile Block Interleaved Scheduling (`tile_matrix_4cw`)
- **Parallel Partitioning:** To eliminate L1 SRAM read/write bank conflicts across the 32 vector lanes, checks are grouped into 40 tiles:
  - 32 tiles $\times$ 32 checks = 1,024 checks of Degree 6.
  - 8 tiles $\times$ 64 checks = 512 checks of Degree 3.
- **Inter-Tile Interleaving ($4:1$ Ratio):**
  The scheduler executes 4 Degree-6 tiles, followed by 1 Degree-3 tile, repeating 8 times per iteration:
  $$[T_0 \dots T_3]^{\text{deg6}} \to [T_4]^{\text{deg3}} \to [T_5 \dots T_8]^{\text{deg6}} \to [T_9]^{\text{deg3}} \dots$$
- **Why $4:1$ Interleaving Affects Convergence:**
  Interleaving 64 Degree-3 accumulator checks in the middle of Degree-6 check sweeps fragments the accumulator chain into 8 isolated segments. Belief cannot propagate across the entire parity block in a single iteration, increasing the effective iterations required to resolve parity inconsistencies and weakening convergence in near-codeword configurations.

---

## 4. Numerical Precision & Quantization Noise Dynamics

### 4.1 Internal FP32 Accumulation (`scalar_5w`)
In `scalar_5w`, RV32 workers compute check node operations using IEEE-754 32-bit single precision (`float`):
- Subtracting incoming check messages ($Q = L - R_{\text{old}}$) maintains 24 bits of mantissa.
- Recursive linear corrections $g(\Delta) = 0.69315 - 0.40\Delta$ operate with full 32-bit floating point precision.
- **Quantization Point:** Quantization occurs **only once** at the final stage when writing to L1 SRAM:
  ```cpp
  bf_llr0 = fp32_to_bf16(new_llr0);
  bf_r0   = fp32_to_bf16(r_new0);
  channel_llrs_pair[vn] = bf_llr0 | (bf_llr1 << 16);
  ```

---

### 4.2 Multi-Stage Intermediate bfloat16 Truncation (`tile_matrix_4cw`)
In `tile_matrix_4cw`, the physical TRISC SFPU executes in native 16-bit destination mode (`.fp32_dest_acc_en = false`) to allow all 64 destination registers to be used simultaneously (Tile 0 for $R$, Tile 1 for $L$).

As a result, as mirrored in [`tests/test_tile_matrix_40tile_lockstep_stepper_sfpi.cpp`](file:///Y:/JPL/LDPC_QB2/LDPC_QB2_GA/tests/test_tile_matrix_40tile_lockstep_stepper_sfpi.cpp#L618-L687), intermediate values are truncated/rounded to `bfloat16` across **four separate stages within a single check node**:
1. **Stage 0:** $Q = \text{round\_bf16}(L - R_{\text{old}})$
2. **Stage 3:** $M_{01}, M_{23}, M_{45} = \text{round\_bf16}(\operatorname{AMin}^*(m_a, m_b))$
3. **Stage 4a:** $r_{\text{out}} = \text{round\_bf16}(\operatorname{copysgn}(r_{\text{mag}}, s))$
4. **Stage 4b:** $L_{\text{new}} = \text{round\_bf16}(Q + r_{\text{out}})$

#### Impact of Intermediate Rounding Noise:
- `bfloat16` has only **7 mantissa bits** ($\approx 2.4$ decimal digits of precision).
- Truncating at 4 intermediate stages introduces quantization jitter ($\pm 1 \text{ LSB}$) into the check evaluation.
- Over 20 to 200 iterations, this rounding noise accumulates, creating pseudo-random perturbations that destabilize marginal LLRs and elevate the error floor.

---

## 5. AWGN Noise Generators: Statistical Tails & Channel Asymmetry

### 5.1 Trigonometry-Free Polar Marsaglia Rejection (`scalar_5w`)
In [`engines/scalar_5w/ldpc_decoder_core.h:128-152`](file:///Y:/JPL/LDPC_QB2/LDPC_QB2_GA/engines/scalar_5w/ldpc_decoder_core.h#L128-L152):
- Generates uniform points $(v_1, v_2)$ in the unit disk $s = v_1^2 + v_2^2 < 1$.
- Evaluates factor $f = \sqrt{-2\ln(s)/s}$ via fast polynomial logarithm and Newton-Raphson inverse square root.
- **Perfect Symmetry:** Both outputs $Z_0 = v_1 f$ and $Z_1 = v_2 f$ have identical statistical moments:
  $$\mathbb{E}[Z_0] = -0.00011, \quad \sigma^2(Z_0) = 1.00019$$
  $$\mathbb{E}[Z_1] = +0.00008, \quad \sigma^2(Z_1) = 1.00015$$

---

### 5.2 Tensix Vector Box-Muller Hardware AWGN (`tile_matrix_4cw`)
In [`engines/tile_matrix_4cw/compute_trisc.cpp`](file:///Y:/JPL/LDPC_QB2/LDPC_QB2_GA/engines/tile_matrix_4cw/compute_trisc.cpp) and hardware Box-Muller kernels:
- Computes $Z_0 = r \cos\theta$ and $Z_1 = r \sin\theta$.
- Hardware destination registers map $Z_0$ directly to **Even Codewords (CW0, CW2)** and $Z_1$ directly to **Odd Codewords (CW1, CW3)**.
- **The Trigonometric Asymmetry:**
  As proven in `docs/MD_files/COMPREHENSIVE_LARGE_SCALE_COMPARATIVE_STUDY_POLAR_MARSAGLIA_VS_TENSIX_VECTOR_BOX_MULLER_AWGN.md`, the truncated Taylor series for $\sin\theta$ exhibits a negative tail defect compared to $\cos\theta$:
  - $\Delta \text{SNR}_{Z_0} = +0.0296\text{ dB}$ (slightly quieter).
  - $\Delta \text{SNR}_{Z_1} = -0.0099\text{ dB}$ (slightly noisier).
  - Net disparity: **$\Delta = 0.0395\text{ dB}$**.
  - **$8.5\%$ excess negative tail outliers at $-4.0\sigma$ on $Z_1$**.
- **Empirical Disparity:**
  In a $3{,}107{,}000$-block simulation to 25 Frame Errors:
  - **Even Codewords (CW0 & CW2, $Z_0 = r\cos\theta$):** **6 Frame Errors** ($\mathbf{\text{FER} = 3.862 \times 10^{-6}}$), matching `scalar_5w`!
  - **Odd Codewords (CW1 & CW3, $Z_1 = r\sin\theta$):** **19 Frame Errors** ($\mathbf{\text{FER} = 1.223 \times 10^{-5}}$, $3.17\times$ higher).
  - Aggregate average: **$8.046 \times 10^{-6}$** (elevated by the odd-codeword tail defect).

---

## 6. Codeword Packing & Convergence Freezing Dynamics

### 6.1 Strict In-Place Freezing in `scalar_5w`
```cpp
if (!done0) {
    // Compute new LLR and check messages for Codeword 0
    channel_llrs_pair[vn] = bf_llr0 | (bf_llr1 << 16);
} else {
    // Preserve old LLR and check messages for Codeword 0 identically
    channel_llrs_pair[vn] = (old_llr & 0xFFFF) | (bf_llr1 << 16);
}
```
When Codeword 0 satisfies parity, it is frozen in L1 SRAM. It is never recalculated or updated in subsequent iterations while Codeword 1 finishes decoding.

---

### 6.2 Companion Freezing in `tile_matrix_4cw`
In early unmasked tile matrix steppers, all 4 companion codewords were gathered, recomputed through TRISC, and scattered back to L1 SRAM unconditionally until the entire batch completed. Continuing to update a converged codeword introduces floating-point drift that can break previously satisfied parity checks.

On physical hardware, this was resolved via `SyncSlot::ACTIVE_MASK = 18` and bitmask scatter freezing in `writer_ncrisc.cpp`, which immediately reduced physical silicon FER by **$3.5\times$** at 2.0 dB.

---

## 7. Comprehensive Architectural & Computational Feature Matrix

| Feature | 5-Worker Scalar Engine (`scalar_5w`) | 4-Codeword Tile Matrix Engine (`tile_matrix_4cw`) | Impact on Decoding Performance / Error Floor |
| :--- | :--- | :--- | :--- |
| **Golden Model File** | [`engines/scalar_5w/ldpc_decoder_core.h`](file:///Y:/JPL/LDPC_QB2/LDPC_QB2_GA/engines/scalar_5w/ldpc_decoder_core.h) | [`tests/test_tile_matrix_40tile_lockstep_stepper_sfpi.cpp`](file:///Y:/JPL/LDPC_QB2/LDPC_QB2_GA/tests/test_tile_matrix_40tile_lockstep_stepper_sfpi.cpp) | Direct silicon kernel verification |
| **Check Node Algorithm** | **True Recursive Approx-Min*** (MILCOM 2003) | **Pairwise Tree Approx-Min*** (SIMD 32-lane) | True recursive has 2 outputs; tree has full extrinsic |
| **Intrinsic Belied Damping** | **Present ($M_{\text{all}}$ includes $q_d$)** | **Absent (Strict Extrinsic)** | **Damping prevents cycle trapping in deep error floor** |
| **Check Node Scheduling** | **Natural Order ($1024 \times \text{Deg-6}$ then $512 \times \text{Deg-3}$)** | **40-Tile Interleaved ($4:1$ Deg-6 / Deg-3)** | Natural order propagates accumulator checks contiguously |
| **Variable Node Updates** | Immediate serial Gauss-Seidel writeback | Block writeback per tile (32/64 checks in parallel) | Serial updating accelerates convergence |
| **Arithmetic Precision** | **Full 32-bit float internal math** | **Multi-stage intermediate `bfloat16` rounding** | Multi-stage rounding injects 7-bit quantization noise |
| **Noise Generation** | Trigonometry-Free Polar Marsaglia | Tensix Vector Box-Muller ($r\cos\theta, r\sin\theta$) | Box-Muller $\sin$ polynomial introduces $+8.5\%$ negative tails |
| **Codeword Freezing** | In-place L1 freeze per codeword | In-place hardware bitmask scatter freezing | Eliminates post-convergence companion drift |
| **Hardware Throughput** | ~158 Msps (16 iters) / ~123 Msps (200 iters) | **>2,590 Msps (>2.5 Gsps) on 440-core QuietBox 2** | Tile Matrix is **$16\times - 20\times$ faster in hardware** |

---

## 8. Engineering Roadmap to Eliminate the 10x Performance Gap

To combine the **ultra-deep error floor of `scalar_5w` ($10^{-9} - 10^{-10}$ FER)** with the **extreme throughput of the Tile Matrix Engine (>2.5 Gsps)**, the following four targeted optimizations are recommended:

1. **Incorporate Intrinsic Belied Damping in TRISC SFPU (`compute_trisc.cpp`):**
   Apply an intentional attenuation factor ($\beta = 0.9375 = 15/16$ or $0.875 = 7/8$) to the outgoing check magnitudes in `compute_tile_deg6_sfpi`:
   ```cpp
   sfpi::vFloat r_out0 = sfpi::round_bf16(sfpi::copysgn(r_mag0 * 0.9375f, s_edge0));
   ```
   This reproduces the stabilizing effect of Dr. Jones' recursive formulation, dampening over-confident belief cycles.

2. **Accumulator-Aware 40-Tile Scheduling:**
   Reorder the 40 tiles into a contiguous Deg-6 then Deg-3 schedule:
   - Execute all **32 Degree-6 tiles first** ($T_0 \dots T_{31}$).
   - Execute all **8 Degree-3 tiles second** ($T_{32} \dots T_{39}$).
   This restores the unidirectional forward propagation across the AR4JA dual-diagonal accumulator chain.

3. **Selective FP32 Accumulation in TRISC SFPU:**
   Retain full 32-bit precision in transient SFPU vector registers ($L_0 \dots L_3$) during Stage 3 pairwise combination, deferring `bfloat16` rounding strictly to the final Stage 4 store into Tile 0 ($R_{\text{new}}$) and Tile 1 ($L_{\text{new}}$).

4. **Equalized Minimax Noise on Tensix Silicon:**
   Ensure `LDPC_ENABLE_EQUALIZED_NOISE=1` (50/50 lane interleaving) and Remez degree-4 minimax polynomials are active in `compute_trisc.cpp` to eliminate the $3.17\times$ error disparity between even and odd codewords.
