# QuietBox 2 Tile Matrix 4-Codeword LDPC Decoder Engine (`tile_matrix_4cw`)

## Architectural Summary
The `tile_matrix_4cw` engine implements the **Decoupled Access-Execute (DAE)** paradigm on Tenstorrent QuietBox 2 (4 Blackhole ASICs, 440 Tensix cores, 2,200 RISC-V processors). 

Instead of treating variable updates as sequential scalar operations, this engine packs **4 parallel codewords** across **40 check nodes** into native **$32 \times 32$ Tensix systolic tiles** (960 active elements).

### Core Components
| File | Processor Target | Role |
| :--- | :--- | :--- |
| [`ldpc_tile_matrix_core.h`](file:///y:/JPL/LDPC_QB2/LDPC_QB2_GA/engines/tile_matrix_4cw/ldpc_tile_matrix_core.h) | Shared | Circular Buffer mappings (CB0–CB16), hardware semaphore IDs, tile structures, packed 4-codeword types, and mathematical constants. |
| [`reader_brisc.cpp`](file:///y:/JPL/LDPC_QB2/LDPC_QB2_GA/engines/tile_matrix_4cw/reader_brisc.cpp) | BRISC (DM0) | **Stage 1 (Gather 0)**: Gathers edges $\{0, 1, 2\}$ for 40 checks from active L1 variable buffer into double-buffered ping-pong scratchpads (`CB 3` / `CB 4`). Swaps active pointer at batch boundaries. |
| [`writer_ncrisc.cpp`](file:///y:/JPL/LDPC_QB2/LDPC_QB2_GA/engines/tile_matrix_4cw/writer_ncrisc.cpp) | NCRISC (DM1) | **Stage 3 (Scatter 0)**: Scatters updated variable node posteriors (edges $\{0, 1, 2\}$) from scratchpads (`CB 7` / `CB 8`) back to L1 SRAM. Streams telemetry packets to host DRAM via NoC. |
| [`compute_trisc.cpp`](file:///y:/JPL/LDPC_QB2/LDPC_QB2_GA/engines/tile_matrix_4cw/compute_trisc.cpp) | TRISC0, TRISC1, TRISC2 | **Stage 1–3 Compute Engine**: <br>• **TRISC0 (UNPACK)**: Gathers edges $\{3, 4, 5\}$ and triggers hardware DMA `copy_tile` for $R_{\mathrm{old}}$.<br>• **TRISC1 (MATH)**: Executes Matrix FPU $Q = L - R$, SFPU Approximate-Min* splines, and $L_{\mathrm{new}} = Q + R'$; pre-generates Cartesian Box-Muller AWGN during $687\ \mathrm{idle\ cycles/tile}$.<br>• **TRISC2 (PACK)**: Triggers hardware DMA `pack_tile` for $R_{\mathrm{new}}$, scatters edges $\{3, 4, 5\}$, and tracks parity syndrome violations. |
| [`ldpc_sim_tile_matrix.cpp`](file:///y:/JPL/LDPC_QB2/LDPC_QB2_GA/engines/tile_matrix_4cw/ldpc_sim_tile_matrix.cpp) | Host / QB2 | Host harness: Parses `.chinn.out`, executes balanced graph partitioning into 33 tiles, and performs the mathematical pre-flight collision audit. |

---

### Verified Pre-Flight Collision Audit
```
=======================================================
 PRE-FLIGHT COLLISION AUDIT: MATHEMATICAL VERIFICATION
=======================================================
[AUDIT RESULT] Scheduled Tiles: 33 (26 Deg-6 + 7 Deg-3)
               Covered Edges:   7680 / 7680 (100.0% Coverage)
               Collisions:      IDENTICALLY ZERO (100% COLLISION-FREE)
=======================================================
```
- **Degree-6 Checks ($M_6 = 1,024$)**: 26 tiles (39–40 checks/tile).
- **Degree-3 Checks ($M_3 = 512$)**: 7 tiles (73–74 checks/tile).
- **Total Tiles per Iteration**: Exactly **33 tiles** with **100% collision-free guarantee**.
