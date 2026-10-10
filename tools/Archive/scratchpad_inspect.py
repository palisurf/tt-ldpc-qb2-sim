import sys
import numpy as np
from collections import Counter, defaultdict

# 1. Parse matrix file
filepath = "matrices/AR4JA_r45_4c_128c_r12.chinn.out"
with open(filepath, "r") as f:
    lines = [l.strip().split() for l in f if l.strip() and not l.startswith("#")]

last_line = lines[-1]
M = int(last_line[0])
N = int(last_line[1])
print(f"Matrix Dimension: M={M}, N={N}")

check_to_var = defaultdict(list)
var_to_check = defaultdict(list)
for edge in lines[:-1]:
    r = int(edge[0])
    c = int(edge[1])
    val = float(edge[2])
    if val != 0.0:
        check_to_var[r - 1].append(c - 1)
        var_to_check[c - 1].append(r - 1)

print(f"Non-zero edges: {sum(len(v) for v in check_to_var.values())}")
print(f"Check node degree range: {min(len(v) for v in check_to_var.values())} .. {max(len(v) for v in check_to_var.values())}")
print(f"Variable node degree range: {min(len(v) for v in var_to_check.values())} .. {max(len(v) for v in var_to_check.values())}")

# 2. Inspect Circulant / Protograph structure (Z = 128)
Z = 128
n_block_rows = M // Z
n_block_cols = N // Z
print(f"\nBlock rows: {n_block_rows}, Block cols: {n_block_cols} (Z = {Z})")

proto = np.zeros((n_block_rows, n_block_cols), dtype=int)
for m, var_list in check_to_var.items():
    br = m // Z
    for vn in var_list:
        bc = vn // Z
        proto[br, bc] += 1

proto_weights = (proto // Z).astype(int)
print("\nProtograph (block degree / weights):")
for r in range(n_block_rows):
    row_str = " ".join(f"{proto[r, c] // Z:2d}" for c in range(n_block_cols))
    print(f"Row {r:2d}: {row_str}")

# Variable degrees in protograph
proto_col_degs = [sum(proto[r, c] // Z for r in range(n_block_rows)) for c in range(n_block_cols)]
print("\nProtograph column degrees (each col is 128 var nodes):")
print("Col:   " + " ".join(f"{c:2d}" for c in range(n_block_cols)))
print("Deg:   " + " ".join(f"{d:2d}" for d in proto_col_degs))

# 3. Simulate C++ Layered Min-Sum decoder
def simulate_layered_min_sum(initial_llrs, max_iter=16, alpha=0.75):
    llrs = np.copy(initial_llrs).astype(np.float32)
    # Check messages r_msg[m][vn]
    r_msg = {m: {vn: 0.0 for vn in check_to_var[m]} for m in range(M)}
    
    for it in range(max_iter):
        for m in range(M):
            var_list = check_to_var[m]
            deg = len(var_list)
            q_vals = []
            signs = []
            abs_qs = []
            
            for vn in var_list:
                q = llrs[vn] - r_msg[m][vn]
                q_vals.append(q)
                signs.append(1 if q < 0 else 0)
                abs_qs.append(abs(q))
            
            global_sign = 0
            for s in signs:
                global_sign ^= s
            
            # Find min1, min2
            sorted_indices = np.argsort(abs_qs)
            min1_idx = sorted_indices[0]
            min1 = abs_qs[min1_idx]
            min2 = abs_qs[sorted_indices[1]] if deg > 1 else min1
            
            # Update
            for d, vn in enumerate(var_list):
                node_sign = signs[d]
                msg_sign = global_sign ^ node_sign
                current_min = min2 if d == min1_idx else min1
                r_mag = alpha * current_min
                r_new = -r_mag if msg_sign == 1 else r_mag
                
                llrs[vn] = q_vals[d] + r_new
                r_msg[m][vn] = r_new
        
        # Syndrome check
        syndrome = 0
        for m in range(M):
            row_parity = 0
            for vn in check_to_var[m]:
                if llrs[vn] < 0:
                    row_parity ^= 1
            if row_parity != 0:
                syndrome = 1
                break
        
        hard_bits = (llrs < 0).astype(int)
        bit_errors = np.sum(hard_bits[:N - 512]) # Unpunctured
        if syndrome == 0:
            return it + 1, bit_errors, True
            
    hard_bits = (llrs < 0).astype(int)
    bit_errors = np.sum(hard_bits[:N - 512])
    return max_iter, bit_errors, (syndrome == 0)

print("\n--- TEST 1: All +5.0 LLRs (Perfect Channel, NO puncturing) ---")
init_llrs = np.full(N, 5.0, dtype=np.float32)
iters, errs, syn = simulate_layered_min_sum(init_llrs)
print(f"Result: {iters} iters, bit errors: {errs}, syndrome valid: {syn}")

print("\n--- TEST 2: +5.0 LLRs on unpunctured, 0.0 on LAST 512 nodes (cols 2048..2559) ---")
init_llrs = np.full(N, 5.0, dtype=np.float32)
init_llrs[2048:2560] = 0.0 # Puncture last 512
iters, errs, syn = simulate_layered_min_sum(init_llrs)
print(f"Result: {iters} iters, bit errors: {errs}, syndrome valid: {syn}")

print("\n--- TEST 3: +5.0 LLRs on unpunctured, 0.0 on other blocks of 512 nodes ---")
for b in range(0, N, 512):
    init_llrs = np.full(N, 5.0, dtype=np.float32)
    init_llrs[b:b+512] = 0.0
    iters, errs, syn = simulate_layered_min_sum(init_llrs)
    print(f"Puncture [{b}:{b+512}] (Cols {b//128}..{(b+512)//128-1}): {iters} iters, bit errors: {errs}, syndrome valid: {syn}")
