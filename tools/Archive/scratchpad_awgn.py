import numpy as np
from collections import defaultdict

# 1. Parse matrix file
filepath = "matrices/AR4JA_r45_4c_128c_r12.chinn.out"
with open(filepath, "r") as f:
    lines = [l.strip().split() for l in f if l.strip() and not l.startswith("#")]

last_line = lines[-1]
M = int(last_line[0])
N = int(last_line[1])

check_to_var = defaultdict(list)
for edge in lines[:-1]:
    r = int(edge[0])
    c = int(edge[1])
    val = float(edge[2])
    if val != 0.0:
        check_to_var[r - 1].append(c - 1)

def simulate_awgn(ebn0_db, P_punctured=512, num_blocks=50, max_iter=16, alpha=0.75):
    rate = (N - M) / (N - P_punctured)
    snr_lin = 10.0 ** (ebn0_db / 10.0)
    mu_llr = 4.0 * rate * snr_lin
    sigma_llr = np.sqrt(8.0 * rate * snr_lin)
    unpunctured_N = N - P_punctured
    
    total_bit_errors = 0
    total_frame_errors = 0
    
    for b in range(num_blocks):
        # Generate AWGN LLRs for all-zeros codeword
        # LLR ~ N(mu_llr, sigma_llr^2)
        llrs = np.zeros(N, dtype=np.float32)
        noise = np.random.normal(0, 1, unpunctured_N)
        llrs[:unpunctured_N] = mu_llr + sigma_llr * noise
        llrs[unpunctured_N:] = 0.0 # Punctured
        
        # Check messages r_msg[m][d]
        r_msg = {m: np.zeros(len(check_to_var[m]), dtype=np.float32) for m in range(M)}
        
        for it in range(max_iter):
            for m in range(M):
                var_list = check_to_var[m]
                deg = len(var_list)
                q_vals = np.empty(deg, dtype=np.float32)
                signs = np.empty(deg, dtype=np.int32)
                abs_qs = np.empty(deg, dtype=np.float32)
                
                for d, vn in enumerate(var_list):
                    q = llrs[vn] - r_msg[m][d]
                    q_vals[d] = q
                    signs[d] = 1 if q < 0 else 0
                    abs_qs[d] = abs(q)
                
                global_sign = 0
                for s in signs:
                    global_sign ^= s
                
                # Min1 and Min2
                min1_idx = np.argmin(abs_qs)
                min1 = abs_qs[min1_idx]
                abs_qs[min1_idx] = 999.0
                min2 = np.min(abs_qs)
                
                # Update
                for d, vn in enumerate(var_list):
                    node_sign = signs[d]
                    msg_sign = global_sign ^ node_sign
                    current_min = min2 if d == min1_idx else min1
                    r_mag = alpha * current_min
                    r_new = -r_mag if msg_sign == 1 else r_mag
                    
                    llrs[vn] = q_vals[d] + r_new
                    r_msg[m][d] = r_new
            
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
            
            if syndrome == 0:
                break
                
        # Count bit errors on unpunctured nodes
        hard_bits = (llrs[:unpunctured_N] < 0).astype(int)
        bit_err = np.sum(hard_bits)
        if bit_err > 0:
            total_bit_errors += bit_err
            total_frame_errors += 1
            
    ber = total_bit_errors / (num_blocks * unpunctured_N)
    fer = total_frame_errors / num_blocks
    print(f"Eb/N0={ebn0_db:.2f} dB, P={P_punctured}: BER={ber:.4e}, FER={fer:.4e} ({total_frame_errors}/{num_blocks} errored blocks)")

print("Testing AWGN with P=512 (last 512 nodes punctured):")
for snr in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
    simulate_awgn(snr, P_punctured=512, num_blocks=50)

print("\nTesting AWGN with P=0 (unpunctured mother code):")
for snr in [1.5, 2.0, 2.5, 3.0]:
    simulate_awgn(snr, P_punctured=0, num_blocks=50)
