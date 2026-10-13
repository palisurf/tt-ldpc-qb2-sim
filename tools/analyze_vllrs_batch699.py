import struct
import math

def bf16_to_fp32(bf):
    u = int(bf) << 16
    return struct.unpack('f', struct.pack('I', u))[0]

def get_tile_vnode_byte_offset(v):
    t = v // 256
    v_loc = v % 256
    row = v_loc // 8
    k = v_loc % 8
    face = 0 if row < 16 and k < 4 else (1 if row < 16 else (2 if k < 4 else 3))
    col_grp = k if k < 4 else (k - 4)
    return t * 2048 + (face * 512) + ((row & 15) << 5) + (col_grp << 3)

def parse_vnodes(raw):
    vnodes = [[0.0]*4 for _ in range(2560)]
    for v in range(2560):
        off = get_tile_vnode_byte_offset(v)
        cws = struct.unpack('<4H', raw[off:off+8])
        for cw in range(4):
            vnodes[v][cw] = bf16_to_fp32(cws[cw])
    return vnodes

def main():
    import sys
    base_dir = sys.argv[1] if len(sys.argv) > 1 else '/home/ttuser/LDPC_QB2_GA/dump'
    
    with open(f'{base_dir}/tensix_initial_vnodes.bin', 'rb') as f:
        init_data = f.read()

    with open(f'{base_dir}/tensix_final_vnodes.bin', 'rb') as f:
        final_tensix_data = f.read()

    v_init = parse_vnodes(init_data)
    v_final_tx = parse_vnodes(final_tensix_data)

    print('='*75)
    print('CHUNK BATCH 699, INVOCATION 12: V_LLR MAGNITUDE ANALYSIS')
    print('='*75)

    print('\n[1] INITIAL CHANNEL OBSERVATIONS (t = 0) - Codeword 2 (Trapped Frame)')
    print('    Top 10 Largest Magnitude Values |V_LLR|:')
    cw2_init = [(abs(v_init[v][2]), v_init[v][2], v) for v in range(2048)] # Transmitted
    cw2_init.sort(key=lambda x: x[0], reverse=True)
    for i, (m, val, v) in enumerate(cw2_init[:10]):
        sign = '+' if val >= 0 else '-'
        print(f'    {i+1:2d}. Var {v:4d}: LLR = {sign}{abs(val):8.4f}  (|LLR| = {m:8.4f})')

    print('\n[2] INITIAL CHANNEL OBSERVATIONS (t = 0) - All 4 Codewords')
    print('    Top 10 Largest Magnitude Values Across CW0..CW3:')
    all_init = [(abs(v_init[v][cw]), v_init[v][cw], v, cw) for v in range(2048) for cw in range(4)]
    all_init.sort(key=lambda x: x[0], reverse=True)
    for i, (m, val, v, cw) in enumerate(all_init[:10]):
        sign = '+' if val >= 0 else '-'
        print(f'    {i+1:2d}. CW{cw}, Var {v:4d}: LLR = {sign}{abs(val):8.4f}  (|LLR| = {m:8.4f})')

    print('\n[3] INITIAL CHANNEL BIT ERRORS (Negative LLRs) - Codeword 2')
    print('    Top 10 Strongest Channel Errors (Most Negative LLRs):')
    cw2_neg = [(v_init[v][2], v) for v in range(2048) if v_init[v][2] < 0]
    cw2_neg.sort(key=lambda x: x[0])
    for i, (val, v) in enumerate(cw2_neg[:10]):
        print(f'    {i+1:2d}. Var {v:4d}: LLR = {val:8.4f}  (|LLR| = {abs(val):8.4f})')
    print(f'    Total Raw Channel Bit Errors in CW2: {len(cw2_neg)} / 2048 ({len(cw2_neg)/2048*100:.2f}%)')

    print('\n[4] FINAL DECODED POSTERIOR LLRs on Tensix Silicon (CW2 at 200 Iters)')
    print('    Top 10 Largest Magnitude Transmitted Nodes (v < 2048):')
    cw2_tx_trans = [(abs(v_final_tx[v][2]), v_final_tx[v][2], v) for v in range(2048) if math.isfinite(v_final_tx[v][2])]
    cw2_tx_trans.sort(key=lambda x: x[0], reverse=True)
    for i, (m, val, v) in enumerate(cw2_tx_trans[:10]):
        sign = '+' if val >= 0 else '-'
        print(f'    {i+1:2d}. Var {v:4d}: LLR = {sign}{abs(val):8.4f}  (|LLR| = {m:8.4f})')

    print('\n[5] FINAL DECODED POSTERIOR LLRs on Tensix Silicon (CW2 at 200 Iters)')
    print('    Top 10 Largest Magnitude Punctured Nodes (2048 <= v < 2560):')
    cw2_tx_punct = [(abs(v_final_tx[v][2]), v_final_tx[v][2], v) for v in range(2048, 2560) if math.isfinite(v_final_tx[v][2])]
    cw2_tx_punct.sort(key=lambda x: x[0], reverse=True)
    for i, (m, val, v) in enumerate(cw2_tx_punct[:10]):
        sign = '+' if val >= 0 else '-'
        print(f'    {i+1:2d}. Var {v:4d}: LLR = {sign}{abs(val):8.4f}  (|LLR| = {m:8.4f})')

if __name__ == '__main__':
    main()
