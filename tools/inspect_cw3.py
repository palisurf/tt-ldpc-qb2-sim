import struct

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

raw_init = open('/home/ttuser/LDPC_QB2_GA/dump/tensix_initial_vnodes.bin', 'rb').read()
raw_final = open('/home/ttuser/LDPC_QB2_GA/dump/tensix_final_vnodes.bin', 'rb').read()

cw3_init = []
cw3_final = []
for v in range(2560):
    off = get_tile_vnode_byte_offset(v)
    cws_i = struct.unpack('<4H', raw_init[off:off+8])
    cws_f = struct.unpack('<4H', raw_final[off:off+8])
    cw3_init.append((bf16_to_fp32(cws_i[3]), v))
    cw3_final.append((bf16_to_fp32(cws_f[3]), v))

print('=== CW3 INITIAL CHANNEL LLRs ===')
neg_init = [x for x in cw3_init[:2048] if x[0] < 0]
neg_init.sort(key=lambda x: x[0])
print(f'Total negative initial LLRs in CW3: {len(neg_init)} / 2048 ({len(neg_init)/2048*100:.2f}%)')
print('Top 10 most negative channel errors in CW3:')
for val, v in neg_init[:10]:
    print(f'  Var {v:4d}: LLR = {val:8.4f}')

pos_init = [x for x in cw3_init[:2048] if x[0] >= 0]
pos_init.sort(key=lambda x: x[0], reverse=True)
print('\nTop 10 most positive channel observations in CW3:')
for val, v in pos_init[:10]:
    print(f'  Var {v:4d}: LLR = {val:8.4f}')

print('\n=== CW3 FINAL DECODED LLRs ON TENSIX SILICON ===')
cw3_final_tx = [(abs(val), val, v) for val, v in cw3_final]
cw3_final_tx.sort(key=lambda x: x[0], reverse=True)
print('Top 10 largest magnitude LLRs in CW3:')
for m, val, v in cw3_final_tx[:10]:
    p = ' (Punctured)' if v >= 2048 else ''
    print(f'  Var {v:4d}: LLR = {val:10.4f} (|LLR| = {m:10.4f}){p}')
