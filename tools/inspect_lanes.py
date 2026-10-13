import struct, numpy as np

def bf(u):
    return struct.unpack('>f', struct.pack('>I', int(u) << 16))[0]

gin = np.fromfile('dump/golden_scratch_in.bin', dtype=np.uint16)
gout = np.fromfile('dump/golden_scratch_out.bin', dtype=np.uint16)
tout = np.fromfile('dump/tensix_scratch_out.bin', dtype=np.uint16)

def get_scratch_byte_offset(reg_idx, lane):
    face = reg_idx // 8
    r_in_face = reg_idx % 8
    k = r_in_face // 2
    is_odd_reg = r_in_face % 2
    row_in_quartet = lane // 8
    c_pair = lane % 8
    return face * 512 + (4 * k + row_in_quartet) * 32 + (2 * c_pair + is_odd_reg) * 2

print('=== Reg 0 (Face 0, Pair 0 Even): All 32 Lanes ===')
print('Lane | RowInQuartet | ColPair | Reg 0 In   | Reg 0 Gold | Reg 0 Tsx  | Diff')
print('-'*70)
for l in range(32):
    off = get_scratch_byte_offset(0, l) // 2
    v_in = bf(gin[off])
    v_gold = bf(gout[off])
    v_tsx = bf(tout[off])
    d = abs(v_gold - v_tsx)
    rq = l // 8
    cp = l % 8
    flag = " MISMATCH" if d > 0.05 else " MATCH"
    print(f'{l:4d} | {rq:12d} | {cp:7d} | {v_in:+10.4f} | {v_gold:+10.4f} | {v_tsx:+10.4f} | {d:7.4f} {flag}')
