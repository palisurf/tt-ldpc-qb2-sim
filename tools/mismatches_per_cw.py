import struct, numpy as np

def u16_to_f(u):
    return struct.unpack('>f', struct.pack('>I', int(u) << 16))[0]

with open('dump/golden_final_vnodes.bin', 'rb') as f: gold_b = f.read()
with open('dump/tensix_final_vnodes.bin', 'rb') as f: ten_b = f.read()

w_gold = np.frombuffer(gold_b, dtype=np.uint16)
w_ten  = np.frombuffer(ten_b,  dtype=np.uint16)

mismatches_per_cw = [0, 0, 0, 0]
total_per_cw = [0, 0, 0, 0]

for idx in range(len(w_gold)):
    byte_off = idx * 2
    cw = (byte_off % 8) // 2
    total_per_cw[cw] += 1
    if w_gold[idx] != w_ten[idx]:
        mismatches_per_cw[cw] += 1

for cw in range(4):
    print(f"CW{cw}: {mismatches_per_cw[cw]} / {total_per_cw[cw]} mismatches ({mismatches_per_cw[cw]/total_per_cw[cw]*100:.2f}%)")
