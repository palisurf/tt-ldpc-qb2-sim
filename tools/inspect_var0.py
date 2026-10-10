import struct, numpy as np

def u16_to_f(u):
    return struct.unpack('>f', struct.pack('>I', int(u) << 16))[0]

with open('dump/tensix_initial_vnodes.bin', 'rb') as f: init_b = f.read()
with open('dump/golden_final_vnodes.bin', 'rb') as f: gold_b = f.read()
with open('dump/tensix_final_vnodes.bin', 'rb') as f: ten_b = f.read()

w_init = np.frombuffer(init_b, dtype=np.uint16)
w_gold = np.frombuffer(gold_b, dtype=np.uint16)
w_ten  = np.frombuffer(ten_b,  dtype=np.uint16)

print("--- Variable 0 Across Codewords ---")
for cw in range(4):
    idx = cw
    init_v = u16_to_f(w_init[idx])
    gold_v = u16_to_f(w_gold[idx])
    ten_v  = u16_to_f(w_ten[idx])
    print(f"CW{cw}: Initial={init_v:+.5f} (0x{w_init[idx]:04X}) | Golden={gold_v:+.5f} (0x{w_gold[idx]:04X}) | Tensix={ten_v:+.5f} (0x{w_ten[idx]:04X}) | Diff={gold_v-ten_v:+.5f}")
    print(f"     Delta Golden: {gold_v - init_v:+.5f} | Delta Tensix: {ten_v - init_v:+.5f}")

print("\n--- Variable 1 Across Codewords ---")
for cw in range(4):
    idx = 4 + cw # v=1
    init_v = u16_to_f(w_init[idx])
    gold_v = u16_to_f(w_gold[idx])
    ten_v  = u16_to_f(w_ten[idx])
    print(f"CW{cw}: Initial={init_v:+.5f} (0x{w_init[idx]:04X}) | Golden={gold_v:+.5f} (0x{w_gold[idx]:04X}) | Tensix={ten_v:+.5f} (0x{w_ten[idx]:04X}) | Diff={gold_v-ten_v:+.5f}")
    print(f"     Delta Golden: {gold_v - init_v:+.5f} | Delta Tensix: {ten_v - init_v:+.5f}")
