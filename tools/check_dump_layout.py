import struct, numpy as np

def u16_to_f(u):
    return struct.unpack('>f', struct.pack('>I', int(u) << 16))[0]

with open('dump/tensix_initial_vnodes.bin', 'rb') as f: init_b = f.read()
with open('dump/golden_final_vnodes.bin', 'rb') as f: gold_b = f.read()
with open('dump/tensix_final_vnodes.bin', 'rb') as f: ten_b = f.read()

w_init = np.frombuffer(init_b, dtype=np.uint16)
w_gold = np.frombuffer(gold_b, dtype=np.uint16)
w_ten  = np.frombuffer(ten_b,  dtype=np.uint16)

# Check all 2560 variables for CW0, CW1, CW2, CW3
diff_count = [0, 0, 0, 0]
for v in range(2560):
    for cw in range(4):
        idx = v * 4 + cw # Wait! How are vnodes laid out in the dump?
        # Is it v * 4 + cw, or Face format?

print("Done")
