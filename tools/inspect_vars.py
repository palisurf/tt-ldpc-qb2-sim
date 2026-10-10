import struct
import numpy as np

# Load initial and final dumps
f_init  = open("engines/tile_matrix_4cw/dump/tensix_initial_vnodes.bin", "rb").read()[:20480]
f_final = open("engines/tile_matrix_4cw/dump/tensix_final_vnodes.bin", "rb").read()[:20480]
f_gold  = open("engines/tile_matrix_4cw/dump/golden_final_vnodes.bin", "rb").read()[:20480]

w_init  = np.frombuffer(f_init, dtype=np.uint16)
w_final = np.frombuffer(f_final, dtype=np.uint16)
w_gold  = np.frombuffer(f_gold, dtype=np.uint16)

def bf_to_f(u):
    return struct.unpack('>f', struct.pack('>HH', int(u), 0))[0]

def f_to_bf(f):
    u = struct.unpack('>I', struct.pack('>f', f))[0]
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16) & 0xFFFF

# Let's inspect Variable 0 (offsets 0, 2, 4, 6)
print("Variable 0:")
for c in range(4):
    init_u = w_init[c]
    gold_u = w_gold[c]
    tens_u = w_final[c]
    print(f"  CW{c}: Init=0x{init_u:04X} ({bf_to_f(init_u):.5f}) | Gold=0x{gold_u:04X} ({bf_to_f(gold_u):.5f}) | Tensix=0x{tens_u:04X} ({bf_to_f(tens_u):.5f}) | diff={int(gold_u)-int(tens_u)}")

# Let's inspect Variable 1 (offsets 4..7 in uint16, bytes 8..15)
print("\nVariable 1:")
for c in range(4):
    init_u = w_init[4 + c]
    gold_u = w_gold[4 + c]
    tens_u = w_final[4 + c]
    print(f"  CW{c}: Init=0x{init_u:04X} ({bf_to_f(init_u):.5f}) | Gold=0x{gold_u:04X} ({bf_to_f(gold_u):.5f}) | Tensix=0x{tens_u:04X} ({bf_to_f(tens_u):.5f}) | diff={int(gold_u)-int(tens_u)}")
