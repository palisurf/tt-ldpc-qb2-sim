import struct
import numpy as np

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

def to_bf16(f):
    return bf_to_f(f_to_bf(f))

# Let's write a complete 40-tile stepper in Python using Model B (vFloat 32-bit register semantics)
# and compare directly against w_final (Tensix output in ttsim)!
print("Evaluating Model B across all 40 tiles...")
