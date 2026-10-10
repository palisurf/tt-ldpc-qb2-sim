import struct
import numpy as np

def bf16_to_fp32(val_u16):
    u32 = int(val_u16) << 16
    return struct.unpack('>f', struct.pack('>I', u32))[0]

init_v = np.fromfile("engines/tile_matrix_4cw/dump/tensix_initial_vnodes.bin", dtype=np.uint16)
final_t = np.fromfile("engines/tile_matrix_4cw/dump/tensix_final_vnodes.bin", dtype=np.uint16)
final_g = np.fromfile("dump/golden_lockstep_final_vnodes.bin", dtype=np.uint16)

print("First 8 words of initial vnodes (Var 0: CW0..3, Var 1: CW0..3):")
for i in range(8):
    v = i // 4
    cw = i % 4
    u_init = init_v[i]
    u_t = final_t[i]
    u_g = final_g[i]
    print(f"Var {v}, CW {cw}: init={u_init:#06x} ({bf16_to_fp32(u_init):+.4f}) | Tensix={u_t:#06x} ({bf16_to_fp32(u_t):+.4f}) | Golden={u_g:#06x} ({bf16_to_fp32(u_g):+.4f}) | match={u_t == u_g}")
