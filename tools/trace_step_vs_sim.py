import struct
import numpy as np

b_step = open("tests/dump/golden_lockstep_final_vnodes.bin", "rb").read()[:20480]
b_sim  = open("engines/tile_matrix_4cw/dump/golden_final_vnodes.bin", "rb").read()[:20480]

w_step = np.frombuffer(b_step, dtype=np.uint16)
w_sim  = np.frombuffer(b_sim, dtype=np.uint16)

diffs = np.where(w_step != w_sim)[0]
print(f"Total differing words between stepper and sim: {len(diffs)} / {len(w_step)}")

for idx in diffs[:15]:
    # Byte offset
    byte_off = idx * 2
    tile_idx = byte_off // 2048
    byte_in_tile = byte_off % 2048
    val_step = w_step[idx]
    val_sim = w_sim[idx]
    def bf_to_f(u):
        return struct.unpack('>f', struct.pack('>HH', u, 0))[0]
    print(f"Word {idx} (Byte 0x{byte_off:04X}, Tile {tile_idx}, byte {byte_in_tile}): Step=0x{val_step:04X} ({bf_to_f(val_step):.5f}) vs Sim=0x{val_sim:04X} ({bf_to_f(val_sim):.5f})")
