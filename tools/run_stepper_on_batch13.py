import numpy as np
import struct
import subprocess

# Let's import generate_tile_channel and export to dump/batch13_vnodes.bin
from inspect_batch13 import generate_tile_channel

# Build the 10 tiles of bfloat16 (20480 bytes)
# In Tensix format, each tile has 1024 bfloat16 values (2048 bytes).
# Wait, pack_tile packs dst_reg into the tile.
# Let's convert float32 to bfloat16 uint16
def float_to_bf16(f):
    u32 = struct.unpack('<I', struct.pack('<f', f))[0]
    # RNE
    lsb = (u32 >> 16) & 1
    bias = 0x7FFF + lsb
    return (u32 + bias) >> 16

tile_bytes = bytearray()
for t in range(10):
    if t < 8:
        regs = generate_tile_channel(13, t)
        # pack_tile packs dst_reg[0..31] into face layout
        # Face 0: regs 0..7
        # Face 1: regs 8..15
        # Face 2: regs 16..23
        # Face 3: regs 24..31
        # In face f, quartet k has 4 rows (r_in_face = 4*k + lane//8), and col = 2*(lane%8) + is_odd_reg
        # Face offset: face * 512 + r_in_face * 32 + col * 2
        tile_buf = [0] * 1024
        for f in range(4):
            base_reg = f * 8
            for k in range(4):
                for sub in range(2):
                    reg_idx = base_reg + 2 * k + sub
                    for lane in range(32):
                        r_in_face = 4 * k + (lane // 8)
                        col_in_face = 2 * (lane % 8) + sub
                        word_idx = (f * 512 + r_in_face * 32 + col_in_face * 2) // 2
                        tile_buf[word_idx] = float_to_bf16(regs[reg_idx, lane])
        for w in tile_buf:
            tile_bytes += struct.pack('<H', w)
    else:
        tile_bytes += b'\x00' * 2048

with open('/home/ttuser/LDPC_QB2_GA/dump/batch13_initial.bin', 'wb') as f:
    f.write(tile_bytes)
print(f'Exported /home/ttuser/LDPC_QB2_GA/dump/batch13_initial.bin: {len(tile_bytes)} bytes')
