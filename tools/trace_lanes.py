import struct
import numpy as np

def to_f(u):
    return struct.unpack('>f', struct.pack('>HH', u, 0))[0]

def fp32_to_bf16(f):
    u = struct.unpack('>I', struct.pack('>f', f))[0]
    return u >> 16

def get_scratch_byte_offset(reg_idx, lane):
    f = reg_idx // 8
    r_face = reg_idx % 8
    row_in_face = (2 * r_face) if lane < 16 else (2 * r_face + 1)
    col_in_face = lane if lane < 16 else (lane - 16)
    return f * 512 + row_in_face * 32 + col_in_face * 2

sin = np.frombuffer(open("dump/tensix_scratch_in.bin", "rb").read(), dtype=np.uint16)
sout = np.frombuffer(open("dump/tensix_scratch_out.bin", "rb").read(), dtype=np.uint16)

# Let's inspect pass 0 (Face 0), lane 0:
# base_r = 0, deg = 6
print("=== Face 0, Lane 0 (reg 0..5, lane 0) ===")
for r in range(6):
    off = get_scratch_byte_offset(0 + r, 0)
    w_idx = off // 2
    val = to_f(sin[w_idx])
    out_val = to_f(sout[w_idx])
    print(f"r={r}: off={off:4d} (word={w_idx:4d}) val={val:+.4f} (hex=0x{sin[w_idx]:04X}) out={out_val:+.4f} (hex=0x{sout[w_idx]:04X})")

print("\n=== Face 2, Lane 0 (reg 16..21, lane 0) ===")
for r in range(6):
    off = get_scratch_byte_offset(16 + r, 0)
    w_idx = off // 2
    val = to_f(sin[w_idx])
    out_val = to_f(sout[w_idx])
    print(f"r={r}: off={off:4d} (word={w_idx:4d}) val={val:+.4f} (hex=0x{sin[w_idx]:04X}) out={out_val:+.4f} (hex=0x{sout[w_idx]:04X})")
