import struct
import sys

def bf16_to_fp32(bf):
    u = int(bf) << 16
    return struct.unpack('f', struct.pack('I', u))[0]

def fp32_to_bf16(f):
    u = struct.unpack('I', struct.pack('f', f))[0]
    lsb = (u >> 16) & 1
    bias = 0x7FFF + lsb
    u = (u + bias) & 0xFFFFFFFF
    return (u >> 16) & 0xFFFF

def to_bf16(f):
    return bf16_to_fp32(fp32_to_bf16(f))

# Let's see what happens on the trapped frame with different clipping values!
# We can load tensix_initial_vnodes.bin and run decode_tile_matrix_batch logic directly or use test_40tile_stepper.
print("Clipping sweep analysis tool ready.")
