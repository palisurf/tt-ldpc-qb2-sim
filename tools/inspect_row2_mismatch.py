import numpy as np
import struct

def u2f(u):
    return struct.unpack('>f', struct.pack('>I', int(u) << 16))[0]

g_in = np.fromfile('/home/ttuser/LDPC_QB2_GA/dump/golden_scratch_in.bin', dtype=np.uint16)
g_out = np.fromfile('/home/ttuser/LDPC_QB2_GA/dump/golden_scratch_out.bin', dtype=np.uint16)
t_out = np.fromfile('/home/ttuser/LDPC_QB2_GA/dump/tensix_scratch_out.bin', dtype=np.uint16)

f0_in = g_in[:256].reshape((16, 16))
f0_gout = g_out[:256].reshape((16, 16))
f0_tout = t_out[:256].reshape((16, 16))

print("Face 0 Inputs for Quartet 0 (Rows 0..3, Col 0):")
print(f"  Row 0, Col 0: {u2f(f0_in[0, 0]):+.3f} (0x{f0_in[0, 0]:04x})")
print(f"  Row 1, Col 0: {u2f(f0_in[1, 0]):+.3f} (0x{f0_in[1, 0]:04x})")
print(f"  Row 2, Col 0: {u2f(f0_in[2, 0]):+.3f} (0x{f0_in[2, 0]:04x})")
print(f"  Row 3, Col 0: {u2f(f0_in[3, 0]):+.3f} (0x{f0_in[3, 0]:04x})")

print("\nFace 0 Outputs for Row 0, Col 0:")
print(f"  Golden: {u2f(f0_gout[0, 0]):+.3f}")
print(f"  Tensix: {u2f(f0_tout[0, 0]):+.3f}")

print("\nFace 0 Outputs for Row 2, Col 0:")
print(f"  Golden: {u2f(f0_gout[2, 0]):+.3f}")
print(f"  Tensix: {u2f(f0_tout[2, 0]):+.3f}")
