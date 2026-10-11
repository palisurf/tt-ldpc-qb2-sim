import struct
import numpy as np

with open('dump/golden_scratch_in.bin', 'rb') as f:
    gold_in = np.frombuffer(f.read(), dtype=np.uint16)
with open('dump/tensix_scratch_in.bin', 'rb') as f:
    tensix_in = np.frombuffer(f.read(), dtype=np.uint16)
with open('dump/golden_scratch_out.bin', 'rb') as f:
    gold_out = np.frombuffer(f.read(), dtype=np.uint16)
with open('dump/tensix_scratch_out.bin', 'rb') as f:
    tensix_out = np.frombuffer(f.read(), dtype=np.uint16)

def bf16_to_float(u):
    return struct.unpack('>f', struct.pack('>HH', u, 0))[0]

print("=== CHECKING SCRATCH IN ===")
for face in range(4):
    eq = np.array_equal(gold_in[face*256:(face+1)*256], tensix_in[face*256:(face+1)*256])
    print(f"Face {face} in match: {eq}")

print("\n=== CHECKING SCRATCH OUT ===")
for face in range(4):
    g_face = gold_out[face*256:(face+1)*256].reshape(16, 16)
    t_face = tensix_out[face*256:(face+1)*256].reshape(16, 16)
    diff_count = np.sum(g_face != t_face)
    print(f"\nFace {face} out diff count: {diff_count} / 256")
    for r in range(16):
        r_diff = np.sum(g_face[r] != t_face[r])
        if r_diff > 0:
            print(f"  Row {r:2d}: {r_diff} diffs")
            for c in range(16):
                if g_face[r, c] != t_face[r, c]:
                    gv = bf16_to_float(g_face[r, c])
                    tv = bf16_to_float(t_face[r, c])
                    iv = bf16_to_float(gold_in[face*256 + r*16 + c])
                    print(f"    col {c:2d}: in={iv:+.4f} gold_out={gv:+.4f} tensix_out={tv:+.4f} diff={tv-gv:+.4f} (hex: gold=0x{g_face[r,c]:04X} tensix=0x{t_face[r,c]:04X})")
