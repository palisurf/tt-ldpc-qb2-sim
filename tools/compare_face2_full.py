import numpy as np
import struct

def u2f(u):
    return struct.unpack('>f', struct.pack('>I', int(u) << 16))[0]

with open('/home/ttuser/LDPC_QB2_GA/dump/golden_scratch_in.bin', 'rb') as f:
    gin = np.frombuffer(f.read(), dtype=np.uint16)
with open('/home/ttuser/LDPC_QB2_GA/dump/golden_scratch_out.bin', 'rb') as f:
    gout = np.frombuffer(f.read(), dtype=np.uint16)
with open('/home/ttuser/LDPC_QB2_GA/dump/tensix_scratch_out.bin', 'rb') as f:
    tout = np.frombuffer(f.read(), dtype=np.uint16)

f2_in = gin[2*256:3*256].reshape((16,16))
f2_gout = gout[2*256:3*256].reshape((16,16))
f2_tout = tout[2*256:3*256].reshape((16,16))

print("=== Face 2 Comparison (all 16 cols for each row) ===")
for r in range(12):
    diffs = np.where(f2_gout[r] != f2_tout[r])[0]
    print(f"\n--- Row {r:2d} ({len(diffs)} diffs) ---")
    for c in range(16):
        gi = u2f(f2_in[r, c])
        go = u2f(f2_gout[r, c])
        to = u2f(f2_tout[r, c])
        diff_str = f" DIFF! (to - go = {to - go:+.4f}, to - in = {to - gi:+.4f})" if f2_gout[r, c] != f2_tout[r, c] else " MATCH"
        print(f"  Col {c:2d}: in={gi:+.3f} (0x{f2_in[r,c]:04x}) | gout={go:+.3f} (0x{f2_gout[r,c]:04x}) | tout={to:+.3f} (0x{f2_tout[r,c]:04x}) | {diff_str}")
