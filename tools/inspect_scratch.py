import struct
import numpy as np

def bf(u):
    return struct.unpack('>f', struct.pack('>I', int(u) << 16))[0]

gin = np.fromfile('dump/golden_scratch_in.bin', dtype=np.uint16)
gout = np.fromfile('dump/golden_scratch_out.bin', dtype=np.uint16)
tout = np.fromfile('dump/tensix_scratch_out.bin', dtype=np.uint16)

print("\n=== Face 0 Full Matrix Comparison (Rows 0..11) ===")
f_gin = gin[:256].reshape((16, 16))
f_gout = gout[:256].reshape((16, 16))
f_tout = tout[:256].reshape((16, 16))
for r in range(12):
    g_row = [round(bf(f_gout[r, c]), 3) for c in range(8)]
    t_row = [round(bf(f_tout[r, c]), 3) for c in range(8)]
    diffs = [round(abs(bf(f_gout[r, c]) - bf(f_tout[r, c])), 3) for c in range(16)]
    print(f"Row {r:2d} (max diff={max(diffs):.3f}):")
    print(f"  G: {g_row}")
    print(f"  T: {t_row}")
    if max(diffs) > 0.05:
        print(f"  D: {diffs[:8]}")


print("\n=== Col 0 (Check 0 Even: rows 0,2,4,6,8,10 | Check 4 Odd: rows 1,3,5,7,9,11) ===")
print("Row | Type | Edge | Input LLR | Golden Out | Tensix Out | Diff")
print("-" * 65)
for r in range(12):
    chk_type = "Even (C0)" if (r % 2 == 0) else "Odd  (C4)"
    edge = r // 2
    inp = bf(f_gin[r, 0])
    gold = bf(f_gout[r, 0])
    tsx = bf(f_tout[r, 0])
    diff = abs(gold - tsx)
    print(f"{r:2d}  | {chk_type} |  E{edge}  | {inp:+9.4f} | {gold:+10.4f} | {tsx:+10.4f} | {diff:.4f}")


print("\n=== Face 0 Probe (Rows 12..15) ===")
f0_tout = tout[:256].reshape((16, 16))
for r in [12, 13, 14, 15]:
    r_mag = [round(bf(f0_tout[r, 2*c]), 4) for c in range(8)]
    m_swap = [round(bf(f0_tout[r, 2*c + 1]), 4) for c in range(8)]
    print(f"Row {r:2d} | r_mag (Even Cols): {r_mag}")
    print(f"       | m_swap (Odd Cols): {m_swap}")




