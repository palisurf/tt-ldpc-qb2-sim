import numpy as np

gin = np.fromfile('dump/golden_scratch_out.bin', dtype=np.uint16)
tin = np.fromfile('dump/tensix_scratch_out.bin', dtype=np.uint16)
diff = gin.astype(int) - tin.astype(int)
nz = diff[diff != 0]

print(f"Total diffs: {len(nz)} / {len(gin)}")
print(f"1 LSB diffs: {np.sum(np.abs(nz) == 1)}")
print(f"2 LSB diffs: {np.sum(np.abs(nz) == 2)}")
print(f"3-4 LSB diffs: {np.sum((np.abs(nz) >= 3) & (np.abs(nz) <= 4))}")
print(f">4 LSB diffs: {np.sum(np.abs(nz) > 4)}")

# Inspect the values of >4 LSB diffs
if np.sum(np.abs(nz) > 4) > 0:
    big_idx = np.where(np.abs(diff) > 4)[0]
    print(f"\nSample big diffs (first 10):")
    for idx in big_idx[:10]:
        face = idx // 256
        rem = idx % 256
        row = rem // 16
        col = rem % 16
        g_val = gin[idx]
        t_val = tin[idx]
        print(f"  Face {face}, Row {row:2d}, Col {col:2d}: G=0x{g_val:04X}, T=0x{t_val:04X}, diff={diff[idx]}")
