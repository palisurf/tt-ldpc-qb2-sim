import numpy as np
import os

candidates = [
    "/home/ttuser/LDPC_QB2_GA/engines/tile_matrix_4cw/dump",
    "engines/tile_matrix_4cw/dump",
    "dump"
]
base = "."
for c in candidates:
    if os.path.exists(os.path.join(c, "tensix_scratch_in.bin")):
        base = c
        break

init_vn = np.fromfile(f"{base}/tensix_initial_vnodes.bin", dtype=np.uint16)
final_vn = np.fromfile(f"{base}/tensix_final_vnodes.bin", dtype=np.uint16)
scat_in = np.fromfile(f"{base}/tensix_scratch_in.bin", dtype=np.uint16)
scat_out = np.fromfile(f"{base}/tensix_scratch_out.bin", dtype=np.uint16)

print(f"init_vn len: {len(init_vn)}, non-zero: {np.count_nonzero(init_vn)}")
print(f"final_vn len: {len(final_vn)}, non-zero: {np.count_nonzero(final_vn)}")
print(f"scat_in len: {len(scat_in)}, non-zero: {np.count_nonzero(scat_in)}")
print(f"scat_out len: {len(scat_out)}, non-zero: {np.count_nonzero(scat_out)}")

diff_scat = np.where(scat_in != scat_out)[0]
print(f"\nTotal diffs between scat_in and scat_out: {len(diff_scat)}")
for idx in diff_scat:
    byte_off = idx * 2
    face = byte_off // 512
    rem = byte_off % 512
    row = rem // 32
    col = (rem % 32) // 2
    print(f"word_idx={idx:4d} (byte 0x{byte_off:03x}): Face {face}, Row {row:2d}, Col {col:2d} | in=0x{scat_in[idx]:04x} ({scat_in[idx]}), out=0x{scat_out[idx]:04x} ({scat_out[idx]})")

diff_vn = np.where(init_vn != final_vn)[0]
print(f"\nTotal diffs between init_vn and final_vn: {len(diff_vn)} / {len(init_vn)}")
zeros_in_final = np.count_nonzero(final_vn[diff_vn] == 0)
non_zeros_in_final = np.count_nonzero(final_vn[diff_vn] != 0)
print(f"Diffs where final_vn == 0: {zeros_in_final} / {len(diff_vn)}")
print(f"Diffs where final_vn != 0: {non_zeros_in_final} / {len(diff_vn)}")

# Group diffs by Vnode
bad_vnodes = set()
for idx in diff_vn:
    v = idx // 4
    bad_vnodes.add(v)
print(f"Unique corrupted Vnodes: {len(bad_vnodes)} / 2560")
for v in sorted(list(bad_vnodes))[:20]:
    vals_init = [init_vn[4 * v + c] for c in range(4)]
    vals_final = [final_vn[4 * v + c] for c in range(4)]
    print(f"  Vnode {v:4d}: init={vals_init}, final={vals_final}")
