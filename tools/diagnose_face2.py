import struct
import numpy as np

def bf16_to_fp32(val_u16):
    u32 = int(val_u16) << 16
    return struct.unpack('>f', struct.pack('>I', u32))[0]

g_in = np.fromfile('/home/ttuser/LDPC_QB2_GA/engines/tile_matrix_4cw/dump/golden_scratch_in.bin', dtype=np.uint16)
t_in = np.fromfile('/home/ttuser/LDPC_QB2_GA/engines/tile_matrix_4cw/dump/tensix_scratch_in.bin', dtype=np.uint16)
g_out = np.fromfile('/home/ttuser/LDPC_QB2_GA/engines/tile_matrix_4cw/dump/golden_scratch_out.bin', dtype=np.uint16)
t_out = np.fromfile('/home/ttuser/LDPC_QB2_GA/engines/tile_matrix_4cw/dump/tensix_scratch_out.bin', dtype=np.uint16)

print("Face 2 inputs (first 16 elements):")
for i in range(16):
    idx = 512 + i
    print(f"[{i:2d}] g_in: {bf16_to_fp32(g_in[idx]):+.4f}, t_in: {bf16_to_fp32(t_in[idx]):+.4f}, g_out: {bf16_to_fp32(g_out[idx]):+.4f}, t_out: {bf16_to_fp32(t_out[idx]):+.4f}")

# Let's inspect where in Face 2 (indices 512..767) mismatches occur:
m_f2 = [i for i in range(256) if g_out[512 + i] != t_out[512 + i]]
print(f"Face 2 has {len(m_f2)} mismatches out of 256.")
for i in m_f2[:10]:
    row_in_f = i // 16
    col_in_f = i % 16
    g_val = bf16_to_fp32(g_out[512 + i])
    t_val = bf16_to_fp32(t_out[512 + i])
    in_val = bf16_to_fp32(t_in[512 + i])
    print(f"  F2 row {row_in_f:2d}, col {col_in_f:2d} (elem {i:3d}): in={in_val:+.4f}, g_out={g_val:+.4f}, t_out={t_val:+.4f}, diff={abs(g_val - t_val):.4f}")
