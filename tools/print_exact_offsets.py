def offset_in_face(r, lane):
    row = 4 * (r // 2) + lane // 8
    col = 2 * (lane % 8) + (r % 2)
    return row * 32 + col * 2

# For r=0 (Edge 0):
print("Register r=0 (Edge 0) byte offsets for lanes 0..31:")
offs_r0 = [offset_in_face(0, l) for l in range(32)]
print(offs_r0)

print("\nRegister r=1 (Edge 1) byte offsets for lanes 0..31:")
offs_r1 = [offset_in_face(1, l) for l in range(32)]
print(offs_r1)
