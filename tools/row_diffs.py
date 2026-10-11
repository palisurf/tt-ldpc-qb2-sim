import numpy as np

gold_out = np.frombuffer(open("dump/golden_scratch_out.bin", "rb").read(), dtype=np.uint16)
tensix_out = np.frombuffer(open("dump/tensix_scratch_out.bin", "rb").read(), dtype=np.uint16)
for face in range(4):
    g = gold_out[face*256:(face+1)*256].reshape(16, 16)
    t = tensix_out[face*256:(face+1)*256].reshape(16, 16)
    print(f"Face {face}:")
    for r in range(16):
        d = np.sum(g[r] != t[r])
        print(f"  Row {r:2d}: {d:2d} diffs")
