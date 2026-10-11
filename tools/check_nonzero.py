with open("dump/tensix_scratch_in.bin", "rb") as f:
    buf = f.read()

# Let's inspect which words are non-zero in all 4 faces of scratch_in
import numpy as np
arr = np.frombuffer(buf, dtype=np.uint16)
for face in range(4):
    f_arr = arr[face*256:(face+1)*256].reshape(16, 16)
    print(f"Face {face}:")
    for r in range(16):
        nz = np.count_nonzero(f_arr[r])
        print(f"  Row {r:2d}: {nz} non-zero elements")
