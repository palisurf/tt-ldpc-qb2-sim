import numpy as np

sin = np.frombuffer(open("dump/tensix_scratch_in.bin", "rb").read(), dtype=np.uint16)
sout = np.frombuffer(open("dump/tensix_scratch_out.bin", "rb").read(), dtype=np.uint16)

for face in range(4):
    si = sin[face*256:(face+1)*256].reshape(16, 16)
    so = sout[face*256:(face+1)*256].reshape(16, 16)
    print(f"Face {face}:")
    for r in range(16):
        num_changed = np.sum(si[r] != so[r])
        print(f"  Row {r:2d}: {num_changed} elements changed between in and out")
