import struct
import numpy as np

# Load golden_scratch_in and tensix_scratch_in (known 100% bit-exact match)
with open('dump/tensix_scratch_in.bin', 'rb') as f:
    t_in = f.read()

# Let's inspect how the hardware and emulator compute each face
print('Hardware formula test script created.')
