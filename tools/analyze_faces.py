import struct

with open('dump/golden_scratch_out.bin', 'rb') as f:
    g = f.read()
with open('dump/tensix_scratch_out.bin', 'rb') as f:
    t = f.read()

fc = [0, 0, 0, 0]
total = 0
for idx in range(1024):
    gw = struct.unpack('<H', g[idx*2:(idx+1)*2])[0]
    tw = struct.unpack('<H', t[idx*2:(idx+1)*2])[0]
    if gw != tw:
        total += 1
        fc[idx // 256] += 1

print(f'Total mismatches: {total}')
for f in range(4):
    print(f'Face {f}: {fc[f]} mismatches')
