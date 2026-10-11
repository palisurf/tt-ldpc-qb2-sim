import struct
with open('dump/golden_scratch_out.bin', 'rb') as f: g = f.read()
with open('dump/tensix_scratch_out.bin', 'rb') as f: t = f.read()
for f in range(4):
    row_counts = [0]*16
    for idx in range(f*256, (f+1)*256):
        gw = struct.unpack('<H', g[idx*2:(idx+1)*2])[0]
        tw = struct.unpack('<H', t[idx*2:(idx+1)*2])[0]
        if gw != tw:
            row_counts[(idx - f*256)//16] += 1
    print(f'Face {f} mismatches per row: {row_counts}')
