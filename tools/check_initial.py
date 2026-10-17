import numpy as np

t_init = np.fromfile('dump/tensix_initial_vnodes.bin', dtype=np.uint16)
print('Total words in tensix_initial_vnodes:', len(t_init))
print('First 16 words:', [hex(x) for x in t_init[:16]])

t_scat_in = np.fromfile('dump/tensix_scratch_in.bin', dtype=np.uint16)
g_scat_in = np.fromfile('dump/golden_scratch_in.bin', dtype=np.uint16)
print('t_scat_in first 16 words:', [hex(x) for x in t_scat_in[:16]])
print('g_scat_in first 16 words:', [hex(x) for x in g_scat_in[:16]])

# Check Tile 0, check 0 (which is row 0, pair 0)
# pcm.h_offsets_pairs[0]
with open('dump/golden_scratch_in.bin', 'rb') as f:
    g_bytes = f.read()

# Let's inspect t_scat_in vs g_scat_in across all 1024 words
diffs = [i for i in range(1024) if t_scat_in[i] != g_scat_in[i]]
print('Total diffs in scratch_in:', len(diffs))
print('First 10 diff indices:', diffs[:10])
for idx in diffs[:5]:
    print(f'idx={idx} (row={idx//32}, col={idx%32}): tensix=0x{t_scat_in[idx]:04x} golden=0x{g_scat_in[idx]:04x}')

for val in [0x405a, 0x403d]:
    matches = np.where(t_init == val)[0]
    print(f'Value 0x{val:04x} found at indices in t_init: {matches[:10]}')
