import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inspect_tile0_h import deg6_tiles, make_pair_word, c_to_v

# Let's compute w0_pairs for deg6_tiles[0] and deg6_tiles[3]
def get_w0(tile):
    w0 = [0]*48
    for p in range(48):
        if p < 24:
            d = p // 4
            kh = p % 4
            c0_even = 0 + kh
            c0_odd = 4 + kh
        else:
            d = (p - 24) // 4
            kh = (p - 24) % 4
            c0_even = 16 + kh
            c0_odd = 20 + kh
        v0_even = c_to_v[tile[c0_even]][d]
        v0_odd = c_to_v[tile[c0_odd]][d]
        w0[p] = make_pair_word(v0_even, v0_odd)
    return w0

w0_tile0 = get_w0(deg6_tiles[0])
w0_tile3 = get_w0(deg6_tiles[3])

print("deg6_tiles[0] w0_pairs[:4]:")
for i in range(4):
    print(f"  [{i}] = 0x{w0_tile0[i]:08x} (even=0x{w0_tile0[i]&0xffff:04x}, odd=0x{w0_tile0[i]>>16:04x})")

print("\ndeg6_tiles[3] w0_pairs[:4]:")
for i in range(4):
    print(f"  [{i}] = 0x{w0_tile3[i]:08x} (even=0x{w0_tile3[i]&0xffff:04x}, odd=0x{w0_tile3[i]>>16:04x})")
