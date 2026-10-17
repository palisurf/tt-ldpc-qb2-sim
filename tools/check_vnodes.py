import struct

def get_tile_vnode_byte_offset(v):
    t = v // 256
    v_loc = v % 256
    row = v_loc // 8
    k = v_loc % 8
    face = (0 if k < 4 else 1) if row < 16 else (2 if k < 4 else 3)
    col_grp = k if k < 4 else (k - 4)
    return t * 2048 + (face * 512) + ((row & 15) << 5) + (col_grp << 3)

def main():
    with open('dump/tensix_initial_vnodes.bin', 'rb') as f:
        vnodes = f.read()
    with open('dump/tensix_scratch_in.bin', 'rb') as f:
        scat_in = f.read()

    scat_w = [x[0] for x in struct.iter_unpack('<H', scat_in[:16])]
    print("Scat_in first 8 words:", [hex(x) for x in scat_w[:8]])

    # Let's check what vnode 571 has in tensix_initial_vnodes.bin:
    off_571 = get_tile_vnode_byte_offset(571)
    v571_words = [x[0] for x in struct.iter_unpack('<H', vnodes[off_571 : off_571 + 8])]
    print(f"Vnode 571 words (off=0x{off_571:x}):", [hex(x) for x in v571_words])

    # Let's check what vnode 43 has in tensix_initial_vnodes.bin:
    off_43 = get_tile_vnode_byte_offset(43)
    v43_words = [x[0] for x in struct.iter_unpack('<H', vnodes[off_43 : off_43 + 8])]
    print(f"Vnode 43 words (off=0x{off_43:x}):", [hex(x) for x in v43_words])

    # Where do scat_w[0..4] come from in vnodes?
    v_words = [x[0] for x in struct.iter_unpack('<H', vnodes)]
    for v in range(2560):
        off = get_tile_vnode_byte_offset(v)
        chunk = [x[0] for x in struct.iter_unpack('<H', vnodes[off : off + 8])]
        if chunk == scat_w[:4]:
            print(f"Found scat_w[0..3] at vnode {v}, byte_off=0x{off:x}")

if __name__ == '__main__':
    main()
