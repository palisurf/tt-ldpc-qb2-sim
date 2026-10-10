import struct

def bf_to_f(u):
    return struct.unpack('>f', struct.pack('>HH', int(u), 0))[0]

tensix_final = open('dump/tensix_final_vnodes.bin', 'rb').read()
golden_final = open('dump/golden_final_vnodes.bin', 'rb').read()
tensix_init = open('dump/tensix_initial_vnodes.bin', 'rb').read()

print('Initial v0 cw0..3:', [hex(struct.unpack('<H', tensix_init[i*2:i*2+2])[0]) for i in range(4)])
print('Tensix  v0 cw0..3:', [hex(struct.unpack('<H', tensix_final[i*2:i*2+2])[0]) for i in range(4)])
print('Golden  v0 cw0..3:', [hex(struct.unpack('<H', golden_final[i*2:i*2+2])[0]) for i in range(4)])
