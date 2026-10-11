import ctypes
import os

so_path = '/home/ttuser/tt-metal/sim/libttsim_bh.so'
stat = os.stat(so_path)
print(f'Library: {so_path}')
print(f'Size:    {stat.st_size} bytes')
print(f'MTime:   {stat.st_mtime}')

lib = ctypes.CDLL(so_path)
print('Calling libttsim_init()...')
lib.libttsim_init()
print('libttsim_init() succeeded!')

print('Stepping 100 clock cycles via libttsim_clock(100)...')
lib.libttsim_clock(100)
print('Clock step completed!')

print('Calling libttsim_exit()...')
lib.libttsim_exit()
print('libttsim_exit() succeeded!')
