import os
import subprocess

os.chdir('/home/ttuser/LDPC_QB2_GA')
for b in range(20):
    cmd = ['./test_40tile_stepper', '200', f'dump/batch_{b}_in.bin', f'dump/batch_{b}_out.bin']
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    conv = [l for l in p.stdout.splitlines() if 'CONVERGENCE' in l]
    first = conv[0] if conv else 'NO CONVERGENCE'
    print(f'Batch {b:2d}: {first}')
