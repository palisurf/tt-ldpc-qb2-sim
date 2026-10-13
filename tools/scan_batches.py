import subprocess, re
for b in range(25):
    cmd = ['bash', '-c', f'export TT_METAL_HOME=/home/ttuser/tt-metal && export TT_METAL_RUNTIME_ROOT=/home/ttuser/tt-metal && export ARCH_NAME=blackhole && ./engines/tile_matrix_4cw/ldpc_sim_tile_matrix --diff-test-real 2.0 200 {b}']
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    m = re.search(r'Total Bit Errors : (\d+)', p.stdout)
    be = m.group(1) if m else 'ERR'
    m_fe = re.search(r'Total Frame Errs : (\d+)', p.stdout)
    fe = m_fe.group(1) if m_fe else 'ERR'
    m_it = re.search(r'CW Iters Taken   : (.*)', p.stdout)
    it = m_it.group(1) if m_it else 'ERR'
    print(f'Batch {b:2d}: BE={be:>4} FE={fe} | {it}')
