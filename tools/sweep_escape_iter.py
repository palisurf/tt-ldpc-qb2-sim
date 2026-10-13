#!/usr/bin/env python3
import subprocess, re

for it in range(40, 52):
    cmd = f"source /home/ttuser/.tenstorrent-venv/bin/activate && export TT_METAL_HOME=/home/ttuser/tt-metal && export TT_METAL_RUNTIME_ROOT=/home/ttuser/tt-metal && export TT_METAL_LOGGER_LEVEL=WARN && cd /home/ttuser/LDPC_QB2_GA && ./engines/tile_matrix_4cw/ldpc_sim_tile_matrix --diff-test-real 2.0 {it} 777594 1"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
    m = re.search(r"Final Batch Errs\s+:\s+(.*)", res.stdout)
    m_iters = re.search(r"Final Batch Iters:\s+(.*)", res.stdout)
    errs_str = m.group(1) if m else "N/A"
    iters_str = m_iters.group(1) if m_iters else "N/A"
    print(f"Iter {it:2d}: Errs=[{errs_str}] | Iters=[{iters_str}]", flush=True)
