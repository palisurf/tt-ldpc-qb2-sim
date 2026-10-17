import numpy as np
import os

p_sim_gold = "dump/golden_final_vnodes.bin"
p_step_gold = "dump/golden_lockstep_final_vnodes.bin"

if not os.path.exists(p_sim_gold):
    p_sim_gold = "z:/LDPC_QB2_GA/" + p_sim_gold
    p_step_gold = "z:/LDPC_QB2_GA/" + p_step_gold

g_sim = np.fromfile(p_sim_gold, dtype=np.uint16)
g_step = np.fromfile(p_step_gold, dtype=np.uint16)

diff = (g_sim != g_step)
print(f"Diff between ldpc_sim golden and stepper golden: {np.sum(diff)} / {len(g_sim)}")

if np.sum(diff) > 0:
    idx = np.where(diff)[0][:10]
    for i in idx:
        print(f"Word {i}: sim={hex(g_sim[i])}, step={hex(g_step[i])}")
