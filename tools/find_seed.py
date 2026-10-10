import subprocess

for seed in range(100):
    cmd = f"./test_40tile_stepper --standalone 2.0 50 {seed}"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd="/home/ttuser/LDPC_QB2_GA")
    if "[CONVERGENCE]" not in res.stdout:
        print(f"Seed {seed} did not converge:")
        for line in res.stdout.splitlines()[-15:]:
            print("  ", line)
        break
else:
    print("All 100 seeds converged!")
