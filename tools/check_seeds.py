import subprocess

failed = 0
for b in range(1000):
    seed = 3235967 + b * 17
    res = subprocess.run(["./test_40tile_stepper", "--standalone", "2.0", "16", str(seed)], capture_output=True, text=True, cwd="/home/ttuser/LDPC_QB2_GA")
    if "CONVERGENCE" not in res.stdout:
        failed += 1
        print(f"Batch {b} (seed {seed}) FAILED!")
        for line in res.stdout.splitlines():
            if "Iter" in line or "Initial" in line:
                print("  ", line)
        if failed >= 5:
            break
print(f"Scanned {b+1} batches ({(b+1)*4} codewords): {failed} failures.")
