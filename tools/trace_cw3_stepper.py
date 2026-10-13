import subprocess

res = subprocess.run(
    ['./test_40tile_stepper', '200', 'dump/tensix_initial_vnodes.bin', 'dump/tensix_final_vnodes.bin'],
    cwd='/home/ttuser/LDPC_QB2_GA',
    capture_output=True,
    text=True
)

lines = res.stdout.splitlines()
print("=== GOLDEN LOCKSTEPPER TRAJECTORY (Invocation 17, Batch 3653) ===")
first_conv = None
for l in lines:
    if '[Iter ' in l:
        parts = l.split('|')
        # Check CW3 errors: Info BE: [CW0, CW1, CW2, CW3]
        print(l)
    elif 'CONVERGENCE' in l:
        if first_conv is None:
            first_conv = l
            print(">>> FIRST CONVERGENCE EVENT:", l)
