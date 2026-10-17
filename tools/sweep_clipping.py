import subprocess

for clip in [0.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0]:
    res = subprocess.run(
        ['./test_40tile_stepper', '40', 'dump/tensix_initial_vnodes.bin', 'dump/tensix_final_vnodes.bin', str(clip)],
        cwd='/home/ttuser/LDPC_QB2_GA',
        capture_output=True,
        text=True
    )
    lines = [l for l in res.stdout.splitlines() if 'Final' in l or 'Converged' in l or 'Iterations' in l or 'CW2' in l or 'Errors' in l or 'Iter' in l]
    print(f'=== CLIP {clip:5.1f} ===')
    # Print the last 6 lines of relevant output
    for l in lines[-8:]:
        print(' ', l)
