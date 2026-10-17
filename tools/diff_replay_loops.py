with open("engines/tile_matrix_4cw/ldpc_sim_tile_matrix.cpp") as f:
    sim_lines = f.readlines()

with open("tests/test_tile_matrix_40tile_lockstep_stepper.cpp") as f:
    step_lines = f.readlines()

# In ldpc_sim_tile_matrix.cpp, find where golden_l1 is processed
# Starts around "for (uint32_t iter = 0; iter < max_iterations; ++iter)" in diff_test block
sim_start = None
for i, l in enumerate(sim_lines):
    if "for (uint32_t iter = 0; iter < max_iterations; ++iter)" in l and i > 600:
        sim_start = i
        break

# In stepper, find where replay iter starts
step_start = None
for i, l in enumerate(step_lines):
    if "for (uint32_t iter = 0; iter < num_iterations; ++iter)" in l and i > 800:
        step_start = i
        break

print(f"sim_start: {sim_start}, step_start: {step_start}")

# Print 120 lines from each
for k in range(120):
    l1 = sim_lines[sim_start + k].strip() if sim_start + k < len(sim_lines) else ""
    l2 = step_lines[step_start + k].strip() if step_start + k < len(step_lines) else ""
    if l1 != l2:
        print(f"L{k:03d} SIM : {l1}")
        print(f"L{k:03d} STEP: {l2}")
        print("-" * 50)
