def compare():
    with open("tests/test_tile_matrix_40tile_lockstep_stepper.cpp") as f1, open("engines/tile_matrix_4cw/ldpc_sim_tile_matrix.cpp") as f2:
        t1 = f1.read()
        t2 = f2.read()
        
    # Find loop in stepper
    p1 = t1.find("for (uint32_t iter = 0; iter < num_iterations; ++iter)")
    p2 = t2.find("for (uint32_t iter = 0; iter < max_iterations; ++iter)")
    
    sub1 = t1[p1:p1+5500]
    sub2 = t2[p2:p2+5500]
    
    import difflib
    diff = list(difflib.unified_diff(sub1.splitlines(), sub2.splitlines(), fromfile="stepper_main_loop", tofile="sim_diff_test_loop", lineterm=""))
    for d in diff:
        print(d)

if __name__ == "__main__":
    compare()
