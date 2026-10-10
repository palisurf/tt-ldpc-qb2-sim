def check():
    f1 = "tests/test_tile_matrix_40tile_lockstep_stepper.cpp"
    f2 = "engines/tile_matrix_4cw/ldpc_sim_tile_matrix.cpp"
    with open(f1) as fp1, open(f2) as fp2:
        t1 = fp1.read()
        t2 = fp2.read()
        
    s1_pos = t1.find("build_circulant_aware_tile_schedule")
    s2_pos = t2.find("build_circulant_aware_tile_schedule")
    
    sub1 = t1[s1_pos:s1_pos+4000]
    sub2 = t2[s2_pos:s2_pos+4000]
    
    import difflib
    diff = list(difflib.unified_diff(sub1.splitlines(), sub2.splitlines(), fromfile="stepper", tofile="sim", lineterm=""))
    for d in diff:
        print(d)

if __name__ == "__main__":
    check()
