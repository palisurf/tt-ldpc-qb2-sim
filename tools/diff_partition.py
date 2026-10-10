def compare():
    with open("tests/test_tile_matrix_40tile_lockstep_stepper.cpp") as f1, open("engines/tile_matrix_4cw/ldpc_sim_tile_matrix.cpp") as f2:
        t1 = f1.read()
        t2 = f2.read()
        
    p1 = t1.find("void partition_checks_balanced")
    p2 = t2.find("void partition_checks_balanced")
    
    sub1 = t1[p1:p1+2500]
    sub2 = t2[p2:p2+2500]
    
    import difflib
    diff = list(difflib.unified_diff(sub1.splitlines(), sub2.splitlines(), fromfile="stepper_partition", tofile="sim_partition", lineterm=""))
    print("Partition differences:", len(diff))
    for d in diff:
        print(d)

if __name__ == "__main__":
    compare()
