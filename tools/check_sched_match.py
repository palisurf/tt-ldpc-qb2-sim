import subprocess

def test():
    # Let's inspect the first 5 tiles in test_40tile_stepper vs ldpc_sim_tile_matrix
    cmd1 = "cd /home/ttuser/LDPC_QB2_GA/tests && ./test_40tile_stepper 1 /home/ttuser/LDPC_QB2_GA/engines/tile_matrix_4cw/dump/tensix_initial_vnodes.bin /home/ttuser/LDPC_QB2_GA/engines/tile_matrix_4cw/dump/tensix_final_vnodes.bin"
    # Wait, we can dump the schedule check indices from both!
    print("Testing schedule check indices...")

if __name__ == "__main__":
    test()
