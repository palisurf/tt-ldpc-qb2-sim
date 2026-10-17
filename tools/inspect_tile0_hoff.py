# Let's inspect what pcm.h_offsets_pairs has for Tile 0!
# We can run a small C++ program on QB2 that prints schedule[0] and tile_hoff_w0[0..47]
prog = """
#include <iostream>
#include <vector>
#include "ldpc_tile_matrix_core.h"
#include "matrices/ldpc_matrices.h"

// We can read schedule from ldpc_sim_tile_matrix if we compile a small test
"""
