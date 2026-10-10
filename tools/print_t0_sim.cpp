#include <iostream>
#include <vector>
#include <fstream>
#include "../engines/tile_matrix_4cw/ldpc_tile_matrix_core.h"

// Let's include the schedule generator
#include "../engines/tile_matrix_4cw/ldpc_sim_tile_matrix.cpp"

int main() {
    ParityCheckMatrix pcm;
    parse_chinn_out("matrices/AR4JA_r45_4c_128c_r12.chinn.out", pcm);
    auto sched = build_circulant_aware_tile_schedule(pcm);
    
    std::cout << "Tile 0 checks in sim build: ";
    for (int i = 0; i < std::min<int>(10, sched[0].check_indices.size()); ++i) {
        std::cout << sched[0].check_indices[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}
