#include <iostream>
#include <iomanip>
#include <vector>
#include "ldpc_tile_matrix_core.h"

int main() {
    ParityCheckMatrix pcm;
    if (!parse_chinn_out("matrices/AR4JA_r45_4c_128c_r12.chinn.out", pcm)) {
        return 1;
    }
    auto schedule = build_circulant_aware_tile_schedule(pcm);
    std::cout << "Schedule tiles: " << schedule.size() << std::endl;
    std::cout << "pcm.h_offsets_pairs size: " << pcm.h_offsets_pairs.size() << std::endl;

    for (int i = 0; i < 8; ++i) {
        uint32_t val = pcm.h_offsets_pairs[i];
        uint32_t even_off = val & 0xFFFF;
        uint32_t odd_off  = val >> 16;
        std::cout << "Pair " << i << ": 0x" << std::hex << val 
                  << " (even=0x" << even_off << ", odd=0x" << odd_off << ")" << std::dec << std::endl;
    }
    return 0;
}
