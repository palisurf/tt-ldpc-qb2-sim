#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <fstream>
#include "engines/tile_matrix_4cw/ldpc_tile_matrix_core.h"

// Let's inspect golden_final_vnodes.bin at byte 0
int main() {
    std::ifstream f_init("dump/tensix_initial_vnodes.bin", std::ios::binary);
    std::ifstream f_gold("dump/golden_final_vnodes.bin", std::ios::binary);
    std::ifstream f_ten("dump/tensix_final_vnodes.bin", std::ios::binary);

    std::vector<uint16_t> init(10240), gold(10240), ten(10240);
    f_init.read(reinterpret_cast<char*>(init.data()), 20480);
    f_gold.read(reinterpret_cast<char*>(gold.data()), 20480);
    f_ten.read(reinterpret_cast<char*>(ten.data()), 20480);

    std::cout << "Byte 0 (Var 0, CW0..3):" << std::endl;
    for (int cw = 0; cw < 4; ++cw) {
        std::cout << "  CW" << cw << ": init=0x" << std::hex << init[cw]
                  << " (" << bf16_to_fp32(init[cw]) << ") | gold=0x"
                  << gold[cw] << " (" << bf16_to_fp32(gold[cw]) << ") | ten=0x"
                  << ten[cw] << " (" << bf16_to_fp32(ten[cw]) << ")" << std::dec << std::endl;
    }
    return 0;
}
