#include <iostream>
#include "ldpc_tile_matrix_core.h"

int main() {
    uint32_t v = 1683;
    uint32_t off = get_tile_vnode_byte_offset(v);
    std::cout << "v=" << v << " offset=" << off << " (byte " << off << " of 20480)" << std::endl;
    return 0;
}
