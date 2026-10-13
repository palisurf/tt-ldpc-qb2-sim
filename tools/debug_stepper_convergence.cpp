#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <fstream>
#include " ../engines/tile_matrix_4cw/ldpc_tile_matrix_core.h\

struct ParityCheckMatrix {
 uint32_t M = 0, N = 0, K = 0, P = 0, circulant_p = 0;
 std::vector<std::vector<uint32_t>> check_to_var;
 std::vector<std::vector<uint32_t>> var_to_check;
 std::vector<std::vector<uint32_t>> check_to_var_offsets;
 std::vector<uint32_t> info_var_offsets;
 std::vector<uint32_t> h_offsets_pairs;
};

// We will check why initial syn or decoding fails
int main() {
 std::cout << \Diagnostic starting...\ << std::endl;
 return 0;
}
