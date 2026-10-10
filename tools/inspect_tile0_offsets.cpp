#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include "../engines/tile_matrix_4cw/ldpc_tile_matrix_core.h"

// Copy from ldpc_sim_tile_matrix.cpp to inspect Tile 0
struct ParityCheckMatrix {
    uint32_t M = 0;
    uint32_t N = 0;
    uint32_t K = 0;
    uint32_t P = 0;
    uint32_t circulant_p = 0;
    std::vector<std::vector<uint32_t>> check_to_var;
    std::vector<std::vector<uint32_t>> var_to_check;
    std::vector<uint32_t> h_offsets_pairs;
};

static inline uint32_t make_pair_word(uint32_t ve, uint32_t vo) {
    uint32_t oe = get_tile_vnode_byte_offset(ve) & 0xFFFF;
    uint32_t oo = get_tile_vnode_byte_offset(vo) & 0xFFFF;
    return oe | (oo << 16);
}

// Invert get_tile_vnode_byte_offset
uint32_t byte_offset_to_vnode(uint32_t off) {
    for (uint32_t v = 0; v < 2560; ++v) {
        if (get_tile_vnode_byte_offset(v) == off) return v;
    }
    return 9999;
}

int main() {
    std::string matrix_path = "matrices/AR4JA_r45_4c_128c_r12.chinn.out";
    std::ifstream infile(matrix_path);
    if (!infile.is_open()) return 1;

    struct RawEdge { uint32_t r, c; float val; };
    std::vector<RawEdge> raw_lines;
    uint32_t r, c; float val;
    while (infile >> r >> c >> val) raw_lines.push_back({r, c, val});
    const auto& dim_line = raw_lines.back();
    ParityCheckMatrix pcm;
    pcm.M = dim_line.r; pcm.N = dim_line.c; pcm.K = pcm.N - pcm.M;
    pcm.circulant_p = pcm.M / 3; pcm.P = pcm.circulant_p;
    pcm.check_to_var.resize(pcm.M); pcm.var_to_check.resize(pcm.N);
    for (size_t i = 0; i + 1 < raw_lines.size(); ++i) {
        if (std::abs(raw_lines[i].val) > 1e-6f) {
            pcm.check_to_var[raw_lines[i].r - 1].push_back(raw_lines[i].c - 1);
            pcm.var_to_check[raw_lines[i].c - 1].push_back(raw_lines[i].r - 1);
        }
    }

    // Print which variables are touched by the checks in the original partition
    // Let's see what is stored in dump/
    return 0;
}
