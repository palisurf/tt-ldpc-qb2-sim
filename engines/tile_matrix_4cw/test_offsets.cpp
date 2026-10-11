#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <cstring>
#include <unordered_set>
#include <cassert>
#include "ldpc_tile_matrix_core.h"

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

#define TILES_PER_DEG6_BLOCK 16
#define TILES_PER_DEG3_BLOCK 8
#define CHECKS_PER_TILE_DEG6 32
#define CHECKS_PER_TILE_DEG3 64
#define TOTAL_TILES_PER_ITER 40

static inline uint32_t make_pair_word(uint32_t ve, uint32_t vo) {
    uint32_t oe = get_tile_vnode_byte_offset(ve) & 0xFFFF;
    uint32_t oo = get_tile_vnode_byte_offset(vo) & 0xFFFF;
    return oe | (oo << 16);
}

void partition_checks_balanced(
    const std::vector<uint32_t>& checks,
    const ParityCheckMatrix& pcm,
    uint32_t target_tile_count,
    uint32_t max_tile_size,
    std::vector<std::vector<uint32_t>>& out_tiles
) {
    out_tiles.assign(target_tile_count, {});
    std::vector<std::unordered_set<uint32_t>> tile_vars(target_tile_count);

    for (uint32_t c : checks) {
        const auto& c_vars = pcm.check_to_var[c];
        int best_tile = -1;
        size_t min_size = max_tile_size + 1;

        for (uint32_t t = 0; t < target_tile_count; ++t) {
            if (out_tiles[t].size() < max_tile_size) {
                bool conflict = false;
                for (uint32_t v : c_vars) {
                    if (tile_vars[t].count(v)) {
                        conflict = true;
                        break;
                    }
                }
                if (!conflict && out_tiles[t].size() < min_size) {
                    min_size = out_tiles[t].size();
                    best_tile = static_cast<int>(t);
                }
            }
        }
        assert(best_tile != -1);
        out_tiles[best_tile].push_back(c);
        for (uint32_t v : c_vars) {
            tile_vars[best_tile].insert(v);
        }
    }
}

int main() {
    std::ifstream infile("../../matrices/AR4JA_r45_4c_128c_r12.chinn.out");
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

    std::vector<uint32_t> deg6_checks;
    for (uint32_t m = 0; m < pcm.M; ++m) {
        if (pcm.check_to_var[m].size() == 6) deg6_checks.push_back(m);
    }
    std::vector<std::vector<uint32_t>> deg6_tiles;
    partition_checks_balanced(deg6_checks, pcm, TILES_PER_DEG6_BLOCK * 2, CHECKS_PER_TILE_DEG6, deg6_tiles);

    const auto& tile0 = deg6_tiles[0];
    std::ofstream out("dump/c_tile0_offsets.txt");

    for (uint32_t p = 0; p < 48; ++p) {
        uint32_t d, kh, c0_even, c0_odd, c1_even, c1_odd;
        if (p < 24) {
            d = p / 4; kh = p % 4;
            c0_even = 0 + kh; c0_odd = 4 + kh;
            c1_even = 8 + kh; c1_odd = 12 + kh;
        } else {
            d = (p - 24) / 4; kh = (p - 24) % 4;
            c0_even = 16 + kh; c0_odd = 20 + kh;
            c1_even = 24 + kh; c1_odd = 28 + kh;
        }
        uint32_t v0_e = pcm.check_to_var[tile0[c0_even]][d];
        uint32_t v0_o = pcm.check_to_var[tile0[c0_odd]][d];
        uint32_t v1_e = pcm.check_to_var[tile0[c1_even]][d];
        uint32_t v1_o = pcm.check_to_var[tile0[c1_odd]][d];

        uint32_t off0_e = get_tile_vnode_byte_offset(v0_e);
        uint32_t off0_o = get_tile_vnode_byte_offset(v0_o);
        uint32_t off1_e = get_tile_vnode_byte_offset(v1_e);
        uint32_t off1_o = get_tile_vnode_byte_offset(v1_o);

        out << off0_e << " " << v0_e << " w0_even p=" << p << "\n";
        out << off0_o << " " << v0_o << " w0_odd p=" << p << "\n";
        out << off1_e << " " << v1_e << " w1_even p=" << p << "\n";
        out << off1_o << " " << v1_o << " w1_odd p=" << p << "\n";
    }
    out.close();
    std::cout << "Dumped 192 offsets to dump/c_tile0_offsets.txt" << std::endl;
    return 0;
}
