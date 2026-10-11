#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <cstring>
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

bool parse_chinn_out(const std::string& filepath, ParityCheckMatrix& pcm) {
    std::ifstream infile(filepath);
    if (!infile.is_open()) return false;
    struct RawEdge { uint32_t r, c; float val; };
    std::vector<RawEdge> raw_lines;
    uint32_t r, c; float val;
    while (infile >> r >> c >> val) raw_lines.push_back({r, c, val});
    const auto& dim_line = raw_lines.back();
    pcm.M = dim_line.r; pcm.N = dim_line.c; pcm.K = pcm.N - pcm.M;
    pcm.circulant_p = pcm.M / 3; pcm.P = pcm.circulant_p;
    pcm.check_to_var.resize(pcm.M); pcm.var_to_check.resize(pcm.N);
    for (size_t i = 0; i + 1 < raw_lines.size(); ++i) {
        if (std::abs(raw_lines[i].val) > 1e-6f) {
            uint32_t m_idx = raw_lines[i].r - 1;
            uint32_t v_idx = raw_lines[i].c - 1;
            pcm.check_to_var[m_idx].push_back(v_idx);
            pcm.var_to_check[v_idx].push_back(m_idx);
        }
    }
    return true;
}

static void partition_checks_balanced(
    const std::vector<uint32_t>& checks,
    const ParityCheckMatrix& pcm,
    uint32_t target_tiles,
    uint32_t max_checks_per_tile,
    std::vector<std::vector<uint32_t>>& out_tiles
) {
    out_tiles.clear();
    out_tiles.resize(target_tiles);
    std::vector<std::unordered_set<uint32_t>> tile_vars(target_tiles);
    for (uint32_t c : checks) {
        const auto& c_vars = pcm.check_to_var[c];
        int best_tile = -1;
        size_t min_size = 999999;
        for (uint32_t t = 0; t < target_tiles; ++t) {
            if (out_tiles[t].size() < max_checks_per_tile) {
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
    ParityCheckMatrix pcm;
    parse_chinn_out("matrices/AR4JA_r45_4c_128c_r12.chinn.out", pcm);
    std::vector<uint32_t> deg6_checks;
    for (uint32_t m = 0; m < pcm.M; ++m) {
        if (pcm.check_to_var[m].size() == 6) deg6_checks.push_back(m);
    }
    std::vector<std::vector<uint32_t>> deg6_tiles;
    partition_checks_balanced(deg6_checks, pcm, 32, 32, deg6_tiles);

    const auto& tile = deg6_tiles[0];
    for (size_t c = 0; c < 32; ++c) {
        uint32_t chk = tile[c];
        std::cout << "Check " << c << " (chk " << chk << "): ";
        for (int d = 0; d < 6; ++d) {
            uint32_t v = pcm.check_to_var[chk][d];
            std::cout << v << (v >= 2048 ? "(P) " : " ");
        }
        std::cout << std::endl;
    }
    return 0;
}
