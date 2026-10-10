#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <unordered_set>
#include <algorithm>
#include "engines/tile_matrix_4cw/ldpc_tile_matrix_core.h"

struct ParityCheckMatrix {
    uint32_t M = 0, N = 0, K = 0, P = 0, circulant_p = 0;
    std::vector<std::vector<uint32_t>> check_to_var;
    std::vector<std::vector<uint32_t>> var_to_check;
    std::vector<uint32_t> h_offsets_pairs;
};

struct DynamicTileDescriptor {
    std::vector<uint32_t> check_indices;
    uint8_t  check_degree;
    uint8_t  num_checks;
    uint16_t num_edges;
};

static inline uint32_t make_pair_word(uint32_t ve, uint32_t vo) {
    uint32_t oe = get_tile_vnode_byte_offset(ve) & 0xFFFF;
    uint32_t oo = get_tile_vnode_byte_offset(vo) & 0xFFFF;
    return oe | (oo << 16);
}

inline float approx_min_star(float a, float b) {
    float min_v = std::min(a, b);
    float max_v = std::max(a, b);
    float diff  = max_v - min_v;
    float line1 = 0.69315f - 0.4000f * diff;
    float line2 = 0.65715f - 0.3280f * diff;
    float corr  = std::max(std::max(line1, line2), 0.0f);
    return std::max(min_v - corr, 0.0f);
}

int main() {
    std::ifstream infile("matrices/AR4JA_r45_4c_128c_r12.chinn.out");
    ParityCheckMatrix pcm;
    struct RawEdge { uint32_t r, c; float val; };
    std::vector<RawEdge> raw;
    uint32_t r, c; float val;
    while (infile >> r >> c >> val) raw.push_back({r, c, val});
    pcm.M = raw.back().r; pcm.N = raw.back().c;
    pcm.check_to_var.resize(pcm.M);
    for (size_t i = 0; i + 1 < raw.size(); ++i) {
        if (std::abs(raw[i].val) > 1e-6f) pcm.check_to_var[raw[i].r - 1].push_back(raw[i].c - 1);
    }

    std::vector<uint32_t> deg6_checks, deg3_checks;
    for (uint32_t m = 0; m < pcm.M; ++m) {
        if (pcm.check_to_var[m].size() == 6) deg6_checks.push_back(m);
        else if (pcm.check_to_var[m].size() == 3) deg3_checks.push_back(m);
    }

    auto partition = [&](const std::vector<uint32_t>& check_list, uint32_t target_count, uint32_t max_size) {
        std::vector<std::vector<uint32_t>> var_to_c(2560);
        for (uint32_t ch : check_list) for (uint32_t v : pcm.check_to_var[ch]) var_to_c[v].push_back(ch);
        std::vector<std::pair<uint32_t, uint32_t>> check_conflicts;
        for (uint32_t ch : check_list) {
            std::unordered_set<uint32_t> conf;
            for (uint32_t v : pcm.check_to_var[ch]) for (uint32_t other : var_to_c[v]) if (other != ch) conf.insert(other);
            check_conflicts.push_back({ch, (uint32_t)conf.size()});
        }
        std::sort(check_conflicts.begin(), check_conflicts.end(), [](auto& a, auto& b){ return a.second > b.second; });
        std::vector<std::vector<uint32_t>> out_tiles(target_count);
        std::vector<std::unordered_set<uint32_t>> tile_vars(target_count);
        for (auto& item : check_conflicts) {
            uint32_t ch = item.first;
            std::unordered_set<uint32_t> c_vars(pcm.check_to_var[ch].begin(), pcm.check_to_var[ch].end());
            int best = -1; size_t min_s = max_size + 1;
            for (uint32_t t = 0; t < target_count; ++t) {
                if (out_tiles[t].size() < max_size) {
                    bool conf = false;
                    for (uint32_t v : c_vars) if (tile_vars[t].count(v)) { conf = true; break; }
                    if (!conf && out_tiles[t].size() < min_s) { min_s = out_tiles[t].size(); best = t; }
                }
            }
            out_tiles[best].push_back(ch);
            for (uint32_t v : c_vars) tile_vars[best].insert(v);
        }
        return out_tiles;
    };

    auto deg6_tiles = partition(deg6_checks, 26, 40);
    auto deg3_tiles = partition(deg3_checks, 7, 80);

    std::vector<DynamicTileDescriptor> schedule;
    pcm.h_offsets_pairs.clear();

    auto add_tile = [&](const std::vector<uint32_t>& tile, uint8_t deg) {
        DynamicTileDescriptor desc{};
        desc.check_indices = tile;
        desc.check_degree = deg;
        desc.num_checks = tile.size();
        desc.num_edges = tile.size() * deg;

        std::vector<uint32_t> w0(60), w1(60);
        for (uint32_t p = 0; p < 60; ++p) {
            uint32_t r_pair = p / 4;
            uint32_t kh     = p % 4;
            uint32_t r_even = 2 * r_pair;
            uint32_t r_odd  = 2 * r_pair + 1;
            uint32_t b_even = r_even / deg;
            uint32_t d_even = r_even % deg;
            uint32_t b_odd  = r_odd / deg;
            uint32_t d_odd  = r_odd % deg;

            uint32_t c0_even = 8 * b_even + kh;
            uint32_t c0_odd  = 8 * b_odd + kh;
            uint32_t v0_even = (c0_even < tile.size()) ? pcm.check_to_var[tile[c0_even]][d_even] : 2559;
            uint32_t v0_odd  = (c0_odd < tile.size())  ? pcm.check_to_var[tile[c0_odd]][d_odd]   : 2559;
            w0[p] = make_pair_word(v0_even, v0_odd);

            uint32_t c1_even = 8 * b_even + 4 + kh;
            uint32_t c1_odd  = 8 * b_odd + 4 + kh;
            uint32_t v1_even = (c1_even < tile.size()) ? pcm.check_to_var[tile[c1_even]][d_even] : 2559;
            uint32_t v1_odd  = (c1_odd < tile.size())  ? pcm.check_to_var[tile[c1_odd]][d_odd]   : 2559;
            w1[p] = make_pair_word(v1_even, v1_odd);
        }
        for (int p = 0; p < 60; ++p) pcm.h_offsets_pairs.push_back(w0[p]);
        for (int p = 0; p < 60; ++p) pcm.h_offsets_pairs.push_back(w1[p]);
        schedule.push_back(desc);
    };

    size_t i6 = 0, i3 = 0;
    while (i6 < deg6_tiles.size() || i3 < deg3_tiles.size()) {
        size_t quota_i6 = (i3 + 1 < deg3_tiles.size()) ? ((i3 + 1) * deg6_tiles.size() / deg3_tiles.size()) : deg6_tiles.size();
        while (i6 < quota_i6 && i6 < deg6_tiles.size()) add_tile(deg6_tiles[i6++], 6);
        if (i3 < deg3_tiles.size()) add_tile(deg3_tiles[i3++], 3);
    }

    for (size_t t = 0; t < schedule.size(); ++t) {
        for (int p = 0; p < 60; ++p) {
            uint32_t off0 = pcm.h_offsets_pairs[t * 120 + p];
            uint32_t off1 = pcm.h_offsets_pairs[t * 120 + 60 + p];
            if ((off0 & 0xFFFF) == 0) std::cout << "Tile " << t << " W0 even p=" << p << " has off=0" << std::endl;
            if ((off0 >> 16) == 0)    std::cout << "Tile " << t << " W0 odd  p=" << p << " has off=0" << std::endl;
            if ((off1 & 0xFFFF) == 0) std::cout << "Tile " << t << " W1 even p=" << p << " has off=0" << std::endl;
            if ((off1 >> 16) == 0)    std::cout << "Tile " << t << " W1 odd  p=" << p << " has off=0" << std::endl;
        }
    }

    std::ifstream f_init("dump/tensix_initial_vnodes.bin", std::ios::binary);
    std::vector<uint8_t> golden_l1(20480);
    f_init.read(reinterpret_cast<char*>(golden_l1.data()), 20480);
    uint8_t* vnode_l1 = golden_l1.data();

    std::vector<std::vector<uint16_t>> r_msg_fifo(schedule.size(), std::vector<uint16_t>(1024, 0));
    std::vector<uint16_t> scratch_in(1024, 0);
    std::vector<uint16_t> scratch_out(1024, 0);

    for (size_t t = 0; t < schedule.size(); ++t) {
        const auto& tile = schedule[t];
        uint32_t deg = tile.check_degree;
        uint32_t num_blocks = (deg == 6) ? 5 : 10;
        const uint32_t* tile_hoff_w0 = pcm.h_offsets_pairs.data() + t * 120;
        const uint32_t* tile_hoff_w1 = tile_hoff_w0 + 60;

        for (int p = 0; p < 60; ++p) {
            uint32_t off0 = tile_hoff_w0[p];
            uint32_t off0_even = off0 & 0xFFFF;
            uint32_t off0_odd  = off0 >> 16;
            uint32_t s0_even = SCRATCH_FACE_LUT.w0_even[p];
            uint32_t s0_odd  = SCRATCH_FACE_LUT.w0_odd[p];

            *reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(scratch_in.data()) + s0_even) =
                *reinterpret_cast<const uint64_t*>(vnode_l1 + off0_even);
            *reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(scratch_in.data()) + s0_odd) =
                *reinterpret_cast<const uint64_t*>(vnode_l1 + off0_odd);

            uint32_t off1 = tile_hoff_w1[p];
            uint32_t off1_even = off1 & 0xFFFF;
            uint32_t off1_odd  = off1 >> 16;
            uint32_t s1_even = SCRATCH_FACE_LUT.w1_even[p];
            uint32_t s1_odd  = SCRATCH_FACE_LUT.w1_odd[p];

            *reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(scratch_in.data()) + s1_even) =
                *reinterpret_cast<const uint64_t*>(vnode_l1 + off1_even);
            *reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(scratch_in.data()) + s1_odd) =
                *reinterpret_cast<const uint64_t*>(vnode_l1 + off1_odd);
        }

        for (uint32_t b = 0; b < num_blocks; ++b) {
            uint32_t base_r = b * deg;
            for (uint32_t lane = 0; lane < 32; ++lane) {
                uint32_t col = lane;
                float q[8], signs[8], mags[8];
                float total_sign = 1.0f;
                uint32_t min_idx = 0;
                float min_mag = 99999.0f;

                for (uint32_t r = 0; r < deg; ++r) {
                    uint32_t offset = get_scratch_byte_offset(base_r + r, col);
                    float l_val = bf16_to_fp32(*reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(scratch_in.data()) + offset));
                    float r_old = bf16_to_fp32(*reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(r_msg_fifo[t].data()) + offset));

                    q[r] = l_val - r_old;
                    signs[r] = (q[r] < 0.0f) ? -1.0f : 1.0f;
                    mags[r]  = std::fabs(q[r]);
                    total_sign *= signs[r];
                    if (mags[r] < min_mag) {
                        min_mag = mags[r];
                        min_idx = r;
                    }
                }

                float m_others = (min_idx == 0) ? 1000.0f : mags[0];
                for (uint32_t r = 1; r < deg; ++r) {
                    float m = (min_idx == r) ? 1000.0f : mags[r];
                    m_others = approx_min_star(m_others, m);
                }
                float m_all = approx_min_star(m_others, min_mag);

                if (t == 10 && b == 3 && col == 28) {
                    std::cout << "[Tile 10 b=3 col=28] deg=" << deg << " min_idx=" << min_idx << " min_mag=" << min_mag << std::endl;
                    for (uint32_t r = 0; r < deg; ++r) {
                        uint32_t offset = get_scratch_byte_offset(base_r + r, col);
                        float l_val = bf16_to_fp32(*reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(scratch_in.data()) + offset));
                        std::cout << "  r=" << r << ": q=" << q[r] << " (l_val=" << l_val << ", offset=" << offset << ")" << std::endl;
                    }
                    std::cout << "  m_others=" << m_others << ", m_all=" << m_all << ", total_sign=" << total_sign << std::endl;
                }

                for (uint32_t r = 0; r < deg; ++r) {
                    uint32_t offset = get_scratch_byte_offset(base_r + r, col);
                    float edge_sign = total_sign * signs[r];
                    float edge_mag = (r == min_idx) ? m_others : m_all;
                    float r_new = edge_sign * edge_mag;
                    float l_new = q[r] + r_new;

                    if (t == 10 && b == 3 && col == 28 && r == 0) {
                        std::cout << "  r=0 output: edge_sign=" << edge_sign << " edge_mag=" << edge_mag
                                  << " r_new=" << r_new << " l_new=" << l_new << " (bf16=0x"
                                  << std::hex << fp32_to_bf16(l_new) << std::dec << ")" << std::endl;
                    }

                    *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(r_msg_fifo[t].data()) + offset) = fp32_to_bf16(r_new);
                    *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(scratch_out.data()) + offset)   = fp32_to_bf16(l_new);
                }
            }
        }

        if (t == 10) {
            uint32_t off1 = tile_hoff_w1[39];
            uint32_t s1_even = SCRATCH_FACE_LUT.w1_even[39];
            uint16_t val_to_scatter = *reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(scratch_out.data()) + s1_even);
            std::cout << "[Tile 10] p=39: off1_even=" << (off1 & 0xFFFF) << ", s1_even=" << s1_even
                      << ", val_to_scatter=0x" << std::hex << val_to_scatter
                      << " (" << bf16_to_fp32(val_to_scatter) << ")" << std::dec << std::endl;

            uint32_t r_pair = 39 / 4, kh = 39 % 4, r_even = 2 * r_pair;
            uint32_t b_even = r_even / deg, d_even = r_even % deg;
            uint32_t c1_even = 8 * b_even + 4 + kh;
            std::cout << "  b_even=" << b_even << " d_even=" << d_even << " c1_even=" << c1_even
                      << " tile.size=" << tile.num_checks;
            if (c1_even < tile.num_checks) {
                uint32_t check_idx = tile.check_indices[c1_even];
                std::cout << " check_idx=" << check_idx << " var=" << pcm.check_to_var[check_idx][d_even] << std::endl;
            } else {
                std::cout << " (PADDING)" << std::endl;
            }
        }
        if (t == 16) {
            uint32_t off1 = tile_hoff_w1[14];
            uint32_t s1_even = SCRATCH_FACE_LUT.w1_even[14];
            uint16_t val_to_scatter = *reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(scratch_out.data()) + s1_even);
            std::cout << "[Tile 16] p=14: off1_even=" << (off1 & 0xFFFF) << ", s1_even=" << s1_even
                      << ", val_to_scatter=0x" << std::hex << val_to_scatter
                      << " (" << bf16_to_fp32(val_to_scatter) << ")" << std::dec << std::endl;
        }

        for (int p = 0; p < 60; ++p) {
            uint32_t off0 = tile_hoff_w0[p];
            uint32_t off0_even = off0 & 0xFFFF;
            uint32_t off0_odd  = off0 >> 16;
            uint32_t s0_even = SCRATCH_FACE_LUT.w0_even[p];
            uint32_t s0_odd  = SCRATCH_FACE_LUT.w0_odd[p];

            *reinterpret_cast<uint64_t*>(vnode_l1 + off0_even) =
                *reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(scratch_out.data()) + s0_even);
            *reinterpret_cast<uint64_t*>(vnode_l1 + off0_odd) =
                *reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(scratch_out.data()) + s0_odd);

            uint32_t off1 = tile_hoff_w1[p];
            uint32_t off1_even = off1 & 0xFFFF;
            uint32_t off1_odd  = off1 >> 16;
            uint32_t s1_even = SCRATCH_FACE_LUT.w1_even[p];
            uint32_t s1_odd  = SCRATCH_FACE_LUT.w1_odd[p];

            *reinterpret_cast<uint64_t*>(vnode_l1 + off1_even) =
                *reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(scratch_out.data()) + s1_even);
            *reinterpret_cast<uint64_t*>(vnode_l1 + off1_odd) =
                *reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(scratch_out.data()) + s1_odd);
        }

        if (t == 10 || t == 12) {
            uint16_t v0 = *reinterpret_cast<uint16_t*>(vnode_l1);
            std::cout << "[After Tile " << t << "] vnode_l1[0] = 0x" << std::hex << v0
                      << " (" << bf16_to_fp32(v0) << ")" << std::dec << std::endl;
        }
    }

    return 0;
}
