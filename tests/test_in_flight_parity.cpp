#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <fstream>

#include "ldpc_tile_matrix_core.h"

// Bring in SFPI CPU vector emulator
#include "sfpi_host_emu.h"



struct ParityCheckMatrix {
    uint32_t M = 0;
    uint32_t N = 0;
    uint32_t K = 0;
    uint32_t P = 0;
    std::vector<std::vector<uint32_t>> check_to_var;
    std::vector<std::vector<uint32_t>> check_to_var_offsets;
    std::vector<uint32_t> h_offsets_pairs;
};

inline bool load_chinn_matrix(const std::string& filepath, ParityCheckMatrix& pcm) {
    std::ifstream file(filepath);
    if (!file.is_open()) return false;
    uint32_t r, c;
    float val;
    std::vector<std::pair<uint32_t, uint32_t>> edges;
    uint32_t max_r = 0, max_c = 0;
    while (file >> r >> c >> val) {
        if (val == 0.0f) {
            max_r = r;
            max_c = c;
            continue;
        }
        edges.push_back({r - 1, c - 1});
        if (r > max_r) max_r = r;
        if (c > max_c) max_c = c;
    }
    pcm.M = max_r;
    pcm.N = max_c;
    pcm.K = 1024;
    pcm.P = 512;
    pcm.check_to_var.assign(pcm.M, {});
    pcm.check_to_var_offsets.assign(pcm.M, {});
    for (const auto& edge : edges) {
        pcm.check_to_var[edge.first].push_back(edge.second);
        pcm.check_to_var_offsets[edge.first].push_back(get_tile_vnode_byte_offset(edge.second));
    }
    return true;
}

struct DynamicTileDescriptor {
    std::vector<uint32_t> check_indices;
    uint16_t h_offset_start;
    uint16_t r_msg_page_offset;
    uint8_t check_degree;
    uint8_t num_checks;
    uint16_t num_edges;
};

inline std::vector<DynamicTileDescriptor> build_40tile_schedule(ParityCheckMatrix& pcm) {
    std::vector<uint32_t> deg6_checks, deg3_checks;
    for (uint32_t m = 0; m < pcm.M; ++m) {
        if (pcm.check_to_var[m].size() == 6) deg6_checks.push_back(m);
        else if (pcm.check_to_var[m].size() == 3) deg3_checks.push_back(m);
    }
    std::vector<std::vector<uint32_t>> deg6_tiles(32);
    for (size_t i = 0; i < deg6_checks.size(); ++i) deg6_tiles[i / 32].push_back(deg6_checks[i]);
    std::vector<std::vector<uint32_t>> deg3_tiles(8);
    for (size_t i = 0; i < deg3_checks.size(); ++i) deg3_tiles[i / 64].push_back(deg3_checks[i]);

    std::vector<DynamicTileDescriptor> schedule;
    uint16_t current_h_offset = 0;
    uint16_t current_r_page   = 0;
    pcm.h_offsets_pairs.clear();

    auto add_tile = [&](const std::vector<uint32_t>& tile, uint8_t deg) {
        DynamicTileDescriptor desc{};
        desc.check_indices     = tile;
        desc.check_degree      = deg;
        desc.num_checks        = static_cast<uint8_t>(tile.size());
        desc.num_edges         = static_cast<uint16_t>(desc.num_checks * deg);
        desc.h_offset_start    = current_h_offset;
        desc.r_msg_page_offset = current_r_page;

        std::vector<uint32_t> w0_pairs(48, 0), w1_pairs(48, 0);
        for (uint32_t p = 0; p < 48; ++p) {
            uint32_t d, kh, c0_even, c0_odd, c1_even, c1_odd;
            if (deg == 6) {
                if (p < 24) {
                    d = p / 4; kh = p % 4;
                    c0_even = 0 + kh; c0_odd = 4 + kh;
                    c1_even = 8 + kh; c1_odd = 12 + kh;
                } else {
                    d = (p - 24) / 4; kh = (p - 24) % 4;
                    c0_even = 16 + kh; c0_odd = 20 + kh;
                    c1_even = 24 + kh; c1_odd = 28 + kh;
                }
            } else {
                if (p < 24) {
                    uint32_t blk = p / 12; d = (p % 12) / 4; kh = (p % 12) % 4;
                    c0_even = 0  + blk * 8 + kh; c0_odd = 0  + blk * 8 + 4 + kh;
                    c1_even = 16 + blk * 8 + kh; c1_odd = 16 + blk * 8 + 4 + kh;
                } else {
                    uint32_t blk = (p - 24) / 12; d = ((p - 24) % 12) / 4; kh = ((p - 24) % 12) % 4;
                    c0_even = 32 + blk * 8 + kh; c0_odd = 32 + blk * 8 + 4 + kh;
                    c1_even = 48 + blk * 8 + kh; c1_odd = 48 + blk * 8 + 4 + kh;
                }
            }
            uint32_t v0_even = (c0_even < tile.size() && d < pcm.check_to_var[tile[c0_even]].size()) ? pcm.check_to_var[tile[c0_even]][d] : 0;
            uint32_t v0_odd  = (c0_odd  < tile.size() && d < pcm.check_to_var[tile[c0_odd]].size())  ? pcm.check_to_var[tile[c0_odd]][d]  : 0;
            uint32_t v1_even = (c1_even < tile.size() && d < pcm.check_to_var[tile[c1_even]].size()) ? pcm.check_to_var[tile[c1_even]][d] : 0;
            uint32_t v1_odd  = (c1_odd  < tile.size() && d < pcm.check_to_var[tile[c1_odd]].size())  ? pcm.check_to_var[tile[c1_odd]][d]  : 0;

            w0_pairs[p] = (get_tile_vnode_byte_offset(v0_odd) << 16)  | (get_tile_vnode_byte_offset(v0_even) & 0xFFFF);
            w1_pairs[p] = (get_tile_vnode_byte_offset(v1_odd) << 16)  | (get_tile_vnode_byte_offset(v1_even) & 0xFFFF);
        }
        for (uint32_t p = 0; p < 48; ++p) pcm.h_offsets_pairs.push_back(w0_pairs[p]);
        for (uint32_t p = 0; p < 48; ++p) pcm.h_offsets_pairs.push_back(w1_pairs[p]);
        current_h_offset += 96 * sizeof(uint32_t);
        current_r_page   += 1;
        schedule.push_back(desc);
    };

    for (size_t i = 0; i < 8; ++i) {
        for (size_t j = 0; j < 4; ++j) add_tile(deg6_tiles[i * 4 + j], 6);
        add_tile(deg3_tiles[i], 3);
    }
    return schedule;
}

// Bit-Exact SFPI Vector TRISC Check Node Compute with In-Flight Parity Packing
inline void compute_tile_deg6_with_parity(uint32_t iter, uint32_t t = 0, const ParityCheckMatrix* pcm_ptr = nullptr, const std::vector<DynamicTileDescriptor>* sched_ptr = nullptr, const uint8_t* vnode_l1 = nullptr) {
    const int face_bases[4] = {0, 8, 16, 24};
    sfpi::dst_reg[30] = 0.0f;
    sfpi::dst_reg[31] = 0.0f;

    for (int f = 0; f < 4; ++f) {
        int base_r = face_bases[f];
        int sc0 = (f == 3) ? 6 : (base_r + 6);
        int sc1 = (f == 3) ? 7 : (base_r + 7);
        int sc2 = 32 + base_r + 6;
        int sc3 = 32 + base_r + 7;

        for (int sub = 0; sub < 2; ++sub) {
            int p0 = base_r + 0 + sub;
            int p1 = base_r + 2 + sub;
            int p2 = base_r + 4 + sub;

            if (iter != 0) {
                sfpi::dst_reg[p0] = sfpi::round_bf16(sfpi::vFloat(sfpi::dst_reg[p0]) - sfpi::vFloat(sfpi::dst_reg[32 + p0]));
                sfpi::dst_reg[p1] = sfpi::round_bf16(sfpi::vFloat(sfpi::dst_reg[p1]) - sfpi::vFloat(sfpi::dst_reg[32 + p1]));
                sfpi::dst_reg[p2] = sfpi::round_bf16(sfpi::vFloat(sfpi::dst_reg[p2]) - sfpi::vFloat(sfpi::dst_reg[32 + p2]));
            }

            sfpi::dst_reg[32 + p0] = sfpi::swap_subvec16(sfpi::vFloat(sfpi::dst_reg[p0]));
            sfpi::dst_reg[32 + p1] = sfpi::swap_subvec16(sfpi::vFloat(sfpi::dst_reg[p1]));
            sfpi::dst_reg[32 + p2] = sfpi::swap_subvec16(sfpi::vFloat(sfpi::dst_reg[p2]));

            if (iter == 6 && t == 8 && f == 0 && sub == 1) {
                std::cout << "[DIAGNOSTIC RAW Q BEFORE STAGE 2]:\n";
                float q0_lo = sfpi::dst_reg[p0][0];
                float q0_hi = sfpi::dst_reg[32 + p0][0];
                float q1_lo = sfpi::dst_reg[p1][0];
                float q1_hi = sfpi::dst_reg[32 + p1][0];
                float q2_lo = sfpi::dst_reg[p2][0];
                float q2_hi = sfpi::dst_reg[32 + p2][0];
                std::cout << "  q0_lo=" << q0_lo << " q0_hi=" << q0_hi << " q1_lo=" << q1_lo 
                          << " q1_hi=" << q1_hi << " q2_lo=" << q2_lo << " q2_hi=" << q2_hi << "\n";
            }

            // Stage 2: Sign parity
            {
                sfpi::vInt s0 = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[p0])) ^ sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[32 + p0]));
                sfpi::vInt s1 = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[p1])) ^ sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[32 + p1]));
                sfpi::vInt s2 = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[p2])) ^ sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[32 + p2]));
                sfpi::dst_reg[sc0] = sfpi::as<sfpi::vFloat>((s0 ^ s1) ^ s2); // s_tot in sc0
            }

            // Stage 3: AMin*
            {
                sfpi::vFloat m0    = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[p0]));
                sfpi::vFloat m0_sw = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[32 + p0]));
                sfpi::dst_reg[sc1] = sfpi::round_bf16(sfpi::sfpi_amin_star_branchless(m0, m0_sw));
            }
            {
                sfpi::vFloat m1    = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[p1]));
                sfpi::vFloat m1_sw = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[32 + p1]));
                sfpi::dst_reg[sc2] = sfpi::round_bf16(sfpi::sfpi_amin_star_branchless(m1, m1_sw));
            }
            {
                sfpi::vFloat m2    = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[p2]));
                sfpi::vFloat m2_sw = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[32 + p2]));
                sfpi::dst_reg[sc3] = sfpi::round_bf16(sfpi::sfpi_amin_star_branchless(m2, m2_sw));
            }

            // Stage 4: Outgoing
            {
                sfpi::vFloat m_oth0 = sfpi::sfpi_amin_star_branchless(sfpi::vFloat(sfpi::dst_reg[sc2]), sfpi::vFloat(sfpi::dst_reg[sc3]));
                sfpi::vFloat m0_sw  = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[32 + p0]));
                sfpi::vFloat r_mag0 = sfpi::sfpi_amin_star_branchless(m_oth0, m0_sw);
                sfpi::vInt s_edge0  = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc0])) ^ sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[p0]));
                sfpi::vFloat r_out0 = sfpi::round_bf16(sfpi::copysgn(r_mag0, s_edge0));
                sfpi::dst_reg[32 + p0] = sfpi::round_bf16(sfpi::vFloat(sfpi::dst_reg[p0]) + r_out0);
                sfpi::dst_reg[p0]      = r_out0;
            }
            {
                sfpi::vFloat m_oth1 = sfpi::sfpi_amin_star_branchless(sfpi::vFloat(sfpi::dst_reg[sc1]), sfpi::vFloat(sfpi::dst_reg[sc3]));
                sfpi::vFloat m1_sw  = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[32 + p1]));
                sfpi::vFloat r_mag1 = sfpi::sfpi_amin_star_branchless(m_oth1, m1_sw);
                sfpi::vInt s_edge1  = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc0])) ^ sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[p1]));
                sfpi::vFloat r_out1 = sfpi::round_bf16(sfpi::copysgn(r_mag1, s_edge1));
                sfpi::dst_reg[32 + p1] = sfpi::round_bf16(sfpi::vFloat(sfpi::dst_reg[p1]) + r_out1);
                sfpi::dst_reg[p1]      = r_out1;
            }
            {
                sfpi::vFloat m_oth2 = sfpi::sfpi_amin_star_branchless(sfpi::vFloat(sfpi::dst_reg[sc1]), sfpi::vFloat(sfpi::dst_reg[sc2]));
                sfpi::vFloat m2_sw  = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[32 + p2]));
                sfpi::vFloat r_mag2 = sfpi::sfpi_amin_star_branchless(m_oth2, m2_sw);
                sfpi::vInt s_edge2  = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[sc0])) ^ sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[p2]));
                sfpi::vFloat r_out2 = sfpi::round_bf16(sfpi::copysgn(r_mag2, s_edge2));
                sfpi::dst_reg[32 + p2] = sfpi::round_bf16(sfpi::vFloat(sfpi::dst_reg[p2]) + r_out2);
                sfpi::dst_reg[p2]      = r_out2;
            }

            // In-flight parity of updated posterior LLRs (L_new) for Deg 6
            sfpi::vInt syn0 = sfpi::as<sfpi::vInt>(sfpi::dst_reg[32 + p0]) ^ sfpi::as<sfpi::vInt>(sfpi::swap_subvec16(sfpi::dst_reg[32 + p0]));
            sfpi::vInt syn1 = sfpi::as<sfpi::vInt>(sfpi::dst_reg[32 + p1]) ^ sfpi::as<sfpi::vInt>(sfpi::swap_subvec16(sfpi::dst_reg[32 + p1]));
            sfpi::vInt syn2 = sfpi::as<sfpi::vInt>(sfpi::dst_reg[32 + p2]) ^ sfpi::as<sfpi::vInt>(sfpi::swap_subvec16(sfpi::dst_reg[32 + p2]));
            sfpi::vInt parity_Lnew = ((syn0 ^ syn1) ^ syn2) & 0x80000000;

            // Accumulate parity into running registers
            if (sub == 0) {
                sfpi::dst_reg[30] = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(sfpi::dst_reg[30]) | parity_Lnew);
            } else {
                sfpi::dst_reg[31] = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(sfpi::dst_reg[31]) | parity_Lnew);
            }
        }
        sfpi::dst_reg[sc0] = 0.0f;
        sfpi::dst_reg[sc1] = 0.0f;
        sfpi::dst_reg[sc2] = 0.0f;
        sfpi::dst_reg[sc3] = 0.0f;
    }

    // Fold lanes 16..31 into lanes 0..15 and transfer to Tile 1 Face 1 scratch
    sfpi::dst_reg[32 + 14] = sfpi::as<sfpi::vFloat>(
        sfpi::as<sfpi::vInt>(sfpi::dst_reg[30]) | sfpi::as<sfpi::vInt>(sfpi::swap_subvec16(sfpi::dst_reg[30]))
    );
    sfpi::dst_reg[32 + 15] = sfpi::as<sfpi::vFloat>(
        sfpi::as<sfpi::vInt>(sfpi::dst_reg[31]) | sfpi::as<sfpi::vInt>(sfpi::swap_subvec16(sfpi::dst_reg[31]))
    );
    sfpi::dst_reg[30] = 0.0f;
    sfpi::dst_reg[31] = 0.0f;
}

inline void compute_tile_deg3_with_parity(uint32_t iter) {
    const int face_bases[4] = {0, 8, 16, 24};
    sfpi::dst_reg[30] = 0.0f;
    sfpi::dst_reg[31] = 0.0f;

    for (int f = 0; f < 4; ++f) {
        int base_r = face_bases[f];
        int sc0 = (f == 3) ? 6 : (base_r + 6);
        int sc1 = (f == 3) ? 7 : (base_r + 7);
        int sc2 = 32 + base_r + 6;
        int sc3 = 32 + base_r + 7;

        for (int sub = 0; sub < 2; ++sub) {
            int p0 = base_r + 0 + sub;
            int p1 = base_r + 2 + sub;
            int p2 = base_r + 4 + sub;

            if (iter != 0) {
                sfpi::dst_reg[p0] = sfpi::round_bf16(sfpi::vFloat(sfpi::dst_reg[p0]) - sfpi::vFloat(sfpi::dst_reg[32 + p0]));
                sfpi::dst_reg[p1] = sfpi::round_bf16(sfpi::vFloat(sfpi::dst_reg[p1]) - sfpi::vFloat(sfpi::dst_reg[32 + p1]));
                sfpi::dst_reg[p2] = sfpi::round_bf16(sfpi::vFloat(sfpi::dst_reg[p2]) - sfpi::vFloat(sfpi::dst_reg[32 + p2]));
            }

            sfpi::dst_reg[sc0] = sfpi::swap_subvec16(sfpi::vFloat(sfpi::dst_reg[p0]));
            sfpi::dst_reg[sc1] = sfpi::swap_subvec16(sfpi::vFloat(sfpi::dst_reg[p1]));
            sfpi::dst_reg[sc2] = sfpi::swap_subvec16(sfpi::vFloat(sfpi::dst_reg[p2]));

            {
                sfpi::vFloat q1    = sfpi::dst_reg[p1];
                sfpi::vFloat q1_sw = sfpi::dst_reg[sc1];
                sfpi::vFloat q_e2 = q1;
                v_if (sfpi::as<sfpi::vUInt>(sfpi::vConstTileId) >= 32) {
                    q_e2 = q1_sw;
                } v_endif;
                sfpi::dst_reg[sc1] = q_e2;

                sfpi::vFloat q_e0 = q1;
                v_if (sfpi::as<sfpi::vUInt>(sfpi::vConstTileId) < 32) {
                    q_e0 = q1_sw;
                } v_endif;
                sfpi::dst_reg[sc3] = q_e0;
            }

            sfpi::vInt s_tot_blk0;
            {
                sfpi::vFloat q0    = sfpi::dst_reg[p0];
                sfpi::vFloat q0_sw = sfpi::dst_reg[sc0];
                sfpi::vFloat q_e2  = sfpi::dst_reg[sc1];

                sfpi::vInt s0    = sfpi::as<sfpi::vInt>(q0);
                sfpi::vInt s0_sw = sfpi::as<sfpi::vInt>(q0_sw);
                sfpi::vInt s_e2  = sfpi::as<sfpi::vInt>(q_e2);
                s_tot_blk0       = (s0 ^ s0_sw) ^ s_e2;

                sfpi::vFloat m0    = sfpi::abs(q0);
                sfpi::vFloat m0_sw = sfpi::abs(q0_sw);
                sfpi::vFloat m_e2  = sfpi::abs(q_e2);

                sfpi::vFloat r_mag0 = sfpi::sfpi_amin_star_branchless(m0_sw, m_e2);
                sfpi::vFloat r_out0 = sfpi::round_bf16(sfpi::copysgn(r_mag0, s_tot_blk0 ^ s0));
                sfpi::dst_reg[32 + p0] = sfpi::round_bf16(q0 + r_out0);
                sfpi::dst_reg[p0]      = r_out0;

                sfpi::vFloat r_mag_e2 = sfpi::sfpi_amin_star_branchless(m0, m0_sw);
                sfpi::vFloat r_out_e2 = sfpi::round_bf16(sfpi::copysgn(r_mag_e2, s_tot_blk0 ^ s_e2));
                sfpi::dst_reg[sc1] = r_out_e2;
            }

            sfpi::vInt s_tot_blk1;
            {
                sfpi::vFloat q2    = sfpi::dst_reg[p2];
                sfpi::vFloat q2_sw = sfpi::dst_reg[sc2];
                sfpi::vFloat q_e0  = sfpi::dst_reg[sc3];

                sfpi::vInt s2    = sfpi::as<sfpi::vInt>(q2);
                sfpi::vInt s2_sw = sfpi::as<sfpi::vInt>(q2_sw);
                sfpi::vInt s_e0  = sfpi::as<sfpi::vInt>(q_e0);
                s_tot_blk1       = (s2 ^ s2_sw) ^ s_e0;

                sfpi::vFloat m2    = sfpi::abs(q2);
                sfpi::vFloat m2_sw = sfpi::abs(q2_sw);
                sfpi::vFloat m_e0  = sfpi::abs(q_e0);

                sfpi::vFloat r_mag2 = sfpi::sfpi_amin_star_branchless(m2_sw, m_e0);
                sfpi::vFloat r_out2 = sfpi::round_bf16(sfpi::copysgn(r_mag2, s_tot_blk1 ^ s2));
                sfpi::dst_reg[32 + p2] = sfpi::round_bf16(q2 + r_out2);
                sfpi::dst_reg[p2]      = r_out2;

                sfpi::vFloat r_mag_e0 = sfpi::sfpi_amin_star_branchless(m2, m2_sw);
                sfpi::vFloat r_out_e0 = sfpi::round_bf16(sfpi::copysgn(r_mag_e0, s_tot_blk1 ^ s_e0));
                sfpi::dst_reg[sc3] = r_out_e0;
            }

            {
                sfpi::vFloat r_out1 = sfpi::dst_reg[sc1];
                v_if (sfpi::as<sfpi::vUInt>(sfpi::vConstTileId) >= 32) {
                    r_out1 = sfpi::vFloat(sfpi::dst_reg[sc3]);
                } v_endif;
                sfpi::dst_reg[32 + p1] = sfpi::round_bf16(sfpi::vFloat(sfpi::dst_reg[p1]) + r_out1);
                sfpi::dst_reg[p1]      = r_out1;
            }

            // In-flight parity of updated posterior LLRs (L_new) for Deg 3
            sfpi::vInt syn0_deg3 = sfpi::as<sfpi::vInt>(sfpi::dst_reg[32 + p0]) ^ sfpi::as<sfpi::vInt>(sfpi::swap_subvec16(sfpi::dst_reg[32 + p0]));
            sfpi::vInt syn2_deg3 = sfpi::as<sfpi::vInt>(sfpi::dst_reg[32 + p2]) ^ sfpi::as<sfpi::vInt>(sfpi::swap_subvec16(sfpi::dst_reg[32 + p2]));
            sfpi::vInt syn_l1    = sfpi::as<sfpi::vInt>(sfpi::dst_reg[32 + p1]);

            sfpi::vInt parity_blk0 = syn0_deg3 ^ syn_l1;
            sfpi::vInt parity_blk1 = syn2_deg3 ^ syn_l1;

            sfpi::vInt parity_deg3 = parity_blk0;
            v_if (sfpi::as<sfpi::vUInt>(sfpi::vConstTileId) >= 32) {
                parity_deg3 = parity_blk1;
            } v_endif;
            parity_deg3 = parity_deg3 & 0x80000000;

            if (sub == 0) {
                sfpi::dst_reg[30] = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(sfpi::dst_reg[30]) | parity_deg3);
            } else {
                sfpi::dst_reg[31] = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(sfpi::dst_reg[31]) | parity_deg3);
            }
        }
        sfpi::dst_reg[sc0] = 0.0f;
        sfpi::dst_reg[sc1] = 0.0f;
        sfpi::dst_reg[sc2] = 0.0f;
        sfpi::dst_reg[sc3] = 0.0f;
    }

    // Fold lanes 16..31 into lanes 0..15 and transfer to Tile 1 Face 1 scratch
    sfpi::dst_reg[32 + 14] = sfpi::as<sfpi::vFloat>(
        sfpi::as<sfpi::vInt>(sfpi::dst_reg[30]) | sfpi::as<sfpi::vInt>(sfpi::swap_subvec16(sfpi::dst_reg[30]))
    );
    sfpi::dst_reg[32 + 15] = sfpi::as<sfpi::vFloat>(
        sfpi::as<sfpi::vInt>(sfpi::dst_reg[31]) | sfpi::as<sfpi::vInt>(sfpi::swap_subvec16(sfpi::dst_reg[31]))
    );
    sfpi::dst_reg[30] = 0.0f;
    sfpi::dst_reg[31] = 0.0f;
}

int main() {
    std::cout << "=========================================================" << std::endl;
    std::cout << " TEST IN-FLIGHT TRISC CHECK NODE PARITY ACCUMULATION" << std::endl;
    std::cout << "=========================================================" << std::endl;

    ParityCheckMatrix pcm;
    if (!load_chinn_matrix("matrices/AR4JA_r45_4c_128c_r12.chinn.out", pcm)) {
        std::cerr << "Failed to load matrix!" << std::endl;
        return 1;
    }
    auto schedule = build_40tile_schedule(pcm);
    std::cout << "[INFO] Loaded matrix (M=" << pcm.M << ", N=" << pcm.N << "), scheduled " << schedule.size() << " tiles." << std::endl;

    // Load initial Vnodes from real batch 777594
    std::string init_path = "engines/tile_matrix_4cw/dump/tensix_initial_vnodes.bin";
    std::ifstream f_init(init_path, std::ios::binary);
    if (!f_init.is_open()) {
        std::cerr << "Could not open " << init_path << std::endl;
        return 1;
    }
    std::vector<uint8_t> vnode_l1(20480);
    f_init.read(reinterpret_cast<char*>(vnode_l1.data()), 20480);
    f_init.close();
    std::cout << "[INFO] Loaded 20,480 bytes from " << init_path << std::endl;

    std::vector<std::vector<uint16_t>> r_msg_fifo(schedule.size(), std::vector<uint16_t>(1024, 0));
    std::vector<uint16_t> scratch_in(1024, 0);
    std::vector<uint16_t> scratch_out(1024, 0);

    const uint32_t* p_hoff_base = pcm.h_offsets_pairs.data();

    uint32_t mismatches = 0;

    for (uint32_t iter = 0; iter < 16; ++iter) {
        uint32_t iter_unsat_cw01 = 0;
        uint32_t iter_unsat_cw23 = 0;
        uint32_t gather_err_cw01 = 0;
        uint32_t gather_err_cw23 = 0;
        uint32_t scatter_err_cw01 = 0;
        uint32_t scatter_err_cw23 = 0;

        for (size_t t = 0; t < schedule.size(); ++t) {
            uint32_t deg = schedule[t].check_degree;
            const uint32_t* tile_hoff_w0 = p_hoff_base + t * 96;
            const uint32_t* tile_hoff_w1 = tile_hoff_w0 + 48;

            // Gather
            for (int p = 0; p < 48; ++p) {
                uint32_t off0 = tile_hoff_w0[p];
                uint32_t off0_e = off0 & 0xFFFF;
                uint32_t off0_o = off0 >> 16;
                uint64_t val0_e = *reinterpret_cast<const uint64_t*>(vnode_l1.data() + off0_e);
                uint64_t val0_o = *reinterpret_cast<const uint64_t*>(vnode_l1.data() + off0_o);
                *reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(scratch_in.data()) + SCRATCH_FACE_LUT.w0_even[p]) = val0_e;
                *reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(scratch_in.data()) + SCRATCH_FACE_LUT.w0_odd[p])  = val0_o;

                if (off0_e < 16384) {
                    gather_err_cw01 |= (static_cast<uint32_t>(val0_e) & 0x80008000);
                    gather_err_cw23 |= (static_cast<uint32_t>(val0_e >> 32) & 0x80008000);
                }
                if (off0_o < 16384) {
                    gather_err_cw01 |= (static_cast<uint32_t>(val0_o) & 0x80008000);
                    gather_err_cw23 |= (static_cast<uint32_t>(val0_o >> 32) & 0x80008000);
                }

                uint32_t off1 = tile_hoff_w1[p];
                uint32_t off1_e = off1 & 0xFFFF;
                uint32_t off1_o = off1 >> 16;
                uint64_t val1_e = *reinterpret_cast<const uint64_t*>(vnode_l1.data() + off1_e);
                uint64_t val1_o = *reinterpret_cast<const uint64_t*>(vnode_l1.data() + off1_o);
                *reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(scratch_in.data()) + SCRATCH_FACE_LUT.w1_even[p]) = val1_e;
                *reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(scratch_in.data()) + SCRATCH_FACE_LUT.w1_odd[p])  = val1_o;

                if (off1_e < 16384) {
                    gather_err_cw01 |= (static_cast<uint32_t>(val1_e) & 0x80008000);
                    gather_err_cw23 |= (static_cast<uint32_t>(val1_e >> 32) & 0x80008000);
                }
                if (off1_o < 16384) {
                    gather_err_cw01 |= (static_cast<uint32_t>(val1_o) & 0x80008000);
                    gather_err_cw23 |= (static_cast<uint32_t>(val1_o >> 32) & 0x80008000);
                }
            }

            // TRISC Step
            for (int r = 0; r < 32; ++r) {
                for (uint32_t lane = 0; lane < 32; ++lane) {
                    uint32_t off = get_scratch_byte_offset(r, lane);
                    sfpi::dst_reg[r][lane]      = bf16_to_fp32(*reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(scratch_in.data()) + off));
                    sfpi::dst_reg[32 + r][lane] = bf16_to_fp32(*reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(r_msg_fifo[t].data()) + off));
                }
            }

            if (deg == 6) compute_tile_deg6_with_parity(iter, t, &pcm, &schedule, vnode_l1.data());
            else          compute_tile_deg3_with_parity(iter);

            // Pack
            for (int r = 0; r < 32; ++r) {
                for (uint32_t lane = 0; lane < 32; ++lane) {
                    uint32_t off = get_scratch_byte_offset(r, lane);
                    *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(r_msg_fifo[t].data()) + off) = fp32_to_bf16(sfpi::dst_reg[r][lane]);
                    *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(scratch_out.data()) + off)   = fp32_to_bf16(sfpi::dst_reg[32 + r][lane]);
                }
            }

            // Scatter
            for (int p = 0; p < 48; ++p) {
                uint32_t off0 = tile_hoff_w0[p];
                uint32_t off0_e = off0 & 0xFFFF;
                uint32_t off0_o = off0 >> 16;
                uint64_t sval0_e = *reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(scratch_out.data()) + SCRATCH_FACE_LUT.w0_even[p]);
                uint64_t sval0_o = *reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(scratch_out.data()) + SCRATCH_FACE_LUT.w0_odd[p]);
                *reinterpret_cast<uint64_t*>(vnode_l1.data() + off0_e) = sval0_e;
                *reinterpret_cast<uint64_t*>(vnode_l1.data() + off0_o) = sval0_o;

                if (off0_e < 16384) {
                    scatter_err_cw01 |= (static_cast<uint32_t>(sval0_e) & 0x80008000);
                    scatter_err_cw23 |= (static_cast<uint32_t>(sval0_e >> 32) & 0x80008000);
                }
                if (off0_o < 16384) {
                    scatter_err_cw01 |= (static_cast<uint32_t>(sval0_o) & 0x80008000);
                    scatter_err_cw23 |= (static_cast<uint32_t>(sval0_o >> 32) & 0x80008000);
                }

                uint32_t off1 = tile_hoff_w1[p];
                uint32_t off1_e = off1 & 0xFFFF;
                uint32_t off1_o = off1 >> 16;
                uint64_t sval1_e = *reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(scratch_out.data()) + SCRATCH_FACE_LUT.w1_even[p]);
                uint64_t sval1_o = *reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(scratch_out.data()) + SCRATCH_FACE_LUT.w1_odd[p]);
                *reinterpret_cast<uint64_t*>(vnode_l1.data() + off1_e) = sval1_e;
                *reinterpret_cast<uint64_t*>(vnode_l1.data() + off1_o) = sval1_o;

                if (off1_e < 16384) {
                    scatter_err_cw01 |= (static_cast<uint32_t>(sval1_e) & 0x80008000);
                    scatter_err_cw23 |= (static_cast<uint32_t>(sval1_e >> 32) & 0x80008000);
                }
                if (off1_o < 16384) {
                    scatter_err_cw01 |= (static_cast<uint32_t>(sval1_o) & 0x80008000);
                    scatter_err_cw23 |= (static_cast<uint32_t>(sval1_o >> 32) & 0x80008000);
                }
            }

            // NCRISC In-Flight Parity Read: Read rows 12 & 13 of Face 1 (byte offsets 896 and 928)
            const auto* p_unsat0 = reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(scratch_out.data()) + 896);
            iter_unsat_cw01 |= (p_unsat0[0] | p_unsat0[2] | p_unsat0[4] | p_unsat0[6]);
            iter_unsat_cw23 |= (p_unsat0[1] | p_unsat0[3] | p_unsat0[5] | p_unsat0[7]);

            const auto* p_unsat1 = reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(scratch_out.data()) + 928);
            uint32_t t_cw01 = (p_unsat0[0] | p_unsat0[2] | p_unsat0[4] | p_unsat0[6] | p_unsat1[0] | p_unsat1[2] | p_unsat1[4] | p_unsat1[6]);
            iter_unsat_cw01 |= t_cw01;
            iter_unsat_cw23 |= (p_unsat0[1] | p_unsat0[3] | p_unsat0[5] | p_unsat0[7] | p_unsat1[1] | p_unsat1[3] | p_unsat1[5] | p_unsat1[7]);
        }

        bool gather_conv[4] = {
            (gather_err_cw01 & 0x00008000) == 0,
            (gather_err_cw01 & 0x80000000) == 0,
            (gather_err_cw23 & 0x00008000) == 0,
            (gather_err_cw23 & 0x80000000) == 0
        };

        bool scatter_conv[4] = {
            (scatter_err_cw01 & 0x00008000) == 0,
            (scatter_err_cw01 & 0x80000000) == 0,
            (scatter_err_cw23 & 0x00008000) == 0,
            (scatter_err_cw23 & 0x80000000) == 0
        };

        // Method A: In-flight Parity from TRISC
        bool stot_conv[4] = {
            (iter_unsat_cw01 & 0x00008000) == 0,
            (iter_unsat_cw01 & 0x80000000) == 0,
            (iter_unsat_cw23 & 0x00008000) == 0,
            (iter_unsat_cw23 & 0x80000000) == 0
        };

        // Method B: Ground Truth Syndrome across all 1536 check nodes
        bool syn_conv[4] = {true, true, true, true};
        uint32_t unsat_count[4] = {0, 0, 0, 0};
        for (uint32_t m = 0; m < pcm.M; ++m) {
            int syn[4] = {0, 0, 0, 0};
            for (uint32_t off : pcm.check_to_var_offsets[m]) {
                const uint16_t* p = reinterpret_cast<const uint16_t*>(vnode_l1.data() + off);
                for (int c = 0; c < 4; ++c) {
                    if (bf16_to_fp32(p[c]) < 0.0f) syn[c] ^= 1;
                }
            }
            for (int c = 0; c < 4; ++c) {
                if (syn[c] != 0) {
                    syn_conv[c] = false;
                    unsat_count[c]++;
                }
            }
        }

        uint32_t info_errs[4] = {0, 0, 0, 0};
        for (uint32_t v = 0; v < 1024; ++v) {
            uint32_t off = get_tile_vnode_byte_offset(v);
            const uint16_t* p = reinterpret_cast<const uint16_t*>(vnode_l1.data() + off);
            for (int c = 0; c < 4; ++c) {
                if (bf16_to_fp32(p[c]) < 0.0f) info_errs[c]++;
            }
        }

        uint32_t tx_neg[4] = {0, 0, 0, 0};
        for (uint32_t v = 0; v < 2048; ++v) {
            uint32_t off = get_tile_vnode_byte_offset(v);
            const uint16_t* p = reinterpret_cast<const uint16_t*>(vnode_l1.data() + off);
            for (int c = 0; c < 4; ++c) {
                if (bf16_to_fp32(p[c]) < 0.0f) tx_neg[c]++;
            }
        }

        std::cout << "[Iter " << std::setw(2) << (iter + 1) << "] "
                  << "Gather: [" << gather_conv[0] << "," << gather_conv[1] << "," << gather_conv[2] << "," << gather_conv[3] << "] | "
                  << "Scatter: [" << scatter_conv[0] << "," << scatter_conv[1] << "," << scatter_conv[2] << "," << scatter_conv[3] << "] | "
                  << "TrueSyn: [" << syn_conv[0]  << "," << syn_conv[1]  << "," << syn_conv[2]  << "," << syn_conv[3]  << "] | "
                  << "TxNeg: [" << tx_neg[0] << "," << tx_neg[1] << "," << tx_neg[2] << "," << tx_neg[3] << "] | "
                  << "InfoBE: [" << info_errs[0] << "," << info_errs[1] << "," << info_errs[2] << "," << info_errs[3] << "]"
                  << std::endl;

        for (int c = 0; c < 4; ++c) {
            if (stot_conv[c] != syn_conv[c]) {
                std::cerr << ">>> [MISMATCH] Iter " << (iter + 1) << " CW" << c 
                          << ": TRISC=" << stot_conv[c] << " vs TrueSyndrome=" << syn_conv[c] << std::endl;
                mismatches++;
            }
        }
    }

    if (mismatches == 0) {
        std::cout << "\n>>> [VERIFICATION SUCCESS] 100% BIT-EXACT MATCH BETWEEN IN-FLIGHT TRISC PARITY AND FULL 1536-CHECK SYNDROME!" << std::endl;
    } else {
        std::cerr << "\n>>> [VERIFICATION FAILED] " << mismatches << " mismatches detected!" << std::endl;
    }

    return (mismatches == 0) ? 0 : 1;
}
