#include <iostream>
#include <vector>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <filesystem>

#include "../engines/tile_matrix_4cw/ldpc_tile_matrix_core.h"

// ==============================================================================
// Unit Test: 128-Bit Fused Vector Data Movement Engine Co-Verification
// Validates gather_face_128bit and scatter_face_128bit bit-exact equivalence
// against golden 64-bit scalar engines across all 40 tiles of AR4JA Rate-1/2
// ==============================================================================

struct Edge { uint32_t r, c; float val; };

struct ParityCheckMatrix {
    uint32_t M = 0;
    uint32_t N = 0;
    uint32_t circulant_p = 0;
    std::vector<std::vector<uint32_t>> check_to_var;
    std::vector<uint32_t> h_offsets_pairs;
};

struct DynamicTileDescriptor {
    std::vector<uint32_t> check_indices;
    uint8_t  check_degree;
    uint8_t  num_checks;
    uint16_t num_edges;
};

static bool parse_chinn_out(const std::string& filepath, ParityCheckMatrix& pcm) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Could not open matrix file: " << filepath << std::endl;
        return false;
    }

    std::vector<Edge> raw_lines;
    raw_lines.reserve(100000);
    std::string line;
    while (std::getline(file, line)) {
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos || line[start] == '#') continue;
        std::stringstream ss(line);
        uint32_t r = 0, c = 0; float val = 0.0f;
        if (ss >> r >> c >> val) raw_lines.push_back({r, c, val});
    }
    if (raw_lines.empty()) return false;

    Edge dim_line = raw_lines.back();
    pcm.M = dim_line.r;
    pcm.N = dim_line.c;
    pcm.circulant_p = pcm.M / 3;
    pcm.check_to_var.resize(pcm.M);

    for (size_t i = 0; i + 1 < raw_lines.size(); i++) {
        const auto& edge = raw_lines[i];
        if (std::abs(edge.val) > 1e-6f) {
            pcm.check_to_var[edge.r - 1].push_back(edge.c - 1);
        }
    }
    return true;
}

static std::vector<DynamicTileDescriptor> build_schedule(const ParityCheckMatrix& pcm) {
    std::vector<DynamicTileDescriptor> sched;
    // 32 tiles of degree 6 (checks 0..1023)
    for (uint32_t t = 0; t < 32; ++t) {
        DynamicTileDescriptor desc;
        desc.check_degree = 6;
        desc.num_checks = 32;
        desc.num_edges = 32 * 6;
        for (uint32_t i = 0; i < 32; ++i) desc.check_indices.push_back(t * 32 + i);
        sched.push_back(desc);
    }
    // 8 tiles of degree 3 (checks 1024..1535, pairs of 32)
    for (uint32_t t = 0; t < 8; ++t) {
        DynamicTileDescriptor desc;
        desc.check_degree = 3;
        desc.num_checks = 32;
        desc.num_edges = 32 * 3;
        // 16 checks from block 0 (1024 + t*16) and 16 checks from block 1 (1024 + 128 + t*16)
        for (uint32_t i = 0; i < 16; ++i) desc.check_indices.push_back(1024 + t * 16 + i);
        for (uint32_t i = 0; i < 16; ++i) desc.check_indices.push_back(1024 + 128 + t * 16 + i);
        sched.push_back(desc);
    }
    return sched;
}

static void pack_h_offsets(ParityCheckMatrix& pcm, const std::vector<DynamicTileDescriptor>& sched) {
    pcm.h_offsets_pairs.assign(sched.size() * 96, 0);

    for (size_t t = 0; t < sched.size(); ++t) {
        const auto& tile = sched[t];
        uint32_t* tile_hoff = pcm.h_offsets_pairs.data() + t * 96;

        for (uint32_t f = 0; f < 4; ++f) {
            uint32_t* face_hoff = tile_hoff + f * 24;
            uint32_t c_start = (f < 2) ? 0 : 16; // Rows 0..15 vs 16..31
            uint32_t e_start = (f % 2 == 0) ? 0 : 3; // Edges 0..2 vs 3..5

            for (uint32_t d = 0; d < 6; ++d) {
                uint32_t* deg_hoff = face_hoff + d * 4;
                for (uint32_t p = 0; p < 4; ++p) {
                    uint32_t c0 = c_start + 4 * (d / 2) + 2 * (p / 2) + 0;
                    uint32_t c1 = c_start + 4 * (d / 2) + 2 * (p / 2) + 1;
                    uint32_t e_idx = e_start + (d % 2) * 2 + (p % 2);

                    uint32_t v0 = 0, v1 = 0;
                    if (c0 < tile.check_indices.size()) {
                        uint32_t ch0 = tile.check_indices[c0];
                        if (e_idx < pcm.check_to_var[ch0].size()) v0 = pcm.check_to_var[ch0][e_idx];
                    }
                    if (c1 < tile.check_indices.size()) {
                        uint32_t ch1 = tile.check_indices[c1];
                        if (e_idx < pcm.check_to_var[ch1].size()) v1 = pcm.check_to_var[ch1][e_idx];
                    }

                    // Pack into 32-bit offset word: [15:0] = v0 * 8, [31:16] = v1 * 8
                    deg_hoff[p] = ((v0 * 8) & 0xFFFF) | (((v1 * 8) & 0xFFFF) << 16);
                }
            }
        }
    }
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << " 128-BIT VECTOR GATHER & SCATTER ENGINE CO-VERIFICATION SUITE" << std::endl;
    std::cout << " Testing 128-Bit Fused Vector Implementation vs Golden Reference" << std::endl;
    std::cout << "=================================================================" << std::endl;

    std::string matrix_path = "matrices/AR4JA_r45_4c_128c_r12.chinn.out";
    if (!std::filesystem::exists(matrix_path)) {
        matrix_path = "../matrices/AR4JA_r45_4c_128c_r12.chinn.out";
    }

    ParityCheckMatrix pcm;
    if (!parse_chinn_out(matrix_path, pcm)) {
        std::cerr << "[FAIL] Could not load AR4JA matrix from " << matrix_path << std::endl;
        return 1;
    }
    std::cout << "[INFO] Loaded AR4JA matrix: N=" << pcm.N << ", M=" << pcm.M << std::endl;

    auto schedule = build_schedule(pcm);
    pack_h_offsets(pcm, schedule);
    std::cout << "[INFO] Generated 40 circulant-aware tiles with pre-shifted H-table." << std::endl;

    // Allocate variable nodes:
    // Batch A: 2560 x 8 bytes (CW 0..3)
    // Batch B: 2560 x 8 bytes (CW 4..7)
    // Unified: 2560 x 16 bytes (CW 0..7)
    std::vector<uint64_t> vnodes_4cw_a(pcm.N);
    std::vector<uint64_t> vnodes_4cw_b(pcm.N);
    std::vector<VNode128>  vnodes_128(pcm.N);

    // Populate with deterministic non-trivial data
    for (uint32_t v = 0; v < pcm.N; ++v) {
        uint16_t cw0 = static_cast<uint16_t>(1000 + v * 4 + 0);
        uint16_t cw1 = static_cast<uint16_t>(1000 + v * 4 + 1);
        uint16_t cw2 = static_cast<uint16_t>(1000 + v * 4 + 2);
        uint16_t cw3 = static_cast<uint16_t>(1000 + v * 4 + 3);

        uint16_t cw4 = static_cast<uint16_t>(5000 + v * 4 + 0);
        uint16_t cw5 = static_cast<uint16_t>(5000 + v * 4 + 1);
        uint16_t cw6 = static_cast<uint16_t>(5000 + v * 4 + 2);
        uint16_t cw7 = static_cast<uint16_t>(5000 + v * 4 + 3);

        uint64_t val_a = (static_cast<uint64_t>(cw0)) |
                         (static_cast<uint64_t>(cw1) << 16) |
                         (static_cast<uint64_t>(cw2) << 32) |
                         (static_cast<uint64_t>(cw3) << 48);

        uint64_t val_b = (static_cast<uint64_t>(cw4)) |
                         (static_cast<uint64_t>(cw5) << 16) |
                         (static_cast<uint64_t>(cw6) << 32) |
                         (static_cast<uint64_t>(cw7) << 48);

        vnodes_4cw_a[v] = val_a;
        vnodes_4cw_b[v] = val_b;
        vnodes_128[v].set(val_a, val_b);
    }

    std::cout << "[INFO] Initialized 2,560 variable nodes across 8 codewords." << std::endl;

    // Scratch buffers for 1 tile (4 faces x 512 B = 2048 B)
    alignas(16) uint8_t scratch_gold_a[2048];
    alignas(16) uint8_t scratch_gold_b[2048];
    alignas(16) uint8_t scratch_128_a[2048];
    alignas(16) uint8_t scratch_128_b[2048];

    uint32_t total_gather_mismatches = 0;
    uint32_t total_scatter_mismatches = 0;

    // Test Gather across all 40 tiles
    std::cout << "\n>>> [TEST 1] Verifying gather_face_128bit across all 40 tiles..." << std::endl;
    for (size_t t = 0; t < schedule.size(); ++t) {
        std::memset(scratch_gold_a, 0xAA, sizeof(scratch_gold_a));
        std::memset(scratch_gold_b, 0xBB, sizeof(scratch_gold_b));
        std::memset(scratch_128_a, 0xCC, sizeof(scratch_128_a));
        std::memset(scratch_128_b, 0xDD, sizeof(scratch_128_b));

        const uint32_t* tile_hoff = pcm.h_offsets_pairs.data() + t * 96;

        for (uint32_t f = 0; f < 4; ++f) {
            uint32_t face_offset = f * 512;
            const uint32_t* p_hoff_24 = tile_hoff + f * 24;

            // Golden 64-bit scalar gathers
            gather_face_24pairs(reinterpret_cast<uintptr_t>(scratch_gold_a + face_offset),
                                p_hoff_24,
                                reinterpret_cast<const char*>(vnodes_4cw_a.data()));

            gather_face_24pairs(reinterpret_cast<uintptr_t>(scratch_gold_b + face_offset),
                                p_hoff_24,
                                reinterpret_cast<const char*>(vnodes_4cw_b.data()));

            // Fused 128-bit vector gather (offset_shift = 1 because H table has v * 8)
            gather_face_128bit(reinterpret_cast<uintptr_t>(scratch_128_a + face_offset),
                               reinterpret_cast<uintptr_t>(scratch_128_b + face_offset),
                               p_hoff_24,
                               reinterpret_cast<const char*>(vnodes_128.data()),
                               1);

            // Compare face buffers
            int cmp_a = std::memcmp(scratch_gold_a + face_offset, scratch_128_a + face_offset, 512);
            int cmp_b = std::memcmp(scratch_gold_b + face_offset, scratch_128_b + face_offset, 512);
            if (cmp_a != 0 || cmp_b != 0) {
                std::cerr << "[ERROR] Gather mismatch on Tile " << t << ", Face " << f << "!" << std::endl;
                total_gather_mismatches++;
            }
        }
    }

    if (total_gather_mismatches == 0) {
        std::cout << ">>> [PASS] gather_face_128bit: 100% BIT-EXACT across all 40 tiles (160 faces, 81,920 bytes)!" << std::endl;
    } else {
        std::cerr << ">>> [FAIL] gather_face_128bit had " << total_gather_mismatches << " face mismatches!" << std::endl;
        return 1;
    }

    // Test Scatter across all 40 tiles
    std::cout << "\n>>> [TEST 2] Verifying scatter_face_128bit across all 40 tiles..." << std::endl;

    // Create unique update patterns in scratch buffers
    for (size_t i = 0; i < 2048; ++i) {
        scratch_128_a[i] = static_cast<uint8_t>((i * 7 + 13) & 0xFF);
        scratch_128_b[i] = static_cast<uint8_t>((i * 11 + 29) & 0xFF);
    }
    std::memcpy(scratch_gold_a, scratch_128_a, 2048);
    std::memcpy(scratch_gold_b, scratch_128_b, 2048);

    std::vector<uint64_t> vnodes_gold_out_a(pcm.N, 0);
    std::vector<uint64_t> vnodes_gold_out_b(pcm.N, 0);
    std::vector<VNode128>  vnodes_128_out(pcm.N);
    std::memset(vnodes_128_out.data(), 0, vnodes_128_out.size() * sizeof(VNode128));

    for (size_t t = 0; t < schedule.size(); ++t) {
        const uint32_t* tile_hoff = pcm.h_offsets_pairs.data() + t * 96;

        for (uint32_t f = 0; f < 4; ++f) {
            uint32_t face_offset = f * 512;
            const uint32_t* p_hoff_24 = tile_hoff + f * 24;

            // Golden 64-bit scalar scatters
            scatter_face_24pairs_fast(reinterpret_cast<uintptr_t>(scratch_gold_a + face_offset),
                                      p_hoff_24,
                                      reinterpret_cast<char*>(vnodes_gold_out_a.data()));

            scatter_face_24pairs_fast(reinterpret_cast<uintptr_t>(scratch_gold_b + face_offset),
                                      p_hoff_24,
                                      reinterpret_cast<char*>(vnodes_gold_out_b.data()));

            // Fused 128-bit vector scatter
            scatter_face_128bit(reinterpret_cast<uintptr_t>(scratch_128_a + face_offset),
                                reinterpret_cast<uintptr_t>(scratch_128_b + face_offset),
                                p_hoff_24,
                                reinterpret_cast<char*>(vnodes_128_out.data()),
                                1);

            // Verify bit-exact consistency of mutated vnodes
            for (uint32_t v = 0; v < pcm.N; ++v) {
                if (vnodes_128_out[v].batch_a() != vnodes_gold_out_a[v] ||
                    vnodes_128_out[v].batch_b() != vnodes_gold_out_b[v]) {
                    std::cerr << "[ERROR] Scatter mismatch on Tile " << t << ", Face " << f << ", Vnode " << v << "!" << std::endl;
                    total_scatter_mismatches++;
                    break;
                }
            }
        }
    }

    if (total_scatter_mismatches == 0) {
        std::cout << ">>> [PASS] scatter_face_128bit: 100% BIT-EXACT across all 40 tiles (160 faces)!" << std::endl;
    } else {
        std::cerr << ">>> [FAIL] scatter_face_128bit had " << total_scatter_mismatches << " face mismatches!" << std::endl;
        return 1;
    }

    // Test 2-Face 48-Pair Helpers (Worker 0: Faces 0 & 2, Worker 1: Faces 1 & 3)
    std::cout << "\n>>> [TEST 3] Verifying 2-Face Helper Functions (Worker 0 & Worker 1)..." << std::endl;
    uint32_t helper_mismatches = 0;
    for (size_t t = 0; t < schedule.size(); ++t) {
        const uint32_t* tile_hoff = pcm.h_offsets_pairs.data() + t * 96;

        // Reset buffers
        std::memset(scratch_128_a, 0, 2048);
        std::memset(scratch_128_b, 0, 2048);
        std::memset(scratch_gold_a, 0, 2048);
        std::memset(scratch_gold_b, 0, 2048);

        // Worker 0 gathers Faces 0 & 2
        gather_face_48pairs_w0_128bit(reinterpret_cast<uintptr_t>(scratch_128_a),
                                      reinterpret_cast<uintptr_t>(scratch_128_b),
                                      tile_hoff + 0,
                                      reinterpret_cast<const char*>(vnodes_128.data()),
                                      1);

        gather_face_48pairs_w0(reinterpret_cast<uintptr_t>(scratch_gold_a),
                               tile_hoff + 0,
                               reinterpret_cast<const char*>(vnodes_4cw_a.data()));
        gather_face_48pairs_w0(reinterpret_cast<uintptr_t>(scratch_gold_b),
                               tile_hoff + 0,
                               reinterpret_cast<const char*>(vnodes_4cw_b.data()));

        // Worker 1 gathers Faces 1 & 3
        gather_face_48pairs_w1_128bit(reinterpret_cast<uintptr_t>(scratch_128_a),
                                      reinterpret_cast<uintptr_t>(scratch_128_b),
                                      tile_hoff + 48,
                                      reinterpret_cast<const char*>(vnodes_128.data()),
                                      1);

        gather_face_48pairs_w1(reinterpret_cast<uintptr_t>(scratch_gold_a),
                               tile_hoff + 48,
                               reinterpret_cast<const char*>(vnodes_4cw_a.data()));
        gather_face_48pairs_w1(reinterpret_cast<uintptr_t>(scratch_gold_b),
                               tile_hoff + 48,
                               reinterpret_cast<const char*>(vnodes_4cw_b.data()));

        if (std::memcmp(scratch_gold_a, scratch_128_a, 2048) != 0 ||
            std::memcmp(scratch_gold_b, scratch_128_b, 2048) != 0) {
            std::cerr << "[ERROR] 48-Pair gather mismatch on Tile " << t << "!" << std::endl;
            helper_mismatches++;
        }
    }

    if (helper_mismatches == 0) {
        std::cout << ">>> [PASS] gather_face_48pairs_w0/w1_128bit: 100% BIT-EXACT across all 40 tiles!" << std::endl;
    } else {
        std::cerr << ">>> [FAIL] 48-Pair gather had " << helper_mismatches << " mismatches!" << std::endl;
        return 1;
    }

    std::cout << "\n=================================================================" << std::endl;
    std::cout << " ALL 128-BIT VECTOR GATHER/SCATTER VERIFICATION TESTS PASSED (0 MISMATCHES)!" << std::endl;
    std::cout << "=================================================================" << std::endl;

    return 0;
}
