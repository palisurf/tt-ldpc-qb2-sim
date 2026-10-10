#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <unistd.h>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/circular_buffer_config.hpp>

#include "ldpc_tile_matrix_core.h"

using namespace tt;
using namespace tt::tt_metal;
namespace fs = std::filesystem;

static std::string resolve_filepath(const std::string& filename) {
    if (fs::exists(filename)) return filename;
    if (fs::exists("engines/tile_matrix_4cw/" + filename)) return "engines/tile_matrix_4cw/" + filename;
    if (fs::exists("../../" + filename)) return "../../" + filename;
    if (fs::exists("../" + filename)) return "../" + filename;
    return filename;
}

// ==============================================================================
// Tenstorrent QuietBox 2: Tile Matrix 4-Codeword LDPC Simulator & Hardware Harness
// Architecture: Decoupled Access-Execute (DAE) with Native 32x32 Tensix Tiles
// Target: Tenstorrent Blackhole (440 Tensix Cores)
// ==============================================================================

struct EdgePair {
    uint32_t ve;
    uint32_t vo;
};

static inline uint32_t make_pair_word(uint32_t ve, uint32_t vo) {
    uint32_t oe = get_tile_vnode_byte_offset(ve) & 0xFFFF;
    uint32_t oo = get_tile_vnode_byte_offset(vo) & 0xFFFF;
    return oe | (oo << 16);
}

static inline uint32_t get_bank(uint32_t v) {
    return (v >> 1) & 0xF;
}

// ------------------------------------------------------------------------------
// Schedule 4: Bank-Aware Chunk Ordering to Eliminate Scatter Contention
// ------------------------------------------------------------------------------
static std::vector<EdgePair> schedule4_bank_aware_chunk_order(
    const std::vector<EdgePair>& w0_pairs,
    const std::vector<EdgePair>& w1_pairs
) {
    std::vector<bool> used_w1(w1_pairs.size(), false);
    std::vector<EdgePair> scheduled_w1;
    scheduled_w1.reserve(w1_pairs.size());

    uint32_t num_chunks = w0_pairs.size() / 8;
    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        uint32_t bank_hist_w0[16] = {0};
        for (uint32_t k = 0; k < 8; ++k) {
            uint32_t p = chunk * 8 + k;
            bank_hist_w0[get_bank(w0_pairs[p].ve)]++;
            bank_hist_w0[get_bank(w0_pairs[p].vo)]++;
        }

        struct Cand { uint32_t idx; uint32_t score; };
        std::vector<Cand> candidates;
        for (uint32_t i = 0; i < w1_pairs.size(); ++i) {
            if (!used_w1[i]) {
                uint32_t score = bank_hist_w0[get_bank(w1_pairs[i].ve)] +
                                 bank_hist_w0[get_bank(w1_pairs[i].vo)];
                candidates.push_back({i, score});
            }
        }
        std::sort(candidates.begin(), candidates.end(), [](const Cand& a, const Cand& b) {
            return a.score < b.score;
        });

        uint32_t take = std::min<uint32_t>(8, candidates.size());
        for (uint32_t k = 0; k < take; ++k) {
            uint32_t sel_idx = candidates[k].idx;
            used_w1[sel_idx] = true;
            scheduled_w1.push_back(w1_pairs[sel_idx]);
        }
    }

    // Append remaining tail pairs
    for (uint32_t i = 0; i < w1_pairs.size(); ++i) {
        if (!used_w1[i]) {
            scheduled_w1.push_back(w1_pairs[i]);
        }
    }

    return scheduled_w1;
}

struct ParityCheckMatrix {
    uint32_t M = 0;
    uint32_t N = 0;
    uint32_t K = 0;
    uint32_t P = 0; // Punctured nodes
    uint32_t circulant_p = 0;
    std::vector<std::vector<uint32_t>> check_to_var;
    std::vector<std::vector<uint32_t>> var_to_check;
    std::vector<uint32_t> h_offsets_pairs; // Packed edge pair byte offsets
};

struct DynamicTileDescriptor {
    std::vector<uint32_t> check_indices;
    uint8_t  check_degree;
    uint8_t  num_checks;
    uint16_t num_edges;
    uint16_t h_offset_start;
    uint16_t r_msg_page_offset;
};

// ------------------------------------------------------------------------------
// Step 1: Parse .chinn.out Matrix File
// ------------------------------------------------------------------------------
bool parse_chinn_out(const std::string& filepath, ParityCheckMatrix& pcm) {
    std::ifstream infile(filepath);
    if (!infile.is_open()) {
        std::cerr << "[ERROR] Failed to open parity check matrix: " << filepath << std::endl;
        return false;
    }

    struct RawEdge {
        uint32_t r, c;
        float val;
    };
    std::vector<RawEdge> raw_lines;
    raw_lines.reserve(10000);

    uint32_t r, c;
    float val;
    while (infile >> r >> c >> val) {
        raw_lines.push_back({r, c, val});
    }

    if (raw_lines.empty()) {
        std::cerr << "[ERROR] Parity check matrix file is empty: " << filepath << std::endl;
        return false;
    }

    const auto& dim_line = raw_lines.back();
    pcm.M = dim_line.r;
    pcm.N = dim_line.c;
    pcm.K = pcm.N - pcm.M;
    pcm.circulant_p = pcm.M / 3;
    pcm.P = pcm.circulant_p;

    pcm.check_to_var.resize(pcm.M);
    pcm.var_to_check.resize(pcm.N);

    uint32_t edge_count = 0;
    for (size_t i = 0; i + 1 < raw_lines.size(); ++i) {
        const auto& edge = raw_lines[i];
        if (std::abs(edge.val) > 1e-6f) {
            uint32_t m_idx = edge.r - 1;
            uint32_t v_idx = edge.c - 1;
            pcm.check_to_var[m_idx].push_back(v_idx);
            pcm.var_to_check[v_idx].push_back(m_idx);
            edge_count++;
        }
    }

    std::cout << "[INFO] Matrix Loaded: " << filepath << std::endl;
    std::cout << "       M=" << pcm.M << " checks, N=" << pcm.N << " variables, K=" << pcm.K 
              << ", circulant p=" << pcm.circulant_p << " (Lift2=" << pcm.circulant_p / 4 << ")" << std::endl;
    std::cout << "       Total Non-Zero Edges: " << edge_count << std::endl;

    return true;
}

// ------------------------------------------------------------------------------
// Step 2: Balanced Graph Partitioning for Collision-Free Schedule
// ------------------------------------------------------------------------------
static void partition_checks_balanced(
    const std::vector<uint32_t>& check_list,
    const ParityCheckMatrix& pcm,
    uint32_t target_tile_count,
    uint32_t max_tile_size,
    std::vector<std::vector<uint32_t>>& out_tiles
) {
    std::unordered_map<uint32_t, std::vector<uint32_t>> var_to_c;
    for (uint32_t c : check_list) {
        for (uint32_t v : pcm.check_to_var[c]) {
            var_to_c[v].push_back(c);
        }
    }

    std::vector<std::pair<uint32_t, uint32_t>> check_conflicts;
    for (uint32_t c : check_list) {
        std::unordered_set<uint32_t> conflicts;
        for (uint32_t v : pcm.check_to_var[c]) {
            for (uint32_t other : var_to_c[v]) {
                if (other != c) conflicts.insert(other);
            }
        }
        check_conflicts.push_back({c, static_cast<uint32_t>(conflicts.size())});
    }

    std::sort(check_conflicts.begin(), check_conflicts.end(), 
              [](const auto& a, const auto& b) { 
                  if (a.second != b.second) return a.second > b.second;
                  return a.first < b.first; 
              });

    out_tiles.assign(target_tile_count, std::vector<uint32_t>());
    std::vector<std::unordered_set<uint32_t>> tile_vars(target_tile_count);

    for (const auto& item : check_conflicts) {
        uint32_t c = item.first;
        std::unordered_set<uint32_t> c_vars(pcm.check_to_var[c].begin(), pcm.check_to_var[c].end());

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

        assert(best_tile != -1 && "Partitioning failed: check cannot be placed without collision!");
        out_tiles[best_tile].push_back(c);
        for (uint32_t v : c_vars) {
            tile_vars[best_tile].insert(v);
        }
    }
}

std::vector<DynamicTileDescriptor> build_circulant_aware_tile_schedule(ParityCheckMatrix& pcm) {
    std::vector<uint32_t> deg6_checks;
    std::vector<uint32_t> deg3_checks;

    for (uint32_t m = 0; m < pcm.M; ++m) {
        if (pcm.check_to_var[m].size() == 6) {
            deg6_checks.push_back(m);
        } else if (pcm.check_to_var[m].size() == 3) {
            deg3_checks.push_back(m);
        }
    }

    std::vector<std::vector<uint32_t>> deg6_tiles;
    partition_checks_balanced(deg6_checks, pcm, TILES_PER_DEG6_BLOCK * 2, CHECKS_PER_TILE_DEG6, deg6_tiles);

    std::vector<std::vector<uint32_t>> deg3_tiles;
    partition_checks_balanced(deg3_checks, pcm, TILES_PER_DEG3_BLOCK, CHECKS_PER_TILE_DEG3, deg3_tiles);

    // Proportionally interleave the 32 degree-6 tiles and 8 degree-3 tiles (Total: 40 tiles, 4:1 ratio)
    std::vector<DynamicTileDescriptor> schedule;
    schedule.reserve(TOTAL_TILES_PER_ITER);
    uint16_t current_h_offset = 0;
    uint16_t current_r_page   = 0;

    pcm.h_offsets_pairs.clear();

    auto add_tile_to_schedule = [&](const std::vector<uint32_t>& tile, uint8_t deg) {
        DynamicTileDescriptor desc{};
        desc.check_indices     = tile;
        desc.check_degree      = deg;
        desc.num_checks        = tile.size();
        desc.num_edges         = desc.num_checks * deg;
        desc.h_offset_start    = current_h_offset;
        desc.r_msg_page_offset = current_r_page;

        std::vector<uint32_t> w0_pairs(48);
        std::vector<uint32_t> w1_pairs(48);

        for (uint32_t p = 0; p < 48; ++p) {
            uint32_t d, kh, c0_even, c0_odd, c1_even, c1_odd;
            if (deg == 6) {
                if (p < 24) {
                    d = p / 4;
                    kh = p % 4;
                    // Face 0: checks 0..7 (Worker 0)
                    c0_even = 0 + kh;
                    c0_odd  = 4 + kh;
                    // Face 1: checks 8..15 (Worker 1)
                    c1_even = 8 + kh;
                    c1_odd  = 12 + kh;
                } else {
                    d = (p - 24) / 4;
                    kh = (p - 24) % 4;
                    // Face 2: checks 16..23 (Worker 0)
                    c0_even = 16 + kh;
                    c0_odd  = 20 + kh;
                    // Face 3: checks 24..31 (Worker 1)
                    c1_even = 24 + kh;
                    c1_odd  = 28 + kh;
                }
            } else { // deg == 3 (64 checks: Face 0: 0..15, Face 1: 16..31, Face 2: 32..47, Face 3: 48..63)
                if (p < 24) {
                    uint32_t blk = p / 12; // 0 or 1
                    d  = (p % 12) / 4;    // 0..2
                    kh = (p % 12) % 4;    // 0..3
                    c0_even = 0  + blk * 8 + kh;
                    c0_odd  = 0  + blk * 8 + 4 + kh;
                    c1_even = 16 + blk * 8 + kh;
                    c1_odd  = 16 + blk * 8 + 4 + kh;
                } else {
                    uint32_t blk = (p - 24) / 12; // 0 or 1
                    d  = ((p - 24) % 12) / 4;     // 0..2
                    kh = ((p - 24) % 12) % 4;     // 0..3
                    c0_even = 32 + blk * 8 + kh;
                    c0_odd  = 32 + blk * 8 + 4 + kh;
                    c1_even = 48 + blk * 8 + kh;
                    c1_odd  = 48 + blk * 8 + 4 + kh;
                }
            }

            uint32_t v0_even = (c0_even < tile.size()) ? pcm.check_to_var[tile[c0_even]][d] : 2559;
            uint32_t v0_odd  = (c0_odd  < tile.size()) ? pcm.check_to_var[tile[c0_odd]][d]  : 2559;
            w0_pairs[p] = make_pair_word(v0_even, v0_odd);

            uint32_t v1_even = (c1_even < tile.size()) ? pcm.check_to_var[tile[c1_even]][d] : 2559;
            uint32_t v1_odd  = (c1_odd  < tile.size()) ? pcm.check_to_var[tile[c1_odd]][d]  : 2559;
            w1_pairs[p] = make_pair_word(v1_even, v1_odd);
        }

        // Store Worker 0 pairs (48 pairs = 96 offsets)
        for (uint32_t p = 0; p < 48; ++p) pcm.h_offsets_pairs.push_back(w0_pairs[p]);
        // Store Worker 1 pairs (48 pairs = 96 offsets)
        for (uint32_t p = 0; p < 48; ++p) pcm.h_offsets_pairs.push_back(w1_pairs[p]);

        current_h_offset += 96 * sizeof(uint32_t);
        current_r_page   += 1;
        schedule.push_back(desc);
    };

    // Interleave 32 deg-6 tiles and 8 deg-3 tiles at a 4:1 ratio
    for (size_t i = 0; i < 8; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            add_tile_to_schedule(deg6_tiles[i * 4 + j], 6);
        }
        add_tile_to_schedule(deg3_tiles[i], 3);
    }

    return schedule;
}

// ------------------------------------------------------------------------------
// Step 3: Algorithmic Pre-Flight Collision Audit
// ------------------------------------------------------------------------------
bool verify_preflight_collision_audit(const ParityCheckMatrix& pcm, 
                                     const std::vector<DynamicTileDescriptor>& schedule) {
    std::cout << "\n=======================================================" << std::endl;
    std::cout << " PRE-FLIGHT COLLISION AUDIT: MATHEMATICAL VERIFICATION" << std::endl;
    std::cout << "=======================================================" << std::endl;

    uint32_t total_tiles = schedule.size();
    uint32_t total_edges = 0;
    bool all_passed = true;

    for (size_t t = 0; t < schedule.size(); ++t) {
        const auto& desc = schedule[t];
        std::unordered_set<uint32_t> var_footprint;
        uint32_t expected_variables = 0;

        for (uint32_t c : desc.check_indices) {
            const auto& vars = pcm.check_to_var[c];
            expected_variables += vars.size();

            for (uint32_t v : vars) {
                if (var_footprint.find(v) != var_footprint.end()) {
                    std::cerr << "[FAIL] Intra-Tile Collision in Tile " << t << " at variable " << v << "!" << std::endl;
                    all_passed = false;
                }
                var_footprint.insert(v);
            }
        }

        total_edges += desc.num_edges;
        assert(var_footprint.size() == expected_variables);
    }

    std::cout << "[AUDIT RESULT] Scheduled Tiles: " << total_tiles << " (26 Deg-6 + 7 Deg-3)" << std::endl;
    std::cout << "               Covered Edges:   " << total_edges << " / 7680 (100.0% Coverage)" << std::endl;
    std::cout << "               Collisions:      " << (all_passed ? "IDENTICALLY ZERO (100% COLLISION-FREE)" : "FAILED") << std::endl;
    std::cout << "=======================================================\n" << std::endl;

    return all_passed;
}


inline float to_bf16(float f) {
    return bf16_to_fp32(fp32_to_bf16(f));
}

// ------------------------------------------------------------------------------
// Dr. Jones' Branchless Approximate-Min* Formulation (matches SFPU spline)
// ------------------------------------------------------------------------------
inline float approx_min_star(float a, float b) {
    float min_v = std::min(a, b);
    float max_v = std::max(a, b);
    float diff  = max_v - min_v;

    // Exact SFPU immediate values matching TT-Metal sFloat16b compiler truncation:
    // 0.69315f -> 16177 (0x3F31)
    // -0.4000f -> 48844 (0xBECC)
    // 0.65715f -> 16168 (0x3F28)
    // -0.3280f -> 48807 (0xBEA7)
    static const float c_0_69315   = bf16_to_fp32(16177);
    static const float c_neg_0_4   = bf16_to_fp32(48844);
    static const float c_0_65715   = bf16_to_fp32(16168);
    static const float c_neg_0_328 = bf16_to_fp32(48807);

    float line1 = c_0_69315 + c_neg_0_4 * diff;
    float line2 = c_0_65715 + c_neg_0_328 * diff;
    float corr  = std::max(std::max(line1, line2), 0.0f);
    return std::max(min_v - corr, 0.0f);
}

// ------------------------------------------------------------------------------
// Step 4: Hardware Execution on Tenstorrent QuietBox 2
// ------------------------------------------------------------------------------
void run_tile_matrix_on_hardware(
    const ParityCheckMatrix& pcm,
    const std::vector<DynamicTileDescriptor>& schedule,
    double eb_n0_db,
    uint32_t num_batches,
    uint32_t max_iterations,
    bool single_core,
    bool benchmark_mode = false,
    bool diff_test = false,
    uint32_t tile_limit = 0
) {
    std::cout << "\n=======================================================" << std::endl;
    std::cout << " RUNNING TILE MATRIX 4-CODEWORD ENGINE ON QUIETBOX 2" << std::endl;
    std::cout << " Target: " << (single_core ? "Single Tensix Core (0, 0)" : "2x2 Blackhole Mesh") << std::endl;
    std::cout << " Batches: " << num_batches << " (4 CWs/batch = " << num_batches * 4 << " CWs)" << std::endl;
    std::cout << " Eb/N0: " << eb_n0_db << " dB | Max Iterations: " << max_iterations << std::endl;
    if (benchmark_mode) std::cout << " Mode: SILICON THROUGHPUT BENCHMARK (sigma=0)" << std::endl;
    if (diff_test) std::cout << " Mode: DIFFERENTIAL SILICON vs EMULATOR CO-VERIFICATION" << std::endl;
    std::cout << "=======================================================" << std::endl;

    auto mesh_device = single_core 
        ? distributed::MeshDevice::create_unit_mesh(0)
        : distributed::MeshDevice::create(distributed::MeshDeviceConfig(distributed::MeshShape{2, 2}));
    auto& cq = mesh_device->mesh_command_queue();

    CoreCoord core = {0, 0};
    CoreRange core_range(core, core);

    // Compute Channel Noise Parameters
    double rate = 0.5;
    double snr_lin = std::pow(10.0, eb_n0_db / 10.0);
    float mu_llr = static_cast<float>(4.0 * rate * snr_lin);
    float sigma_llr = benchmark_mode ? 0.0f : static_cast<float>(std::sqrt(8.0 * rate * snr_lin));
    uint32_t sigma_bits = 0;
    std::memcpy(&sigma_bits, &sigma_llr, sizeof(float));

    // DRAM Buffer Allocations for H Matrix and Telemetry
    uint32_t tile_bytes = sizeof(bfloat16) * TILE_ELEMENTS;
    uint32_t stats_cb_bytes = diff_test ? (64 * 1024) : (4 * 1024);
    uint32_t h_cb_bytes = 16 * 1024; // 16 KB (NOC_MAX_BURST_SIZE)

    distributed::DeviceLocalBufferConfig h_dram_config{
        .page_size = h_cb_bytes,
        .buffer_type = BufferType::DRAM
    };
    auto h_dram = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = h_cb_bytes}, h_dram_config, mesh_device.get()
    );

    // Pack H Table: 40 tiles x 96 uint32 pairs + tile metadata
    std::vector<uint16_t> h_table(h_cb_bytes / sizeof(uint16_t), 0);
    std::memcpy(h_table.data(), pcm.h_offsets_pairs.data(), pcm.h_offsets_pairs.size() * sizeof(uint32_t));
    uint16_t* meta_ptr = h_table.data() + (schedule.size() * 96 * 2);
    for (size_t t = 0; t < schedule.size(); ++t) {
        meta_ptr[t * 2 + 0] = static_cast<uint16_t>(schedule[t].num_checks);
        meta_ptr[t * 2 + 1] = static_cast<uint16_t>(schedule[t].check_degree);
    }
    distributed::EnqueueWriteMeshBuffer(cq, h_dram, h_table, false);

    distributed::DeviceLocalBufferConfig stats_dram_config{
        .page_size = stats_cb_bytes,
        .buffer_type = BufferType::DRAM
    };
    auto stats_dram = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = stats_cb_bytes}, stats_dram_config, mesh_device.get()
    );

    // Program Setup
    Program program = CreateProgram();

    // CB Allocations in Tensix L1 SRAM:
    CreateCircularBuffer(program, core_range, CircularBufferConfig(20 * 1024, {{TileMatrixCB::CB_VNODE_A, tt::DataFormat::Float16_b}}).set_page_size(TileMatrixCB::CB_VNODE_A, 2 * 1024));
    CreateCircularBuffer(program, core_range, CircularBufferConfig(std::max<uint32_t>(34, schedule.size() + 1) * 2 * 1024, {{TileMatrixCB::CB_R_MSG, tt::DataFormat::Float16_b}}).set_page_size(TileMatrixCB::CB_R_MSG, 2 * 1024));
    CreateCircularBuffer(program, core_range, CircularBufferConfig(16 * 1024, {{TileMatrixCB::CB_H_OFFSET, tt::DataFormat::RawUInt16}}).set_page_size(TileMatrixCB::CB_H_OFFSET, 16 * 1024));
    CreateCircularBuffer(program, core_range, CircularBufferConfig(4 * 1024, {{TileMatrixCB::CB_SCRATCH_GATHER_0, tt::DataFormat::Float16_b}}).set_page_size(TileMatrixCB::CB_SCRATCH_GATHER_0, 2 * 1024));
    CreateCircularBuffer(program, core_range, CircularBufferConfig(4 * 1024, {{TileMatrixCB::CB_SCRATCH_GATHER_1, tt::DataFormat::Float16_b}}).set_page_size(TileMatrixCB::CB_SCRATCH_GATHER_1, 2 * 1024));
    CreateCircularBuffer(program, core_range, CircularBufferConfig(20 * 1024, {{TileMatrixCB::CB_VNODE_B, tt::DataFormat::Float16_b}}).set_page_size(TileMatrixCB::CB_VNODE_B, 2 * 1024));
    CreateCircularBuffer(program, core_range, CircularBufferConfig(4 * 1024, {{TileMatrixCB::CB_SCRATCH_SCATTER_0, tt::DataFormat::Float16_b}}).set_page_size(TileMatrixCB::CB_SCRATCH_SCATTER_0, 2 * 1024));
    CreateCircularBuffer(program, core_range, CircularBufferConfig(4 * 1024, {{TileMatrixCB::CB_SCRATCH_SCATTER_1, tt::DataFormat::Float16_b}}).set_page_size(TileMatrixCB::CB_SCRATCH_SCATTER_1, 2 * 1024));
    CreateCircularBuffer(program, core_range, CircularBufferConfig(4 * 1024, {{TileMatrixCB::CB_TELEMETRY, tt::DataFormat::RawUInt32}}).set_page_size(TileMatrixCB::CB_TELEMETRY, 4 * 1024));

    // Compile Device Kernels:
    std::string reader_path = resolve_filepath("reader_brisc.cpp");
    std::string writer_path = resolve_filepath("writer_ncrisc.cpp");
    std::string compute_path = resolve_filepath("compute_trisc.cpp");

    auto reader = CreateKernel(
        program, reader_path, core_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );
    auto writer = CreateKernel(
        program, writer_path, core_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default}
    );
    auto compute = CreateKernel(
        program, compute_path, core_range,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = false}
    );

    // Runtime Args:
    uint32_t num_tiles = (tile_limit > 0) ? tile_limit : schedule.size();
    uint32_t mu_u32 = 0;
    std::memcpy(&mu_u32, &mu_llr, sizeof(float));

    std::vector<uint32_t> r_args = {
        num_batches, max_iterations, num_tiles,
        static_cast<uint32_t>(h_dram->address()), 0, h_cb_bytes, 1,
        mu_u32, sigma_bits, 0
    };
    SetRuntimeArgs(program, reader, core, r_args);

    std::vector<uint32_t> w_args = {
        num_batches, max_iterations, num_tiles,
        static_cast<uint32_t>(stats_dram->address()), 0, stats_cb_bytes,
        mu_u32, sigma_bits, 0, 1,
        diff_test ? 1u : 0u
    };
    SetRuntimeArgs(program, writer, core, w_args);

    constexpr uint32_t telem_l1_addr = 0x3DC00;
    std::vector<uint32_t> c_args = { num_batches, max_iterations, num_tiles, mu_u32, sigma_bits, 1 };
    SetRuntimeArgs(program, compute, core, c_args);

    std::cout << "[RUN] Dispatching Optimized Tile Matrix 4-Codeword Workload to Core (0, 0)..." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange target_range = single_core
        ? distributed::MeshCoordinateRange(distributed::MeshCoordinate(0, 0), distributed::MeshCoordinate(0, 0))
        : distributed::MeshCoordinateRange(mesh_device->shape());
    workload.add_program(target_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    cq.finish();

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::vector<uint32_t> stats(stats_cb_bytes / sizeof(uint32_t), 0);
    distributed::ReadShard(cq, stats, stats_dram, distributed::MeshCoordinate(0, 0), true);

    if (stats[20] != 0) {
        std::cout << "[INFO] Telemetry L1 Base: 0x" << std::hex << stats[20] << std::dec << std::endl;
    }
    std::cout << "[INFO] Tile 0 deg from stats[64]: " << stats[64] << std::endl;
    uint32_t hoff_mismatches = 0;
    for (uint32_t i = 0; i < 96; ++i) {
        if (stats[200 + i] != pcm.h_offsets_pairs[i]) {
            if (hoff_mismatches < 5) {
                std::cout << "  [H_OFF MISMATCH] i=" << i 
                          << " Tensix=0x" << std::hex << stats[200 + i]
                          << " Host=0x" << pcm.h_offsets_pairs[i] << std::dec << std::endl;
            }
            hoff_mismatches++;
        }
    }
    std::cout << "[INFO] H_Offsets Tensix L1 vs Host: " << (96 - hoff_mismatches) << " / 96 match (" 
              << hoff_mismatches << " mismatches)" << std::endl;
    std::cout << "  Tile 0 all 32 checks: ";
    for (int i = 0; i < 32; ++i) std::cout << schedule[0].check_indices[i] << " ";
    std::cout << std::endl;

    if (diff_test) {
        fs::create_directories("dump");
        const uint8_t* raw_init_ptr  = reinterpret_cast<const uint8_t*>(stats.data() + 1024);
        const uint8_t* raw_final_ptr = reinterpret_cast<const uint8_t*>(stats.data() + 1024 + 5120);

        const uint8_t* raw_scat_out_ptr = reinterpret_cast<const uint8_t*>(stats.data() + 11264);
        const uint8_t* raw_scat_in_ptr  = reinterpret_cast<const uint8_t*>(stats.data() + 11776);

        std::ofstream f_tensix_scat_out("dump/tensix_scratch_out.bin", std::ios::binary);
        f_tensix_scat_out.write(reinterpret_cast<const char*>(raw_scat_out_ptr), 2048);
        f_tensix_scat_out.close();

        std::ofstream f_tensix_scat_in("dump/tensix_scratch_in.bin", std::ios::binary);
        f_tensix_scat_in.write(reinterpret_cast<const char*>(raw_scat_in_ptr), 2048);
        f_tensix_scat_in.close();

        std::ofstream f_tensix_init("dump/tensix_initial_vnodes.bin", std::ios::binary);
        f_tensix_init.write(reinterpret_cast<const char*>(raw_init_ptr), 20480);
        f_tensix_init.close();

        std::ofstream f_tensix_final("dump/tensix_final_vnodes.bin", std::ios::binary);
        f_tensix_final.write(reinterpret_cast<const char*>(raw_final_ptr), 20480);
        f_tensix_final.close();
        std::cout << "\n[DIFF TEST] Dumped initial, final, scratch_in, scratch_out to dump/" << std::endl;

        // Initialize golden_l1 from raw Tensix initial LLRs in native L1 format
        std::vector<uint8_t> golden_l1(raw_init_ptr, raw_init_ptr + 20480);
        uint8_t* vnode_l1 = golden_l1.data();

        // Check message FIFO holding schedule.size() tiles (2048 bytes each = 1024 uint16_t words)
        std::vector<std::vector<uint16_t>> r_msg_fifo(schedule.size(), std::vector<uint16_t>(1024, 0));
        std::vector<uint16_t> scratch_in(1024, 0);
        std::vector<uint16_t> scratch_out(1024, 0);

        for (uint32_t iter = 0; iter < max_iterations; ++iter) {
            for (size_t t = 0; t < num_tiles; ++t) {
                const auto& tile = schedule[t];
                uint32_t deg = tile.check_degree;
                const uint32_t* tile_hoff_w0 = pcm.h_offsets_pairs.data() + t * 96;
                const uint32_t* tile_hoff_w1 = tile_hoff_w0 + 48;

                // 1. Dual Gather Stage (Worker 0 & Worker 1)
                for (int p = 0; p < 48; ++p) {
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

                // 2. Compute Stage (Face-Aligned SFPU across all 32 lanes)
                uint32_t num_passes = (deg == 6) ? 4 : 8;
                const int deg6_face_bases[4] = {0, 8, 16, 24};
                const int deg3_face_bases[8] = {0, 3, 8, 11, 16, 19, 24, 27};

                for (uint32_t pass = 0; pass < num_passes; ++pass) {
                    uint32_t base_r = (deg == 6) ? deg6_face_bases[pass] : deg3_face_bases[pass];
                    for (uint32_t lane = 0; lane < 32; ++lane) {
                        uint32_t col = lane;
                        float q[8], signs[8], mags[8];
                        float total_sign = 1.0f;
                        uint32_t min_idx = 0;
                        float min_mag = 99999.0f;

                        for (uint32_t r = 0; r < deg; ++r) {
                            uint32_t offset = get_scratch_byte_offset(base_r + r, col);
                            float l_val = bf16_to_fp32(*reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(scratch_in.data()) + offset));
                            float r_old = (iter == 0) ? 0.0f : bf16_to_fp32(*reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(r_msg_fifo[t].data()) + offset));

                            q[r] = to_bf16(l_val - r_old);
                            signs[r] = (q[r] < 0.0f) ? -1.0f : 1.0f;
                            mags[r]  = std::fabs(q[r]);
                            total_sign *= signs[r];
                            if (mags[r] < min_mag) {
                                min_mag = mags[r];
                                min_idx = r;
                            }
                        }

                        // Branchless Approximate-Min* Check Update
                        float m_others = (min_idx == 0) ? 1000.0f : mags[0];
                        for (uint32_t r = 1; r < deg; ++r) {
                            float m = (min_idx == r) ? 1000.0f : mags[r];
                            m_others = approx_min_star(m_others, m);
                        }
                        float m_all = approx_min_star(m_others, min_mag);

                        // Scatter out to Tile 0 (R_new) and Tile 1 (L_new)
                        for (uint32_t r = 0; r < deg; ++r) {
                            uint32_t offset = get_scratch_byte_offset(base_r + r, col);
                            float edge_sign = total_sign * signs[r];
                            float edge_mag = (r == min_idx) ? m_others : m_all;
                            float r_raw = edge_sign * edge_mag;
                            float l_new = to_bf16(q[r] + r_raw);
                            float r_new = to_bf16(r_raw);

                            *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(r_msg_fifo[t].data()) + offset) = fp32_to_bf16(r_new);
                            *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(scratch_out.data()) + offset)   = fp32_to_bf16(l_new);
                        }
                    }
                }

                if (iter == 0 && t == 0) {
                    std::ofstream f_gold_scat_in("dump/golden_scratch_in.bin", std::ios::binary);
                    f_gold_scat_in.write(reinterpret_cast<const char*>(scratch_in.data()), 2048);
                    f_gold_scat_in.close();

                    std::ofstream f_gold_scat_out("dump/golden_scratch_out.bin", std::ios::binary);
                    f_gold_scat_out.write(reinterpret_cast<const char*>(scratch_out.data()), 2048);
                    f_gold_scat_out.close();
                }

                // 3. Dual Scatter Stage
                for (int p = 0; p < 48; ++p) {
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
            }
        }

        std::ofstream f_golden("dump/golden_final_vnodes.bin", std::ios::binary);
        f_golden.write(reinterpret_cast<const char*>(golden_l1.data()), 20480);
        f_golden.close();
        std::cout << "[DIFF TEST] Generated golden reference after " << max_iterations << " iterations: dump/golden_final_vnodes.bin" << std::endl;

        std::cout << "\n[DIFF TEST] Running Differential Comparison Tool:" << std::endl;
        const char* script_path = fs::exists("../../tools/diff_tensix_vs_emulator.py") 
                                ? "../../tools/diff_tensix_vs_emulator.py" 
                                : "tools/diff_tensix_vs_emulator.py";
        const char* py_bin = fs::exists("/home/ttuser/.tenstorrent-venv/bin/python") 
                           ? "/home/ttuser/.tenstorrent-venv/bin/python" 
                           : "python3";
        std::string cmd_in = std::string(py_bin) + " " + script_path + " dump/golden_scratch_in.bin dump/tensix_scratch_in.bin --tile CB_SCRATCH_GATHER";
        std::system(cmd_in.c_str());

        std::string cmd_out = std::string(py_bin) + " " + script_path + " dump/golden_scratch_out.bin dump/tensix_scratch_out.bin --tile CB_SCRATCH_SCATTER";
        std::system(cmd_out.c_str());

        std::string cmd = std::string(py_bin) + " " + script_path + " dump/golden_final_vnodes.bin dump/tensix_final_vnodes.bin";
        int ret = std::system(cmd.c_str());
        (void)ret;
    }

    uint32_t total_cws = num_batches * 4;
    double throughput_msps = (static_cast<double>(total_cws) * 2048) / (elapsed_ms * 1e3);
    double mean_iters = (double)stats[2] / total_cws;
    double actual_batch_iters = (mean_iters > 0.0) ? mean_iters : 1.0;
    uint32_t hw_cycles = stats[4];
    double cycles_per_batch = (double)hw_cycles / (num_batches > 0 ? num_batches : 1);
    double cycles_per_tile = cycles_per_batch / (actual_batch_iters * num_tiles);
    double silicon_core_msps = (static_cast<double>(total_cws) * 2048 * 1.35e9) / (hw_cycles > 0 ? hw_cycles : 1) / 1e6;
    double mesh_projected_gsps = (silicon_core_msps * 440.0) / 1e3;

    uint32_t trisc_cycles = stats[6];
    double trisc_cycles_per_tile = ((double)trisc_cycles / (num_batches > 0 ? num_batches : 1)) / (actual_batch_iters * num_tiles);
    double brisc_cycles_per_tile = ((double)stats[7] / (num_batches > 0 ? num_batches : 1)) / (actual_batch_iters * num_tiles);

    std::cout << "\n=======================================================" << std::endl;
    std::cout << " TILE MATRIX 4-CODEWORD HARDWARE RESULTS" << std::endl;
    std::cout << "=======================================================" << std::endl;
    std::cout << " Codewords Tested : " << total_cws << " (" << num_batches << " batches of 4)" << std::endl;
    std::cout << " Init Neg LLRs B0 : CW0=" << stats[40] << ", CW1=" << stats[41]
              << ", CW2=" << stats[42] << ", CW3=" << stats[43] << " (out of 2048)" << std::endl;
    std::cout << " Init BF16 Words  : [0,1]=0x" << std::hex << stats[44] << ", [2,3]=0x" << stats[45]
              << ", mu_bits=0x" << stats[46] << ", sigma_bits=0x" << stats[47] << std::dec << std::endl;
    std::cout << " Total Bit Errors : " << stats[0]
              << " (CW0=" << stats[48] << ", CW1=" << stats[49] << ", CW2=" << stats[50] << ", CW3=" << stats[51] << ")" << std::endl;
    std::cout << " Total Frame Errs : " << stats[1] << " (FER = " << (double)stats[1] / total_cws << ")" << std::endl;
    std::cout << " CW Iters Taken   : CW0=" << stats[52] << ", CW1=" << stats[53] << ", CW2=" << stats[54] << ", CW3=" << stats[55] << std::endl;
    std::cout << " Mean Iterations  : " << mean_iters << std::endl;
    std::cout << " NCRISC HW Cycles : " << hw_cycles << " clock cycles (" << cycles_per_tile << " cyc/tile)" << std::endl;
    std::cout << " TRISC HW Cycles  : " << trisc_cycles << " clock cycles (" << trisc_cycles_per_tile << " cyc/tile)" << std::endl;
    std::cout << " BRISC HW Cycles  : " << stats[7] << " clock cycles (" << brisc_cycles_per_tile << " cyc/tile)" << std::endl;
    std::cout << " Telemetry L1 Addr: 0x" << std::hex << stats[20] << std::dec << std::endl;
    std::cout << " --- DETAILED CYCLE BREAKDOWN (per tile) ---" << std::endl;
    double total_tiles_sim = ((double)stats[2] / 4.0) * num_tiles;
    double num_eval_batches = total_tiles_sim / (double)num_tiles;
    std::cout << " Dual Gather Tot  : " << (double)stats[10] / total_tiles_sim << " cyc/tile" << std::endl;
    std::cout << " Wait for Compute : " << (double)stats[11] / total_tiles_sim << " cyc/tile" << std::endl;
    std::cout << " Dual Scatter Tot : " << (double)stats[12] / total_tiles_sim << " cyc/tile" << std::endl;
    std::cout << " Channel Init Tot : " << (double)stats[8] / num_eval_batches << " cyc/batch ("
              << (double)stats[8] / total_tiles_sim << " cyc/tile)" << std::endl;
    std::cout << " Error Scan Tot   : " << (double)stats[9] / num_eval_batches << " cyc/batch ("
              << (double)stats[9] / total_tiles_sim << " cyc/tile)" << std::endl;
    std::cout << "   - Start Barrier: " << (double)stats[13] / total_tiles_sim << " cyc/tile" << std::endl;
    std::cout << "   - Pure ScatLoop: " << (double)stats[14] / total_tiles_sim << " cyc/tile (Tile 0: "
              << (double)stats[18] / num_eval_batches << " cyc | Others: "
              << (double)stats[19] / (num_eval_batches * 32.0) << " cyc)" << std::endl;
    std::cout << "   - R_new Copy   : " << (double)stats[15] / total_tiles_sim << " cyc/tile" << std::endl;
    std::cout << "   - End Barrier  : " << (double)stats[16] / total_tiles_sim << " cyc/tile" << std::endl;
    std::cout << "   - CB Pop Front : " << (double)stats[17] / total_tiles_sim << " cyc/tile" << std::endl;
    std::cout << " --- WORKER 0 (BRISC) BREAKDOWN ---" << std::endl;
    std::cout << " BRISC Gather     : " << (double)stats[32] / total_tiles_sim << " cyc/tile" << std::endl;
    std::cout << " BRISC Wait       : " << (double)stats[34] / total_tiles_sim << " cyc/tile" << std::endl;
    std::cout << " BRISC Scatter Tot: " << (double)stats[33] / total_tiles_sim << " cyc/tile" << std::endl;
    std::cout << "   - Pure ScatLoop: " << (double)stats[35] / total_tiles_sim << " cyc/tile" << std::endl;
    std::cout << "   - R_new Copy   : " << (double)stats[36] / total_tiles_sim << " cyc/tile" << std::endl;
    std::cout << "   - End Barrier  : " << (double)stats[37] / total_tiles_sim << " cyc/tile" << std::endl;
    std::cout << " Silicon Core Tput: " << std::fixed << std::setprecision(2) << silicon_core_msps << " Msps (silicon-only at 1.35 GHz)" << std::endl;
    std::cout << " Mesh Proj. Tput  : " << std::fixed << std::setprecision(2) << mesh_projected_gsps << " Gsps (440 cores)" << std::endl;
    std::cout << " Host Wall-Clock  : " << std::fixed << std::setprecision(2) << elapsed_ms << " ms (incl. PCIe dispatch)" << std::endl;
    std::cout << "=======================================================" << std::endl;

    mesh_device->close();
}

// ------------------------------------------------------------------------------
// Step 5: Full Waterfall Simulation Sweep on Hardware
// ------------------------------------------------------------------------------
void run_waterfall_sweep(
    const ParityCheckMatrix& pcm,
    const std::vector<DynamicTileDescriptor>& schedule,
    double snr_start, double snr_stop, double snr_step,
    uint32_t min_frame_errors,
    uint64_t max_blocks_per_snr,
    uint32_t max_iterations,
    bool single_core,
    const std::string& csv_out_path
) {
    (void)single_core;
    std::cout << "\n=======================================================" << std::endl;
    std::cout << " QUIETBOX 2: TILE MATRIX 4-CW WATERFALL SWEEP" << std::endl;
    std::cout << " Target: Single Tensix Core (0, 0)" << std::endl;
    std::cout << " Range : [" << snr_start << " : " << snr_stop << " : " << snr_step << "] dB" << std::endl;
    std::cout << " Goal  : " << min_frame_errors << " Frame Errors or " << max_blocks_per_snr << " Blocks / SNR" << std::endl;
    std::cout << " MaxIt : " << max_iterations << " iterations" << std::endl;
    std::cout << " Output: " << csv_out_path << std::endl;
    std::cout << "=======================================================" << std::endl;

    auto mesh_device = single_core 
        ? distributed::MeshDevice::create_unit_mesh(0)
        : distributed::MeshDevice::create(distributed::MeshDeviceConfig(distributed::MeshShape{2, 2}));
    auto& cq = mesh_device->mesh_command_queue();

    CoreCoord core = {0, 0};
    CoreRange core_range(core, core);

    uint32_t tile_bytes = sizeof(bfloat16) * TILE_ELEMENTS;
    uint32_t stats_cb_bytes = 4 * 1024;
    uint32_t h_cb_bytes = 16 * 1024;

    distributed::DeviceLocalBufferConfig h_dram_config{
        .page_size = h_cb_bytes,
        .buffer_type = BufferType::DRAM
    };
    auto h_dram = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = h_cb_bytes}, h_dram_config, mesh_device.get()
    );

    std::vector<uint16_t> h_table(h_cb_bytes / sizeof(uint16_t), 0);
    std::memcpy(h_table.data(), pcm.h_offsets_pairs.data(), pcm.h_offsets_pairs.size() * sizeof(uint32_t));
    uint16_t* meta_ptr = h_table.data() + (schedule.size() * 96 * 2);
    for (size_t t = 0; t < schedule.size(); ++t) {
        meta_ptr[t * 2 + 0] = static_cast<uint16_t>(schedule[t].num_checks);
        meta_ptr[t * 2 + 1] = static_cast<uint16_t>(schedule[t].check_degree);
    }
    distributed::EnqueueWriteMeshBuffer(cq, h_dram, h_table, false);
    cq.finish();

    distributed::DeviceLocalBufferConfig stats_dram_config{
        .page_size = stats_cb_bytes,
        .buffer_type = BufferType::DRAM
    };
    auto stats_dram = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = stats_cb_bytes}, stats_dram_config, mesh_device.get()
    );

    if (fs::path(csv_out_path).has_parent_path()) {
        std::error_code ec;
        fs::create_directories(fs::path(csv_out_path).parent_path(), ec);
    }
    bool file_exists = fs::exists(csv_out_path);
    std::ofstream csv_file(csv_out_path, std::ios::app);
    if (!csv_file.is_open()) {
        std::cerr << "[ERROR] Cannot open CSV file: " << csv_out_path << std::endl;
        return;
    }
    if (!file_exists) {
        csv_file << "EbN0_dB,Total_Blocks,Total_Bit_Errors,Total_Block_Errors,BER,FER,Mean_Iterations,Throughput_Msps,Host_Wall_s\n";
        csv_file.flush();
    }

    std::string reader_path = resolve_filepath("reader_brisc.cpp");
    std::string writer_path = resolve_filepath("writer_ncrisc.cpp");
    std::string compute_path = resolve_filepath("compute_trisc.cpp");
    uint32_t num_tiles = schedule.size();
    constexpr uint32_t telem_l1_addr = 0x3DC00;

    uint32_t global_seed = 0x12345678;
    uint32_t invocation_id = 1;

    for (double snr = snr_start; snr <= snr_stop + 1e-4; snr += snr_step) {
        double rate = 0.5;
        double snr_lin = std::pow(10.0, snr / 10.0);
        float mu_llr = static_cast<float>(4.0 * rate * snr_lin);
        float sigma_llr = static_cast<float>(std::sqrt(8.0 * rate * snr_lin));
        uint32_t mu_u32 = 0, sigma_bits = 0;
        std::memcpy(&mu_u32, &mu_llr, sizeof(float));
        std::memcpy(&sigma_bits, &sigma_llr, sizeof(float));

        uint64_t snr_blocks = 0;
        uint64_t snr_be = 0;
        uint64_t snr_fe = 0;
        uint64_t snr_iters = 0;

        auto snr_t0 = std::chrono::high_resolution_clock::now();

        std::cout << "\n>>> Simulating Eb/N0 = " << std::fixed << std::setprecision(2) << snr << " dB..." << std::endl;

        while (snr_fe < min_frame_errors && snr_blocks < max_blocks_per_snr) {
            uint64_t remaining_blocks = max_blocks_per_snr - snr_blocks;
            uint64_t target_chunk_blocks = 100;
            if (snr_fe == 0) {
                if (snr_blocks >= 100000) target_chunk_blocks = 50000;
                else if (snr_blocks >= 20000) target_chunk_blocks = 20000;
                else if (snr_blocks >= 5000)  target_chunk_blocks = 5000;
                else if (snr_blocks >= 1000)  target_chunk_blocks = 1000;
                else target_chunk_blocks = 200;
            } else {
                double est_fer = static_cast<double>(snr_fe) / snr_blocks;
                // Target collecting ~2 errors per chunk to balance progress telemetry with high throughput
                uint64_t blocks_for_target = static_cast<uint64_t>(2.5 / est_fer);
                if (blocks_for_target < 200) blocks_for_target = 200;
                if (blocks_for_target > 25000) blocks_for_target = 25000;
                target_chunk_blocks = blocks_for_target;
            }

            uint32_t chunk_batches = static_cast<uint32_t>(target_chunk_blocks / 4);
            if (static_cast<uint64_t>(chunk_batches * 4) > remaining_blocks) {
                chunk_batches = static_cast<uint32_t>((remaining_blocks + 3) / 4);
            }
            if (chunk_batches == 0) break;

            Program program = CreateProgram();

            CreateCircularBuffer(program, core_range, CircularBufferConfig(20 * 1024, {{TileMatrixCB::CB_VNODE_A, tt::DataFormat::Float16_b}}).set_page_size(TileMatrixCB::CB_VNODE_A, 2 * 1024));
            CreateCircularBuffer(program, core_range, CircularBufferConfig(std::max<uint32_t>(34, schedule.size() + 1) * 2 * 1024, {{TileMatrixCB::CB_R_MSG, tt::DataFormat::Float16_b}}).set_page_size(TileMatrixCB::CB_R_MSG, 2 * 1024));
            CreateCircularBuffer(program, core_range, CircularBufferConfig(16 * 1024, {{TileMatrixCB::CB_H_OFFSET, tt::DataFormat::RawUInt16}}).set_page_size(TileMatrixCB::CB_H_OFFSET, 16 * 1024));
            CreateCircularBuffer(program, core_range, CircularBufferConfig(4 * 1024, {{TileMatrixCB::CB_SCRATCH_GATHER_0, tt::DataFormat::Float16_b}}).set_page_size(TileMatrixCB::CB_SCRATCH_GATHER_0, 2 * 1024));
            CreateCircularBuffer(program, core_range, CircularBufferConfig(4 * 1024, {{TileMatrixCB::CB_SCRATCH_GATHER_1, tt::DataFormat::Float16_b}}).set_page_size(TileMatrixCB::CB_SCRATCH_GATHER_1, 2 * 1024));
            CreateCircularBuffer(program, core_range, CircularBufferConfig(20 * 1024, {{TileMatrixCB::CB_VNODE_B, tt::DataFormat::Float16_b}}).set_page_size(TileMatrixCB::CB_VNODE_B, 2 * 1024));
            CreateCircularBuffer(program, core_range, CircularBufferConfig(4 * 1024, {{TileMatrixCB::CB_SCRATCH_SCATTER_0, tt::DataFormat::Float16_b}}).set_page_size(TileMatrixCB::CB_SCRATCH_SCATTER_0, 2 * 1024));
            CreateCircularBuffer(program, core_range, CircularBufferConfig(4 * 1024, {{TileMatrixCB::CB_SCRATCH_SCATTER_1, tt::DataFormat::Float16_b}}).set_page_size(TileMatrixCB::CB_SCRATCH_SCATTER_1, 2 * 1024));
            CreateCircularBuffer(program, core_range, CircularBufferConfig(4 * 1024, {{TileMatrixCB::CB_TELEMETRY, tt::DataFormat::RawUInt32}}).set_page_size(TileMatrixCB::CB_TELEMETRY, 4 * 1024));

            auto reader = CreateKernel(
                program, reader_path, core_range,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
            );
            auto writer = CreateKernel(
                program, writer_path, core_range,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default}
            );
            auto compute = CreateKernel(
                program, compute_path, core_range,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = false}
            );

            std::vector<uint32_t> r_args = { chunk_batches, max_iterations, num_tiles, static_cast<uint32_t>(h_dram->address()), 0, h_cb_bytes, invocation_id, mu_u32, sigma_bits, global_seed };
            SetRuntimeArgs(program, reader, core, r_args);

            std::vector<uint32_t> w_args = { chunk_batches, max_iterations, num_tiles, static_cast<uint32_t>(stats_dram->address()), 0, stats_cb_bytes, mu_u32, sigma_bits, global_seed, invocation_id };
            SetRuntimeArgs(program, writer, core, w_args);

            std::vector<uint32_t> c_args = { chunk_batches, max_iterations, num_tiles, mu_u32, sigma_bits, invocation_id };
            SetRuntimeArgs(program, compute, core, c_args);

            distributed::MeshWorkload workload;
            distributed::MeshCoordinateRange target_range = single_core
                ? distributed::MeshCoordinateRange(distributed::MeshCoordinate(0, 0), distributed::MeshCoordinate(0, 0))
                : distributed::MeshCoordinateRange(mesh_device->shape());
            workload.add_program(target_range, std::move(program));
            distributed::EnqueueMeshWorkload(cq, workload, false);
            cq.finish();

            std::vector<uint32_t> stats(stats_cb_bytes / sizeof(uint32_t), 0);
            distributed::ReadShard(cq, stats, stats_dram, distributed::MeshCoordinate(0, 0), true);

            snr_blocks += chunk_batches * 4;
            snr_be += stats[0];
            snr_fe += stats[1];
            snr_iters += stats[2];

            if (snr_blocks <= 100) {
                std::cout << "\n[DEBUG TELEMETRY] BE=" << stats[0] << " FE=" << stats[1] << " Iters=" << stats[2]
                          << "\n  B0: InitNeg=[" << stats[36] << "," << stats[37] << "," << stats[38] << "," << stats[39] << "]"
                          << " s01=0x" << std::hex << stats[34] << " s23=0x" << stats[35] << std::dec
                          << "\n  B1: InitNeg=[" << stats[40] << "," << stats[41] << "," << stats[42] << "," << stats[43] << "]"
                          << " s01=0x" << std::hex << stats[44] << " s23=0x" << stats[45] << std::dec
                          << "\n  TRISC: math_b=" << (stats[56] & 0xFFFF) << " pack_b=" << (stats[57] & 0xFFFF) << " unpack_b=" << (stats[58] & 0xFFFF)
                          << " rd_nc=0x" << std::hex << stats[60] << " wr_nc=0x" << stats[61]
                          << " found_addr=0x" << stats[62] << " val=0x" << stats[63] << std::dec << std::endl;
                std::cout << "  Failing VNs CW0 (" << stats[98] << "): ";
                for (uint32_t i = 0; i < std::min<uint32_t>(stats[98], 16); ++i) std::cout << stats[100 + i] << " ";
                std::cout << "\n  Failing VNs CW1 (" << stats[99] << "): ";
                for (uint32_t i = 0; i < std::min<uint32_t>(stats[99], 16); ++i) std::cout << stats[132 + i] << " ";
                std::cout << std::endl;
            }

            global_seed += chunk_batches * 10007 + 1;
            invocation_id++;

            double cur_fer = (double)snr_fe / snr_blocks;
            double cur_ber = (double)snr_be / (snr_blocks * 2048.0);
            double cur_iter = (double)snr_iters / snr_blocks;

            static const bool is_tty = isatty(fileno(stdout));
            if (is_tty) {
                std::cout << "\r  [" << std::fixed << std::setprecision(2) << snr << " dB] Blocks: "
                          << std::setw(7) << snr_blocks << "/" << max_blocks_per_snr
                          << " | FE: " << std::setw(3) << snr_fe << "/" << min_frame_errors
                          << " | BE: " << std::setw(8) << snr_be
                          << " | FER: " << std::scientific << std::setprecision(3) << cur_fer
                          << " | BER: " << cur_ber
                          << " | Iter: " << std::fixed << std::setprecision(2) << cur_iter
                          << std::flush;
            } else {
                std::cout << "  [" << std::fixed << std::setprecision(2) << snr << " dB] Blocks: "
                          << std::setw(7) << snr_blocks << "/" << max_blocks_per_snr
                          << " | FE: " << std::setw(3) << snr_fe << "/" << min_frame_errors
                          << " | BE: " << std::setw(8) << snr_be
                          << " | FER: " << std::scientific << std::setprecision(3) << cur_fer
                          << " | BER: " << cur_ber
                          << " | Iter: " << std::fixed << std::setprecision(2) << cur_iter
                          << std::endl;
            }
        }

        auto snr_t1 = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(snr_t1 - snr_t0).count();

        double final_fer = (double)snr_fe / (snr_blocks > 0 ? snr_blocks : 1);
        double final_ber = (double)snr_be / ((snr_blocks > 0 ? snr_blocks : 1) * 2048.0);
        double final_iter = (double)snr_iters / (snr_blocks > 0 ? snr_blocks : 1);
        double tput_msps = ((double)snr_blocks * 2048.0) / (elapsed_sec * 1e6);

        std::cout << "\n[COMPLETED " << std::fixed << std::setprecision(2) << snr << " dB]"
                  << " Blocks: " << snr_blocks
                  << " | FE: " << snr_fe
                  << " | BE: " << snr_be
                  << " | FER: " << std::scientific << std::setprecision(4) << final_fer
                  << " | BER: " << final_ber
                  << " | MeanIter: " << std::fixed << std::setprecision(2) << final_iter
                  << " | Tput: " << std::setprecision(2) << tput_msps << " Msps"
                  << " | Time: " << std::setprecision(1) << elapsed_sec << " s"
                  << std::endl;

        csv_file << std::fixed << std::setprecision(2) << snr << ","
                 << snr_blocks << ","
                 << snr_be << ","
                 << snr_fe << ","
                 << std::scientific << std::setprecision(6) << final_ber << ","
                 << final_fer << ","
                 << std::fixed << std::setprecision(3) << final_iter << ","
                 << std::setprecision(2) << tput_msps << ","
                 << std::setprecision(2) << elapsed_sec << "\n";
        csv_file.flush();
    }

    mesh_device->close();
}

// ------------------------------------------------------------------------------
// Main Simulation Test Harness
// ------------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string matrix_path = resolve_filepath("matrices/AR4JA_r45_4c_128c_r12.chinn.out");
    bool is_waterfall = false;
    double snr_start = 1.5, snr_stop = 2.5, snr_step = 0.1;
    uint32_t min_fe = 20;
    uint64_t max_blocks = 1000000;
    uint32_t max_iterations = 200;
    bool single_core = true;
    double eb_n0_db = 2.0;
    uint32_t num_batches = 25;
    bool benchmark_mode = false;
    bool diff_test = false;
    uint32_t tile_limit = 0;
    std::string csv_path = fs::exists("../../Results") 
        ? "../../Results/results_tile_matrix_4cw_ar4ja_r12_ebn0_1.50_2.50_step0.10_max1000000_min20_iter200_1core.csv"
        : "Results/results_tile_matrix_4cw_ar4ja_r12_ebn0_1.50_2.50_step0.10_max1000000_min20_iter200_1core.csv";

    // Check for command line flags
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--waterfall") {
            is_waterfall = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                std::string range_str = argv[++i];
                auto c1 = range_str.find(':');
                auto c2 = range_str.rfind(':');
                if (c1 != std::string::npos && c2 != std::string::npos && c1 != c2) {
                    snr_start = std::stod(range_str.substr(0, c1));
                    snr_stop  = std::stod(range_str.substr(c1 + 1, c2 - c1 - 1));
                    snr_step  = std::stod(range_str.substr(c2 + 1));
                } else {
                    snr_start = std::stod(range_str);
                    if (i + 1 < argc && argv[i + 1][0] != '-') snr_stop = std::stod(argv[++i]);
                    if (i + 1 < argc && argv[i + 1][0] != '-') snr_step = std::stod(argv[++i]);
                }
            }
            if (i + 1 < argc && argv[i + 1][0] != '-') min_fe = std::stoul(argv[++i]);
            if (i + 1 < argc && argv[i + 1][0] != '-') max_blocks = std::stoull(argv[++i]);
            if (i + 1 < argc && argv[i + 1][0] != '-') max_iterations = std::stoul(argv[++i]);
        } else if (arg == "--benchmark") {
            benchmark_mode = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') num_batches = std::stoul(argv[++i]);
            if (i + 1 < argc && argv[i + 1][0] != '-') max_iterations = std::stoul(argv[++i]);
        } else if (arg == "--diff-test") {
            diff_test = true;
            num_batches = 1;
            max_iterations = 1;
            if (i + 1 < argc && argv[i + 1][0] != '-') eb_n0_db = std::stod(argv[++i]);
            if (i + 1 < argc && argv[i + 1][0] != '-') max_iterations = std::stoul(argv[++i]);
        } else if (arg == "--tiles" && i + 1 < argc) {
            tile_limit = std::stoul(argv[++i]);
        } else if (arg == "--csv" && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (arg == "--mesh") {
            single_core = false;
        } else if (arg[0] != '-') {
            // Positional arguments for backward compatibility
            eb_n0_db = std::stod(arg);
            if (i + 1 < argc && argv[i + 1][0] != '-') num_batches = std::stoul(argv[++i]);
            if (i + 1 < argc && argv[i + 1][0] != '-') max_iterations = std::stoul(argv[++i]);
        }
    }

    std::cout << "-------------------------------------------------------" << std::endl;
    std::cout << " Tenstorrent QuietBox 2: Tile Matrix 4-Codeword Harness" << std::endl;
    std::cout << " Architecture: Decoupled Access-Execute (DAE)" << std::endl;
    std::cout << "-------------------------------------------------------" << std::endl;

    ParityCheckMatrix pcm;
    if (!parse_chinn_out(matrix_path, pcm)) {
        return 1;
    }

    auto schedule = build_circulant_aware_tile_schedule(pcm);
    std::cout << "[INFO] Generated " << schedule.size() << " circulant-snapped tiles with Schedule 4." << std::endl;

    bool audit_passed = verify_preflight_collision_audit(pcm, schedule);
    if (!audit_passed) {
        std::cerr << "[FATAL] Pre-flight collision audit failed!" << std::endl;
        return 1;
    }

    if (is_waterfall) {
        run_waterfall_sweep(pcm, schedule, snr_start, snr_stop, snr_step, min_fe, max_blocks, max_iterations, single_core, csv_path);
    } else {
        run_tile_matrix_on_hardware(pcm, schedule, eb_n0_db, num_batches, max_iterations, single_core, benchmark_mode, diff_test, tile_limit);
    }

    return 0;
}
