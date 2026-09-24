#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <getopt.h>
#include <chrono>
#include <cstring>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/circular_buffer_config.hpp>

using namespace tt;
using namespace tt::tt_metal;

constexpr uint32_t MAX_N = 16384;
constexpr uint32_t MAX_M = 8192;
constexpr uint32_t MAX_DEG = 12;
constexpr uint32_t TILE_ELEMENTS = 1024;

struct ChinnMatrix {
    uint32_t N = 0;
    uint32_t M = 0;
    uint32_t max_var_deg = 0;
    uint32_t max_check_deg = 0;
    std::vector<uint16_t> flattened_indices;
};

ChinnMatrix parse_chinn_out(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) throw std::runtime_error("Could not open matrix file: " + filepath);

    struct Edge {
        uint32_t r;
        uint32_t c;
        float val;
    };
    std::vector<Edge> raw_lines;
    raw_lines.reserve(100000);

    std::string line;
    while (std::getline(file, line)) {
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        if (line[start] == '#') continue;
        std::stringstream ss(line);
        uint32_t r = 0, c = 0;
        float val = 0.0f;
        if (ss >> r >> c >> val) {
            raw_lines.push_back({r, c, val});
        }
    }

    if (raw_lines.empty()) {
        throw std::runtime_error("Matrix file is empty: " + filepath);
    }

    // The last line specifies the total matrix dimensions: M (rows/checks), N (cols/variables)
    Edge dim_line = raw_lines.back();
    ChinnMatrix chinn;
    chinn.M = dim_line.r;
    chinn.N = dim_line.c;

    if (chinn.N > MAX_N || chinn.M > MAX_M) {
        throw std::runtime_error("Matrix dimensions exceed supported device limits: N=" + 
                                 std::to_string(chinn.N) + ", M=" + std::to_string(chinn.M));
    }

    std::vector<std::vector<uint16_t>> check_nodes(chinn.M);
    std::vector<uint32_t> var_deg(chinn.N, 0);

    // Process all edges before the dimension line
    for (size_t i = 0; i + 1 < raw_lines.size(); i++) {
        const auto& edge = raw_lines[i];
        if (std::abs(edge.val) > 1e-6f) {
            if (edge.r == 0 || edge.r > chinn.M || edge.c == 0 || edge.c > chinn.N) {
                throw std::runtime_error("Edge index out of bounds in " + filepath + 
                                         ": row=" + std::to_string(edge.r) + ", col=" + std::to_string(edge.c));
            }
            uint32_t m_idx = edge.r - 1;   // 1-indexed check node to 0-indexed
            uint32_t vn_idx = edge.c - 1;  // 1-indexed variable node to 0-indexed
            check_nodes[m_idx].push_back(static_cast<uint16_t>(vn_idx));
            var_deg[vn_idx]++;
        }
    }

    chinn.max_check_deg = 0;
    for (uint32_t m = 0; m < chinn.M; m++) {
        if (check_nodes[m].size() > chinn.max_check_deg) {
            chinn.max_check_deg = static_cast<uint32_t>(check_nodes[m].size());
        }
    }
    chinn.max_var_deg = 0;
    for (uint32_t n = 0; n < chinn.N; n++) {
        if (var_deg[n] > chinn.max_var_deg) {
            chinn.max_var_deg = var_deg[n];
        }
    }

    if (chinn.max_check_deg > MAX_DEG) {
        throw std::runtime_error("Check node degree (" + std::to_string(chinn.max_check_deg) + 
                                 ") exceeds MAX_DEG (" + std::to_string(MAX_DEG) + ")");
    }

    chinn.flattened_indices.assign(chinn.M * chinn.max_check_deg, 0xFFFF);
    for (uint32_t m = 0; m < chinn.M; m++) {
        for (size_t d = 0; d < check_nodes[m].size(); d++) {
            chinn.flattened_indices[m * chinn.max_check_deg + d] = check_nodes[m][d];
        }
    }

    return chinn;
}

ChinnMatrix parse_chinn(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) throw std::runtime_error("Could not open matrix file: " + filepath);

    std::string line;
    auto get_clean_line = [&file, &line]() -> bool {
        while (std::getline(file, line)) {
            size_t start = line.find_first_not_of(" \t\r\n");
            if (start == std::string::npos) continue;
            if (line[start] == '#') continue;
            line = line.substr(start);
            return true;
        }
        return false;
    };

    if (!get_clean_line()) throw std::runtime_error("Invalid or empty matrix file: " + filepath);

    // Auto-detect format: 3 tokens per line indicates .chinn.out coordinate format
    std::stringstream test_ss(line);
    std::vector<std::string> first_tokens;
    std::string tok;
    while (test_ss >> tok) first_tokens.push_back(tok);
    file.close();

    if (first_tokens.size() == 3) {
        return parse_chinn_out(filepath);
    }

    // Standard .chinn format
    file.open(filepath);
    if (!get_clean_line()) throw std::runtime_error("Invalid or empty matrix file: " + filepath);

    std::stringstream ss(line);
    ChinnMatrix chinn;
    ss >> chinn.N >> chinn.M;
    if (!(ss >> chinn.max_var_deg >> chinn.max_check_deg)) {
        if (!get_clean_line()) throw std::runtime_error("Missing degree header in matrix file: " + filepath);
        std::stringstream ss_deg(line);
        ss_deg >> chinn.max_var_deg >> chinn.max_check_deg;
    }
    
    if (chinn.N > MAX_N || chinn.M > MAX_M) {
        throw std::runtime_error("Matrix dimensions exceed supported device limits");
    }

    chinn.flattened_indices.assign(chinn.M * chinn.max_check_deg, 0xFFFF); // Sentinel initialized

    for (uint32_t m = 0; m < chinn.M; m++) {
        if (!get_clean_line()) throw std::runtime_error("Unexpected EOF reading check node row " + std::to_string(m));
        std::stringstream row_ss(line);
        uint32_t deg = 0;
        row_ss >> deg;
        for (uint32_t d = 0; d < deg; d++) {
            uint32_t vn_idx = 0;
            row_ss >> vn_idx;
            if (vn_idx > 0) vn_idx -= 1; // Convert 1-indexed to 0-indexed
            if (d < chinn.max_check_deg) {
                chinn.flattened_indices[m * chinn.max_check_deg + d] = static_cast<uint16_t>(vn_idx);
            }
        }
    }
    return chinn;
}

int main(int argc, char** argv) {
    std::string chinn_file = "";
    double eb_n0_db = 2.0;
    uint32_t P_punctured = 0;
    uint64_t target_frame_errors = 50;
    uint64_t blocks_per_core = 100000ULL;
    uint64_t max_blocks = 0;
    uint32_t batch_per_core = 1000; // Dispatch batch size per core for error threshold evaluation
    uint32_t max_iterations = 16;   // Max decoder iterations per codeword (default: 16)

    bool single_core = false;
    bool lockstep_verify = false;
    bool use_nms = false;

    int opt;
    while ((opt = getopt(argc, argv, "c:e:p:t:m:b:k:i:svn")) != -1) {
        switch (opt) {
            case 'c': chinn_file = optarg; break;
            case 'e': eb_n0_db = std::stod(optarg); break;
            case 'p': P_punctured = std::stoul(optarg); break;
            case 't': target_frame_errors = std::stoull(optarg); break;
            case 'm': max_blocks = std::stoull(optarg); break;
            case 'b': batch_per_core = std::stoul(optarg); break;
            case 'k': blocks_per_core = std::stoull(optarg); break;
            case 'i': max_iterations = std::stoul(optarg); break;
            case 's': single_core = true; break;
            case 'v': lockstep_verify = true; break;
            case 'n': use_nms = true; break;
            default: break;
        }
    }

    ChinnMatrix h_mat = parse_chinn(chinn_file);
    std::cout << "[CONFIG] Matrix loaded: " << chinn_file 
              << " (N=" << h_mat.N << ", M=" << h_mat.M 
              << ", max_check_deg=" << h_mat.max_check_deg 
              << ", max_var_deg=" << h_mat.max_var_deg << ")" << std::endl;

    // Initialize Mesh Device: Unit Mesh (1 core) if single_core, else 2x2 Mesh (4 chips = 480 cores)
    std::shared_ptr<distributed::MeshDevice> mesh_device;
    if (single_core) {
        mesh_device = distributed::MeshDevice::create_unit_mesh(0);
        std::cout << "[CONFIG] Running in SINGLE-CORE mode on Tensix Core (0, 0)" << std::endl;
    } else {
        distributed::MeshDeviceConfig mesh_config(distributed::MeshShape{2, 2});
        mesh_device = distributed::MeshDevice::create(mesh_config);
        std::cout << "[CONFIG] Running in MULTI-CORE mode on 2x2 Mesh (4 chips)" << std::endl;
    }
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

    CoreCoord grid_size = single_core ? CoreCoord{1, 1} : mesh_device->compute_with_storage_grid_size();
    CoreRange core_grid({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    uint32_t cores_per_chip = grid_size.x * grid_size.y;
    uint32_t total_mesh_cores = cores_per_chip * mesh_device->num_rows() * mesh_device->num_cols();

    if (max_blocks == 0 && blocks_per_core > 0) {
        max_blocks = blocks_per_core * total_mesh_cores;
    }
    if (blocks_per_core > 0 && batch_per_core > blocks_per_core) {
        batch_per_core = static_cast<uint32_t>(blocks_per_core);
    }

    std::cout << "[CONFIG] Compute grid size per chip: " << grid_size.x << "x" << grid_size.y 
              << " (" << cores_per_chip << " compute cores/chip, " 
              << total_mesh_cores << " total across " 
              << (mesh_device->num_rows() * mesh_device->num_cols()) << " chips)" << std::endl;
    std::cout << "[CONFIG] Stopping criteria: min " << target_frame_errors << " block errors OR max " 
              << blocks_per_core << " blocks/core (Batch size/core: " << batch_per_core 
              << ", max mesh blocks: " << max_blocks << ", max iters: " << max_iterations << ")" << std::endl;
    std::cout << "[CONFIG] Decoder algorithm: " << (use_nms ? "Normalized Min-Sum (alpha=0.75)" : "Approximate-Min* (Christopher Jones MILCOM 2003)") << std::endl;

    uint32_t tile_bytes = sizeof(bfloat16) * TILE_ELEMENTS;
    uint32_t h_bytes = h_mat.flattened_indices.size() * sizeof(uint16_t);
    uint32_t num_h_tiles = (h_bytes + tile_bytes - 1) / tile_bytes;
    uint32_t h_cb_bytes = num_h_tiles * tile_bytes;

    uint32_t stats_raw_bytes = cores_per_chip * 4 * sizeof(uint32_t);
    uint32_t num_stats_tiles = (stats_raw_bytes + tile_bytes - 1) / tile_bytes;
    uint32_t stats_cb_bytes = num_stats_tiles * tile_bytes;

    distributed::DeviceLocalBufferConfig h_dram_config{
        .page_size = h_cb_bytes, 
        .buffer_type = BufferType::DRAM
    };
    distributed::DeviceLocalBufferConfig stats_dram_config{
        .page_size = tile_bytes, 
        .buffer_type = BufferType::DRAM
    };

    auto h_dram = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = h_cb_bytes}, h_dram_config, mesh_device.get()
    );
    auto stats_dram = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = stats_cb_bytes}, stats_dram_config, mesh_device.get()
    );

    h_mat.flattened_indices.resize(h_cb_bytes / sizeof(uint16_t), 0xFFFF);
    EnqueueWriteMeshBuffer(cq, h_dram, h_mat.flattened_indices, false);

    // True Punctured Code Rate: R = (N - M) / (N - P)
    double rate = static_cast<double>(h_mat.N - h_mat.M) / static_cast<double>(h_mat.N - P_punctured);
    double snr_lin = std::pow(10.0, eb_n0_db / 10.0);
    float mu_llr = static_cast<float>(4.0 * rate * snr_lin);
    float sigma_llr = static_cast<float>(std::sqrt(8.0 * rate * snr_lin));

    uint32_t mu_u32 = 0;
    uint32_t sigma_u32 = 0;
    std::memcpy(&mu_u32, &mu_llr, sizeof(float));
    std::memcpy(&sigma_u32, &sigma_llr, sizeof(float));

    uint64_t accumulated_blocks = 0;
    uint64_t accumulated_bit_errs = 0;
    uint64_t accumulated_frame_errs = 0;

    auto sim_start = std::chrono::high_resolution_clock::now();

    while (accumulated_frame_errs < target_frame_errors && accumulated_blocks < max_blocks) {
        uint64_t remaining_blocks = max_blocks - accumulated_blocks;
        uint32_t current_batch_per_core = batch_per_core;
        if (remaining_blocks < static_cast<uint64_t>(batch_per_core) * total_mesh_cores) {
            current_batch_per_core = static_cast<uint32_t>(std::max<uint64_t>(1, remaining_blocks / total_mesh_cores));
        }
        uint32_t current_batch_total = current_batch_per_core * total_mesh_cores;

        Program program = CreateProgram();

        std::map<uint8_t, tt::DataFormat> cb0_spec = {{0, tt::DataFormat::Float16_b}};
        std::map<uint8_t, tt::DataFormat> cb1_spec = {{1, tt::DataFormat::Float16_b}};
        std::map<uint8_t, tt::DataFormat> cb2_spec = {{2, tt::DataFormat::RawUInt16}};
        std::map<uint8_t, tt::DataFormat> cb16_spec = {{16, tt::DataFormat::RawUInt32}};

        // CB 0: Channel LLR Scratchpad (32 KB - bfloat16)
        CreateCircularBuffer(program, core_grid, CircularBufferConfig(32 * 1024, cb0_spec).set_page_size(0, 32 * 1024));
        // CB 1: Check Message r_msg Scratchpad (128 KB - bfloat16)
        CreateCircularBuffer(program, core_grid, CircularBufferConfig(128 * 1024, cb1_spec).set_page_size(1, 128 * 1024));
        // CB 2: Sparse Parity Matrix Input
        CreateCircularBuffer(program, core_grid, CircularBufferConfig(h_cb_bytes, cb2_spec).set_page_size(2, h_cb_bytes));
        // CB 16: Error Statistics Output
        CreateCircularBuffer(program, core_grid, CircularBufferConfig(tile_bytes, cb16_spec).set_page_size(16, tile_bytes));

        auto reader = CreateKernel(
            program, "kernel/reader_ldpc.cpp", core_grid,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
        );

        auto writer = CreateKernel(
            program, "kernel/writer_ldpc.cpp", core_grid,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default}
        );

        std::map<std::string, std::string> compute_defines;
        if (use_nms) {
            compute_defines["USE_NORMALIZED_MIN_SUM"] = "1";
        }

        auto compute = CreateKernel(
            program, "kernel/compute_trisc_ldpc_awgn_sim.cpp", core_grid,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = true,
                .defines = compute_defines
            }
        );

        for (uint32_t core_id = 0; core_id < cores_per_chip; core_id++) {
            CoreCoord core = {core_id % grid_size.x, core_id / grid_size.x};
            uint32_t core_stats_addr = static_cast<uint32_t>(stats_dram->address()) + (core_id * 4 * sizeof(uint32_t));

            std::vector<uint32_t> r_args = {static_cast<uint32_t>(h_dram->address()), 0, h_bytes, 1};
            SetRuntimeArgs(program, reader, core, r_args);

            std::vector<uint32_t> w_args = {core_stats_addr, 0};
            SetRuntimeArgs(program, writer, core, w_args);

            uint32_t seed_lo = lockstep_verify ? 133742 : (1337 + core_id * 10007 + static_cast<uint32_t>(accumulated_blocks));
            uint32_t seed_hi = lockstep_verify ? 0x9E3779B9 : (core_id * 0x85EBCA6B + static_cast<uint32_t>(accumulated_blocks >> 32) + 0x12345678);
            std::vector<uint32_t> c_args = {
                current_batch_per_core, h_mat.N, h_mat.M, P_punctured, h_mat.max_check_deg,
                mu_u32, sigma_u32, seed_lo,
                seed_hi, max_iterations
            };
            SetRuntimeArgs(program, compute, core, c_args);
        }

        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, false);
        cq.finish();

        // Aggregate statistics from all chips using ReadShard
        uint32_t ref_bit_err = 0, ref_frame_err = 0, ref_iters = 0, ref_llr0 = 0;
        bool has_ref = false;
        uint32_t total_verified_cores = 0;
        uint32_t mismatch_count = 0;

        for (uint32_t row = 0; row < mesh_device->num_rows(); row++) {
            for (uint32_t col = 0; col < mesh_device->num_cols(); col++) {
                std::vector<uint32_t> chip_stats(stats_cb_bytes / sizeof(uint32_t), 0);
                distributed::ReadShard(cq, chip_stats, stats_dram, distributed::MeshCoordinate(row, col), true);

                if (lockstep_verify && row == 0 && col == 0) {
                    std::cout << "[DEBUG] First 5 cores on Chip (0,0):" << std::endl;
                    for (uint32_t i = 0; i < 5; i++) {
                        float llr0_f = 0.0f;
                        std::memcpy(&llr0_f, &chip_stats[4 * i + 3], sizeof(float));
                        std::cout << "  Core " << i << ": bit_err=" << chip_stats[4 * i + 0]
                                  << ", frame_err=" << chip_stats[4 * i + 1]
                                  << ", iters=" << chip_stats[4 * i + 2]
                                  << ", llr0=" << llr0_f << " (0x" << std::hex << chip_stats[4 * i + 3] << std::dec << ")"
                                  << std::endl;
                    }
                }

                for (uint32_t i = 0; i < cores_per_chip; i++) {
                    uint32_t b_err = chip_stats[4 * i + 0];
                    uint32_t f_err = chip_stats[4 * i + 1];
                    uint32_t iters = chip_stats[4 * i + 2];
                    uint32_t llr0  = chip_stats[4 * i + 3];

                    if (lockstep_verify) {
                        if (!has_ref) {
                            ref_bit_err = b_err;
                            ref_frame_err = f_err;
                            ref_iters = iters;
                            ref_llr0 = llr0;
                            has_ref = true;
                        } else {
                            if (b_err != ref_bit_err || f_err != ref_frame_err || iters != ref_iters || llr0 != ref_llr0) {
                                if (mismatch_count < 10) {
                                    std::cerr << "[MISMATCH] Chip(" << row << "," << col << ") Core " << i 
                                              << ": got {" << b_err << "," << f_err << "," << iters << ",0x" << std::hex << llr0 << "}"
                                              << " expected {" << ref_bit_err << "," << ref_frame_err << "," << ref_iters << ",0x" << ref_llr0 << "}"
                                              << std::dec << std::endl;
                                }
                                mismatch_count++;
                            }
                        }
                    }

                    accumulated_bit_errs += b_err;
                    accumulated_frame_errs += f_err;
                    total_verified_cores++;
                }
            }
        }

        if (lockstep_verify) {
            float ref_llr0_f = 0.0f;
            std::memcpy(&ref_llr0_f, &ref_llr0, sizeof(float));
            if (mismatch_count == 0) {
                std::cout << "[VERIFICATION PASS] All " << total_verified_cores 
                          << " cores produced IDENTICAL output: bit_err=" << ref_bit_err 
                          << ", frame_err=" << ref_frame_err 
                          << ", iters=" << ref_iters 
                          << ", llr0=" << ref_llr0_f 
                          << " (0x" << std::hex << ref_llr0 << std::dec << ")" << std::endl;
            } else {
                std::cerr << "[VERIFICATION FAIL] " << mismatch_count << " out of " << total_verified_cores 
                          << " cores differed from reference!" << std::endl;
            }
        }

        accumulated_blocks += current_batch_total;

        double curr_elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - sim_start).count();
        double curr_ber = accumulated_blocks > 0 ? static_cast<double>(accumulated_bit_errs) / (accumulated_blocks * (h_mat.N - P_punctured)) : 0.0;
        double curr_fer = accumulated_blocks > 0 ? static_cast<double>(accumulated_frame_errs) / accumulated_blocks : 0.0;
        double curr_msps = curr_elapsed > 0 ? (accumulated_blocks * (h_mat.N - P_punctured) / curr_elapsed) / 1e6 : 0.0;
        std::cout << "{\"progress\": true, \"ebn0\": " << eb_n0_db 
                  << ", \"blocks\": " << accumulated_blocks 
                  << ", \"max_blocks\": " << max_blocks
                  << ", \"bit_errors\": " << accumulated_bit_errs 
                  << ", \"frame_errors\": " << accumulated_frame_errs
                  << ", \"target_errors\": " << target_frame_errors
                  << ", \"ber\": " << curr_ber
                  << ", \"fer\": " << curr_fer
                  << ", \"throughput_msps\": " << curr_msps
                  << ", \"elapsed_sec\": " << curr_elapsed << "}" << std::endl;
    }

    auto sim_end = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(sim_end - sim_start).count();

    std::cout << "{\"ebn0\": " << eb_n0_db 
              << ", \"blocks\": " << accumulated_blocks 
              << ", \"bit_errors\": " << accumulated_bit_errs 
              << ", \"frame_errors\": " << accumulated_frame_errs
              << ", \"elapsed_sec\": " << elapsed_sec << "}" << std::endl;

    mesh_device->close();
    return 0;
}