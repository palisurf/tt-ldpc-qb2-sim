#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <fstream>
#include <sstream>
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

    struct Edge { uint32_t r, c; float val; };
    std::vector<Edge> raw_lines;
    raw_lines.reserve(100000);

    std::string line;
    while (std::getline(file, line)) {
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos || line[start] == '#') continue;
        std::stringstream ss(line);
        uint32_t r = 0, c = 0;
        float val = 0.0f;
        if (ss >> r >> c >> val) {
            raw_lines.push_back({r, c, val});
        }
    }
    if (raw_lines.empty()) throw std::runtime_error("Matrix file empty: " + filepath);

    Edge dim_line = raw_lines.back();
    ChinnMatrix chinn;
    chinn.M = dim_line.r;
    chinn.N = dim_line.c;

    std::vector<std::vector<uint16_t>> check_nodes(chinn.M);
    std::vector<uint32_t> var_deg(chinn.N, 0);

    for (size_t i = 0; i + 1 < raw_lines.size(); i++) {
        const auto& edge = raw_lines[i];
        if (std::abs(edge.val) > 1e-6f) {
            uint32_t m_idx = edge.r - 1;
            uint32_t vn_idx = edge.c - 1;
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
        if (var_deg[n] > chinn.max_var_deg) chinn.max_var_deg = var_deg[n];
    }

    chinn.flattened_indices.assign(chinn.M * chinn.max_check_deg, 0xFFFF);
    for (uint32_t m = 0; m < chinn.M; m++) {
        for (size_t d = 0; d < check_nodes[m].size(); d++) {
            chinn.flattened_indices[m * chinn.max_check_deg + d] = check_nodes[m][d];
        }
    }
    return chinn;
}

int main() {
    std::cout << "=== Tensix Single-Core Hardware Unit Test for AR4JA ===" << std::endl;
    std::string matrix_file = "matrices/AR4JA_r45_4c_128c_r12.chinn.out";
    ChinnMatrix h_mat = parse_chinn_out(matrix_file);
    std::cout << "Host parsed: N=" << h_mat.N << ", M=" << h_mat.M 
              << ", max_check_deg=" << h_mat.max_check_deg 
              << ", first 6 entries of row 0: ";
    for (int d = 0; d < 6; d++) {
        std::cout << h_mat.flattened_indices[d] << " ";
    }
    std::cout << std::endl;

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = mesh_device->mesh_command_queue();

    CoreCoord core = {0, 0};
    CoreRange core_range(core, core);

    uint32_t tile_bytes = sizeof(bfloat16) * TILE_ELEMENTS;
    uint32_t h_bytes = h_mat.flattened_indices.size() * sizeof(uint16_t);
    uint32_t num_h_tiles = (h_bytes + tile_bytes - 1) / tile_bytes;
    uint32_t h_cb_bytes = num_h_tiles * tile_bytes;
    uint32_t stats_cb_bytes = tile_bytes;

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
    distributed::EnqueueWriteMeshBuffer(cq, h_dram, h_mat.flattened_indices, false);

    // Verify host DRAM readback:
    std::vector<uint16_t> h_dram_readback(h_mat.flattened_indices.size(), 0);
    distributed::ReadShard(cq, h_dram_readback, h_dram, distributed::MeshCoordinate(0, 0), true);
    std::cout << "DRAM Readback first 6 entries: ";
    for (int d = 0; d < 6; d++) std::cout << h_dram_readback[d] << " ";
    std::cout << std::endl;

    bool dram_matches = true;
    for (size_t i = 0; i < h_mat.flattened_indices.size(); i++) {
        if (h_mat.flattened_indices[i] != h_dram_readback[i]) {
            std::cout << "MISMATCH at idx " << i << ": expected " 
                      << h_mat.flattened_indices[i] << ", got " << h_dram_readback[i] << std::endl;
            dram_matches = false;
            break;
        }
    }
    std::cout << "DRAM Buffer Integrity: " << (dram_matches ? "MATCH" : "FAIL") << std::endl;

    // Run kernel on device with 1 codeword at 2.0 dB
    Program program = CreateProgram();

    std::map<uint8_t, tt::DataFormat> cb0_spec = {{0, tt::DataFormat::Float32}};
    std::map<uint8_t, tt::DataFormat> cb1_spec = {{1, tt::DataFormat::Float32}};
    std::map<uint8_t, tt::DataFormat> cb2_spec = {{2, tt::DataFormat::RawUInt16}};
    std::map<uint8_t, tt::DataFormat> cb16_spec = {{16, tt::DataFormat::RawUInt32}};

    CreateCircularBuffer(program, core_range, CircularBufferConfig(64 * 1024, cb0_spec).set_page_size(0, 64 * 1024));
    CreateCircularBuffer(program, core_range, CircularBufferConfig(384 * 1024, cb1_spec).set_page_size(1, 384 * 1024));
    CreateCircularBuffer(program, core_range, CircularBufferConfig(h_cb_bytes, cb2_spec).set_page_size(2, h_cb_bytes));
    CreateCircularBuffer(program, core_range, CircularBufferConfig(tile_bytes, cb16_spec).set_page_size(16, tile_bytes));

    auto reader = CreateKernel(
        program, "kernel/reader_ldpc.cpp", core_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );
    auto writer = CreateKernel(
        program, "kernel/writer_ldpc.cpp", core_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default}
    );
    auto compute = CreateKernel(
        program, "kernel/compute_trisc_ldpc_awgn_sim.cpp", core_range,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true}
    );

    uint32_t P_punctured = 512;
    uint32_t unpunctured_nodes = h_mat.N - P_punctured;
    double rate = static_cast<double>(h_mat.N - h_mat.M) / static_cast<double>(unpunctured_nodes);
    double eb_n0_db = 2.0;
    double snr_lin = std::pow(10.0, eb_n0_db / 10.0);
    float mu_llr = static_cast<float>(4.0 * rate * snr_lin);
    float sigma_llr = static_cast<float>(std::sqrt(8.0 * rate * snr_lin));

    uint32_t mu_u32 = 0, sigma_u32 = 0;
    std::memcpy(&mu_u32, &mu_llr, sizeof(float));
    std::memcpy(&sigma_u32, &sigma_llr, sizeof(float));

    std::vector<uint32_t> r_args = {static_cast<uint32_t>(h_dram->address()), 0, h_bytes, 1};
    SetRuntimeArgs(program, reader, core, r_args);

    std::vector<uint32_t> w_args = {static_cast<uint32_t>(stats_dram->address()), 0};
    SetRuntimeArgs(program, writer, core, w_args);

    uint32_t num_test_cws = 100;
    std::vector<uint32_t> c_args = {
        num_test_cws, h_mat.N, h_mat.M, P_punctured, h_mat.max_check_deg,
        mu_u32, sigma_u32, 133742,
        1, 16
    };
    SetRuntimeArgs(program, compute, core, c_args);

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    cq.finish();

    std::vector<uint32_t> stats(stats_cb_bytes / sizeof(uint32_t), 0);
    distributed::ReadShard(cq, stats, stats_dram, distributed::MeshCoordinate(0, 0), true);

    std::cout << "\n=== Device Execution Results ===" << std::endl;
    std::cout << "Codewords: " << num_test_cws << std::endl;
    std::cout << "Bit Errors: " << stats[0] << std::endl;
    std::cout << "Frame Errors: " << stats[1] << " (" << (double)stats[1] / num_test_cws * 100.0 << "%)" << std::endl;

    return 0;
}
