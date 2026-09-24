#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/circular_buffer_config.hpp>

using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char** argv) {
    double eb_n0_db = 2.0; // Default to lower SNR (2.0 dB) to observe channel noise & decoder errors
    if (argc > 1) {
        try {
            eb_n0_db = std::stod(argv[1]);
        } catch (...) {
            std::cerr << "Invalid SNR argument, using default 2.0 dB" << std::endl;
        }
    }
    std::cout << "==========================================================" << std::endl;
    std::cout << "  Tenstorrent Tensix Single-Core Kernel Unit Test (ttsim) " << std::endl;
    std::cout << "==========================================================\n" << std::endl;

    // Create unit mesh (Single Blackhole chip / core)
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = mesh_device->mesh_command_queue();

    CoreCoord core = {0, 0};
    CoreRange core_range(core, core);
    Program program = CreateProgram();

    // Use small test dimensions (N = 12, M = 6, max_check_deg = 4)
    constexpr uint32_t N_var = 12;
    constexpr uint32_t M_check = 6;
    constexpr uint32_t max_check_deg = 4;
    constexpr uint32_t P_punctured = 0;
    constexpr uint32_t num_codewords = 100;

    // Compressed parity check matrix for test_code.chinn
    std::vector<uint16_t> h_matrix = {
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 10, 11,
        0, 4, 8, 2,
        1, 5, 9, 6,
        3, 7, 10, 11
    };

    uint32_t tile_bytes = 2048; // Standard tile size
    uint32_t h_bytes = h_matrix.size() * sizeof(uint16_t);
    uint32_t num_h_tiles = (h_bytes + tile_bytes - 1) / tile_bytes;
    uint32_t h_cb_bytes = num_h_tiles * tile_bytes;

    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = tile_bytes,
        .buffer_type = BufferType::DRAM
    };

    // Allocate DRAM buffers on mesh device
    auto h_dram = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = h_cb_bytes}, dram_config, mesh_device.get()
    );
    auto stats_dram = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = tile_bytes}, dram_config, mesh_device.get()
    );

    cq.enqueue_write_mesh_buffer(h_dram, h_matrix.data(), false);

    // Setup Circular Buffers
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

    // Channel SNR parameters from CLI (default 2.0 dB)
    double rate = static_cast<double>(N_var - M_check) / static_cast<double>(N_var);
    double snr_lin = std::pow(10.0, eb_n0_db / 10.0);
    float mu_llr = static_cast<float>(4.0 * rate * snr_lin);
    float sigma_llr = static_cast<float>(std::sqrt(8.0 * rate * snr_lin));
    uint32_t mu_u32 = 0;
    uint32_t sigma_u32 = 0;
    std::memcpy(&mu_u32, &mu_llr, sizeof(float));
    std::memcpy(&sigma_u32, &sigma_llr, sizeof(float));

    std::vector<uint32_t> reader_args = {
        static_cast<uint32_t>(h_dram->address()), 0, h_bytes, num_h_tiles
    };
    SetRuntimeArgs(program, reader, core, reader_args);

    std::vector<uint32_t> writer_args = {
        static_cast<uint32_t>(stats_dram->address()), 0
    };
    SetRuntimeArgs(program, writer, core, writer_args);

    std::vector<uint32_t> compute_args = {
        num_codewords, N_var, M_check, P_punctured, max_check_deg,
        mu_u32, sigma_u32, 133742, num_h_tiles
    };
    SetRuntimeArgs(program, compute, core, compute_args);

    std::cout << "[SIMULATION] Launching kernel on Tensix Core (0, 0)..." << std::endl;
    std::cout << "  - Eb/N0:              " << eb_n0_db << " dB" << std::endl;
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    cq.enqueue_mesh_workload(workload, false);
    cq.finish();

    std::vector<uint32_t> results(tile_bytes / sizeof(uint32_t), 0);
    cq.enqueue_read_mesh_buffer(results.data(), stats_dram, true);
    cq.finish();

    uint32_t total_bit_errors = results[0];
    uint32_t total_frame_errors = results[1];
    uint32_t total_bits = num_codewords * (N_var - P_punctured);
    double ber = static_cast<double>(total_bit_errors) / static_cast<double>(total_bits);
    double fer = static_cast<double>(total_frame_errors) / static_cast<double>(num_codewords);

    std::cout << "  - Blocks Simulated:   " << num_codewords << std::endl;
    std::cout << "  - Total Bits:         " << total_bits << std::endl;
    std::cout << "  - Total Bit Errors:   " << total_bit_errors << " (BER = " << ber << ")" << std::endl;
    std::cout << "  - Total Frame Errors: " << total_frame_errors << " (FER = " << fer << ")" << std::endl;

    mesh_device->close();

    if (total_bit_errors > 0 || total_frame_errors > 0) {
        std::cout << "\n>>> CONFIRMED: Non-zero errors detected at Eb/N0 = " << eb_n0_db << " dB! <<<\n" << std::endl;
        return 0;
    } else {
        std::cout << "\n>>> Zero errors detected at Eb/N0 = " << eb_n0_db << " dB. <<<\n" << std::endl;
        return 0;
    }
}
