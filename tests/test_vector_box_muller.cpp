#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <cstring>
#include <chrono>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/circular_buffer_config.hpp>

using namespace tt;
using namespace tt::tt_metal;

struct DistributionMoments {
    double mean = 0.0;
    double variance = 0.0;
    double stddev = 0.0;
    double skewness = 0.0;
    double kurtosis = 0.0;
};

DistributionMoments calculate_moments(const std::vector<float>& data) {
    DistributionMoments m;
    size_t n = data.size();
    if (n == 0) return m;

    double sum = 0.0;
    for (float x : data) sum += x;
    m.mean = sum / n;

    double m2 = 0.0, m3 = 0.0, m4 = 0.0;
    for (float x : data) {
        double d = x - m.mean;
        double d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }

    m.variance = m2 / n;
    m.stddev = std::sqrt(m.variance);
    if (m.variance > 1e-12) {
        m.skewness = (m3 / n) / (m.variance * m.stddev);
        m.kurtosis = ((m4 / n) / (m.variance * m.variance)) - 3.0; // Excess kurtosis
    }
    return m;
}

int main() {
    std::cout << "======================================================================" << std::endl;
    std::cout << " SILICON BENCHMARK: 32-LANE VECTOR CARTESIAN BOX-MULLER AWGN GENERATOR" << std::endl;
    std::cout << " Target: Physical Tensix Core (0, 0) | QuietBox 2 (ttuser@192.168.0.195) " << std::endl;
    std::cout << "======================================================================" << std::endl;

    try {
        auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh_device->mesh_command_queue();

        CoreCoord core = {0, 0};
        CoreRange core_range(core, core);

        Program program = CreateProgram();

        constexpr uint32_t TILE_BYTES = 2048; // 1,024 bfloat16 elements
        constexpr uint32_t NUM_TILES = 10;   // 8 AWGN + 2 Punctured (20 KB total)
        constexpr uint32_t TOTAL_BUFFER_BYTES = NUM_TILES * TILE_BYTES;
        constexpr uint32_t STATS_BYTES = 64;

        // Circular Buffer for Output Tiles (CB 16)
        CreateCircularBuffer(program, core_range,
            CircularBufferConfig(TILE_BYTES * 2, {{tt::CBIndex::c_16, tt::DataFormat::Float16_b}})
                .set_page_size(tt::CBIndex::c_16, TILE_BYTES));

        // Circular Buffer for Telemetry (CB 17)
        CreateCircularBuffer(program, core_range,
            CircularBufferConfig(2048, {{tt::CBIndex::c_17, tt::DataFormat::RawUInt32}})
                .set_page_size(tt::CBIndex::c_17, 2048));

        // DRAM Buffers
        distributed::DeviceLocalBufferConfig out_dram_cfg{
            .page_size = TILE_BYTES,
            .buffer_type = BufferType::DRAM
        };
        auto out_dram = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = TOTAL_BUFFER_BYTES}, out_dram_cfg, mesh_device.get()
        );

        distributed::DeviceLocalBufferConfig stats_dram_cfg{
            .page_size = STATS_BYTES,
            .buffer_type = BufferType::DRAM
        };
        auto stats_dram = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = STATS_BYTES}, stats_dram_cfg, mesh_device.get()
        );

        // Standard AWGN Parameters at Eb/N0 = 2.0 dB, Rate R = 1/2
        double ebn0_db = 2.0;
        double rate = 0.5;
        double ebn0_linear = std::pow(10.0, ebn0_db / 10.0);
        double sigma_channel = 1.0 / std::sqrt(2.0 * rate * ebn0_linear);
        float mu_llr = static_cast<float>(2.0 / (sigma_channel * sigma_channel));
        float sigma_llr = static_cast<float>(2.0 / sigma_channel);

        uint32_t mu_bits, sigma_bits;
        std::memcpy(&mu_bits, &mu_llr, 4);
        std::memcpy(&sigma_bits, &sigma_llr, 4);

        std::cout << "[CONFIG] Channel Parameters (Eb/N0 = " << ebn0_db << " dB):" << std::endl;
        std::cout << "  - Channel Sigma    : " << sigma_channel << std::endl;
        std::cout << "  - Theoretical Mean : " << mu_llr << std::endl;
        std::cout << "  - Theoretical Sigma: " << sigma_llr << std::endl;
        std::cout << "  - Theoretical Var  : " << sigma_llr * sigma_llr << std::endl;

        // Kernels
        auto writer = CreateKernel(
            program, "tests/kernel_vector_box_muller_writer.cpp", core_range,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default}
        );
        auto compute = CreateKernel(
            program, "tests/kernel_vector_box_muller.cpp", core_range,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true}
        );

        std::vector<uint32_t> w_args = {
            static_cast<uint32_t>(out_dram->address()),
            static_cast<uint32_t>(stats_dram->address()),
            0, // dram_bank
            TILE_BYTES,
            STATS_BYTES,
            NUM_TILES
        };
        SetRuntimeArgs(program, writer, core, w_args);

        std::vector<uint32_t> c_args = {
            NUM_TILES,
            mu_bits,
            sigma_bits,
            123456789U // seed_base
        };
        SetRuntimeArgs(program, compute, core, c_args);

        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));

        std::cout << "\n[RUN] Dispatching Vector Box-Muller Workload to Blackhole Core (0, 0)..." << std::endl;
        auto t_start = std::chrono::high_resolution_clock::now();
        distributed::EnqueueMeshWorkload(cq, workload, false);
        cq.finish();
        auto t_end = std::chrono::high_resolution_clock::now();
        double wall_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        // Read Back Output Data
        std::vector<bfloat16> hw_out_buffer(NUM_TILES * 1024);
        distributed::ReadShard(cq, hw_out_buffer, out_dram, distributed::MeshCoordinate(0, 0), true);

        std::vector<uint32_t> stats_data(STATS_BYTES / sizeof(uint32_t));
        distributed::ReadShard(cq, stats_data, stats_dram, distributed::MeshCoordinate(0, 0), true);

        uint32_t measured_total_cycles = stats_data[0];
        uint32_t sentinel = stats_data[1];

        // Statistical Verification of Transmitted AWGN Samples (Tiles 0..7 = 8,192 samples)
        std::vector<float> awgn_samples;
        awgn_samples.reserve(8192);
        for (size_t i = 0; i < 8 * 1024; ++i) {
            awgn_samples.push_back(static_cast<float>(hw_out_buffer[i]));
        }

        DistributionMoments moments = calculate_moments(awgn_samples);

        // Verification of Punctured Nodes (Tiles 8..9 = 2,048 samples strictly 0.0)
        uint32_t punctured_nonzeros = 0;
        for (size_t i = 8 * 1024; i < 10 * 1024; ++i) {
            if (static_cast<float>(hw_out_buffer[i]) != 0.0f) {
                punctured_nonzeros++;
            }
        }

        double cyc_per_sample = static_cast<double>(measured_total_cycles) / 8192.0;
        double aiclk_ghz = 1.35;
        double hw_us = (measured_total_cycles / (aiclk_ghz * 1e9)) * 1e6;

        std::cout << "\n======================================================================" << std::endl;
        std::cout << " PHYSICAL BLACKHOLE SILICON RESULTS: VECTOR BOX-MULLER AWGN GENERATOR " << std::endl;
        std::cout << "======================================================================" << std::endl;
        std::cout << " Hardware Sentinel      : 0x" << std::hex << sentinel << std::dec << " ("
                  << (sentinel == 0xAA55AA55 ? "PASSED - NO HANGS" : "CORRUPT") << ")" << std::endl;
        std::cout << " Total Measured Cycles  : " << measured_total_cycles << " clock cycles (10 tiles = 20 KB)" << std::endl;
        std::cout << " Execution Latency      : " << std::fixed << std::setprecision(2) << hw_us << " microseconds (" << hw_us * 1e3 << " ns)" << std::endl;
        std::cout << " Cycles per Sample      : " << std::fixed << std::setprecision(2) << cyc_per_sample << " cycles/sample (8,192 AWGN samples)" << std::endl;
        std::cout << " Host Wall Clock Time   : " << std::fixed << std::setprecision(2) << wall_ms << " ms" << std::endl;

        std::cout << "\n [Statistical Moments Analysis (8,192 Transmitted Samples)]" << std::endl;
        std::cout << "  - Measured Mean LLR   : " << std::fixed << std::setprecision(4) << moments.mean 
                  << " (Theory: " << mu_llr << ", Diff: " << std::abs(moments.mean - mu_llr) << ")" << std::endl;
        std::cout << "  - Measured Var LLR    : " << std::fixed << std::setprecision(4) << moments.variance 
                  << " (Theory: " << sigma_llr * sigma_llr << ", Diff: " << std::abs(moments.variance - sigma_llr * sigma_llr) << ")" << std::endl;
        std::cout << "  - Measured StdDev     : " << std::fixed << std::setprecision(4) << moments.stddev 
                  << " (Theory: " << sigma_llr << ")" << std::endl;
        std::cout << "  - Measured Skewness   : " << std::fixed << std::setprecision(4) << moments.skewness 
                  << " (Ideal: 0.0000)" << std::endl;
        std::cout << "  - Excess Kurtosis     : " << std::fixed << std::setprecision(4) << moments.kurtosis 
                  << " (Ideal: 0.0000)" << std::endl;

        std::cout << "\n [Punctured Nodes Verification (Tiles 8 & 9)]" << std::endl;
        std::cout << "  - Punctured Non-Zeros : " << punctured_nonzeros << " / 2048 (Must be 0)" << std::endl;

        std::cout << "\n [Individual Per-Tile Latencies (10 Tiles)]" << std::endl;
        for (int t = 0; t < 10; ++t) {
            std::cout << "   Tile " << t << " (" << (t < 8 ? "AWGN Transmitted" : "Punctured Zeros") << "): "
                      << std::setw(5) << stats_data[2 + t] << " clock cycles" << std::endl;
        }

        bool stats_ok = (std::abs(moments.mean - mu_llr) < 0.5f) && 
                         (std::abs(moments.variance - sigma_llr * sigma_llr) < 1.0f) &&
                         (punctured_nonzeros == 0) &&
                         (sentinel == 0xAA55AA55);

        std::cout << "\n======================================================================" << std::endl;
        std::cout << " STATUS: " << (stats_ok ? ">>> [100% BENCHMARK PASSED] <<<" : "[BENCHMARK FAILED]") << std::endl;
        std::cout << "======================================================================" << std::endl;

        mesh_device->close();
        return stats_ok ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception caught: " << e.what() << std::endl;
        return 1;
    }
}
