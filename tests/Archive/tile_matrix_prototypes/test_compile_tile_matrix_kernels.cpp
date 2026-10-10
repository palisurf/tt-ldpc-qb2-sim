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

#include "../engines/tile_matrix_4cw/ldpc_tile_matrix_core.h"

using namespace tt;
using namespace tt::tt_metal;

int main() {
    std::cout << "=======================================================" << std::endl;
    std::cout << " Testing Compilation of engines/tile_matrix_4cw Kernels" << std::endl;
    std::cout << "=======================================================" << std::endl;

    try {
        auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh_device->mesh_command_queue();

        CoreCoord core = {0, 0};
        CoreRange core_range(core, core);

        Program program = CreateProgram();

        // CB Allocations in Tensix L1 SRAM:
        // CB 0: Active variable posteriors (20 KB)
        CreateCircularBuffer(program, core_range, CircularBufferConfig(20 * 1024, {{0, tt::DataFormat::Float16_b}}).set_page_size(0, 20 * 1024));
        // CB 1: Check messages (60 KB)
        CreateCircularBuffer(program, core_range, CircularBufferConfig(60 * 1024, {{1, tt::DataFormat::Float16_b}}).set_page_size(1, 60 * 1024));
        // CB 2: H offsets (16 KB)
        CreateCircularBuffer(program, core_range, CircularBufferConfig(16 * 1024, {{2, tt::DataFormat::RawUInt16}}).set_page_size(2, 16 * 1024));
        // CB 3, 4: Gather Scratchpads Ping-Pong (2 KB each)
        CreateCircularBuffer(program, core_range, CircularBufferConfig(2 * 1024, {{3, tt::DataFormat::Float16_b}}).set_page_size(3, 2 * 1024));
        CreateCircularBuffer(program, core_range, CircularBufferConfig(2 * 1024, {{4, tt::DataFormat::Float16_b}}).set_page_size(4, 2 * 1024));
        // CB 5: Background noise buffer (20 KB)
        CreateCircularBuffer(program, core_range, CircularBufferConfig(20 * 1024, {{5, tt::DataFormat::Float16_b}}).set_page_size(5, 20 * 1024));
        // CB 7, 8: Scatter Scratchpads Ping-Pong (2 KB each)
        CreateCircularBuffer(program, core_range, CircularBufferConfig(2 * 1024, {{7, tt::DataFormat::Float16_b}}).set_page_size(7, 2 * 1024));
        CreateCircularBuffer(program, core_range, CircularBufferConfig(2 * 1024, {{8, tt::DataFormat::Float16_b}}).set_page_size(8, 2 * 1024));
        // CB 16: Telemetry Queue (2 KB)
        CreateCircularBuffer(program, core_range, CircularBufferConfig(2 * 1024, {{16, tt::DataFormat::RawUInt32}}).set_page_size(16, 2 * 1024));

        std::cout << "[CONFIG] Circular buffers created." << std::endl;

        auto reader = CreateKernel(
            program, "engines/tile_matrix_4cw/reader_brisc.cpp", core_range,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
        );
        std::cout << "[CONFIG] BRISC kernel created." << std::endl;

        auto writer = CreateKernel(
            program, "engines/tile_matrix_4cw/writer_ncrisc.cpp", core_range,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default}
        );
        std::cout << "[CONFIG] NCRISC kernel created." << std::endl;

        auto compute = CreateKernel(
            program, "engines/tile_matrix_4cw/compute_trisc.cpp", core_range,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = true
            }
        );
        std::cout << "[CONFIG] TRISC compute kernel created." << std::endl;

        std::cout << "[SUCCESS] All kernels and buffers instantiated without error!" << std::endl;
        mesh_device->close();
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
