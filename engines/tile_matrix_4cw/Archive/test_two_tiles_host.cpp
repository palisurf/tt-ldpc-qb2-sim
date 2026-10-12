#include <iostream>
#include <vector>
#include <cmath>
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

static float bfloat16_to_float(uint16_t bf) {
    uint32_t u = static_cast<uint32_t>(bf) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(float));
    return f;
}

int main() {
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = mesh_device->mesh_command_queue();
    Program program = CreateProgram();
    CoreCoord core(0, 0);
    CoreRange core_range(core, core);

    constexpr uint32_t tile_size_bytes = 2048;

    // CB 0: Input 0 (2 KB)
    CreateCircularBuffer(program, core_range,
        CircularBufferConfig(tile_size_bytes, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}}).set_page_size(tt::CBIndex::c_0, tile_size_bytes));
    // CB 1: Input 1 (2 KB)
    CreateCircularBuffer(program, core_range,
        CircularBufferConfig(tile_size_bytes, {{tt::CBIndex::c_1, tt::DataFormat::Float16_b}}).set_page_size(tt::CBIndex::c_1, tile_size_bytes));
    // CB 16: Output (4 KB - 2 tiles)
    CreateCircularBuffer(program, core_range,
        CircularBufferConfig(2 * tile_size_bytes, {{tt::CBIndex::c_16, tt::DataFormat::Float16_b}}).set_page_size(tt::CBIndex::c_16, tile_size_bytes));

    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = 4096,
        .buffer_type = BufferType::DRAM
    };
    auto out_dram = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = 4096}, dram_config, mesh_device.get()
    );

    auto reader = CreateKernel(
        program, "test_two_tiles_reader.cpp", core_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );
    auto writer = CreateKernel(
        program, "test_two_tiles_writer.cpp", core_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default}
    );
    auto compute = CreateKernel(
        program, "test_two_tiles_kernel.cpp", core_range,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = false}
    );

    std::vector<uint32_t> w_args = {
        static_cast<uint32_t>(out_dram->address()), 0
    };
    SetRuntimeArgs(program, writer, core, w_args);

    std::cout << "[TEST] Dispatching 2-tile pack test to Blackhole Core (0, 0)..." << std::endl;
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    cq.finish();

    std::vector<uint16_t> host_out(2048, 0);
    distributed::ReadShard(cq, host_out, out_dram, distributed::MeshCoordinate(0, 0), true);

    float val0 = bfloat16_to_float(host_out[0]);
    float val1 = bfloat16_to_float(host_out[1024]);
    std::cout << "[TEST] Silicon Readback Tile 0: " << val0 << " (expected 10.0)" << std::endl;
    std::cout << "[TEST] Silicon Readback Tile 1: " << val1 << " (expected 30.0)" << std::endl;

    mesh_device->close();
    return 0;
}
