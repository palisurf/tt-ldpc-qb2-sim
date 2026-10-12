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

    constexpr uint32_t tile_bytes = 2048;

    CreateCircularBuffer(program, core_range,
        CircularBufferConfig(tile_bytes, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}}).set_page_size(tt::CBIndex::c_0, tile_bytes));
    CreateCircularBuffer(program, core_range,
        CircularBufferConfig(tile_bytes, {{tt::CBIndex::c_1, tt::DataFormat::Float16_b}}).set_page_size(tt::CBIndex::c_1, tile_bytes));
    CreateCircularBuffer(program, core_range,
        CircularBufferConfig(tile_bytes, {{tt::CBIndex::c_16, tt::DataFormat::Float16_b}}).set_page_size(tt::CBIndex::c_16, tile_bytes));
    CreateCircularBuffer(program, core_range,
        CircularBufferConfig(tile_bytes, {{tt::CBIndex::c_17, tt::DataFormat::Float16_b}}).set_page_size(tt::CBIndex::c_17, tile_bytes));

    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = 2048,
        .buffer_type = BufferType::DRAM
    };
    auto out0_dram = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = 2048}, dram_config, mesh_device.get()
    );
    auto out1_dram = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = 2048}, dram_config, mesh_device.get()
    );

    auto reader = CreateKernel(
        program, "test_two_tiles_reader.cpp", core_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );
    auto writer = CreateKernel(
        program, "test_two_cbs_writer.cpp", core_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default}
    );
    auto compute = CreateKernel(
        program, "test_two_cbs_kernel.cpp", core_range,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = false}
    );

    std::vector<uint32_t> w_args = {
        static_cast<uint32_t>(out0_dram->address()), 0,
        static_cast<uint32_t>(out1_dram->address()), 0
    };
    SetRuntimeArgs(program, writer, core, w_args);

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    cq.finish();

    std::vector<uint16_t> out0(1024, 0);
    std::vector<uint16_t> out1(1024, 0);
    distributed::ReadShard(cq, out0, out0_dram, distributed::MeshCoordinate(0, 0), true);
    distributed::ReadShard(cq, out1, out1_dram, distributed::MeshCoordinate(0, 0), true);

    float val0 = bfloat16_to_float(out0[0]);
    float val1 = bfloat16_to_float(out1[0]);
    std::cout << "[TEST] Silicon Readback Out0: " << val0 << " (expected 11.0)" << std::endl;
    std::cout << "[TEST] Silicon Readback Out1: " << val1 << " (expected 22.0)" << std::endl;

    mesh_device->close();
    return 0;
}
