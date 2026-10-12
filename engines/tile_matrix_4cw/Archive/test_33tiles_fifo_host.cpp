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
    constexpr uint32_t num_tiles  = 33;
    constexpr uint32_t r_cb_bytes = 34 * tile_bytes; // 68 KB capacity

    // CB 0: Input (2 KB)
    CreateCircularBuffer(program, core_range,
        CircularBufferConfig(tile_bytes, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}}).set_page_size(tt::CBIndex::c_0, tile_bytes));
    // CB 16: Output (2 KB)
    CreateCircularBuffer(program, core_range,
        CircularBufferConfig(tile_bytes, {{tt::CBIndex::c_16, tt::DataFormat::Float16_b}}).set_page_size(tt::CBIndex::c_16, tile_bytes));
    // CB 17: R Message 34-Tile Circular FIFO (68 KB)
    CreateCircularBuffer(program, core_range,
        CircularBufferConfig(r_cb_bytes, {{tt::CBIndex::c_17, tt::DataFormat::Float16_b}}).set_page_size(tt::CBIndex::c_17, tile_bytes));

    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = 2048,
        .buffer_type = BufferType::DRAM
    };
    auto out_dram = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = 2048}, dram_config, mesh_device.get()
    );

    auto reader = CreateKernel(
        program, "test_r_fifo_reader.cpp", core_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );
    auto writer = CreateKernel(
        program, "test_r_fifo_writer.cpp", core_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default}
    );
    auto compute = CreateKernel(
        program, "test_33tiles_fifo_kernel.cpp", core_range,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = false}
    );

    std::vector<uint32_t> w_args = {
        static_cast<uint32_t>(out_dram->address()), 0
    };
    SetRuntimeArgs(program, writer, core, w_args);

    std::cout << "[INFO] Dispatching 33-Tile Circular FIFO Test to QuietBox 2 Silicon..." << std::endl;
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    cq.finish();

    std::vector<uint16_t> out(1024, 0);
    distributed::ReadShard(cq, out, out_dram, distributed::MeshCoordinate(0, 0), true);

    float val = bfloat16_to_float(out[0]);
    std::cout << "[TEST] 33-Tile FIFO Feedback Result: " << val 
              << " (expected 132.0: (32 + 1) * 4)" << std::endl;

    mesh_device->close();
    bool pass = (std::abs(val - 132.0f) < 0.1f);
    std::cout << (pass ? "[PASS] 33-Tile Circular FIFO verified successfully on Blackhole silicon!" 
                       : "[FAIL] Unexpected result") << std::endl;
    return pass ? 0 : 1;
}
