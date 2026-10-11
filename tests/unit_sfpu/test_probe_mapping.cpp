#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <iomanip>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/circular_buffer_config.hpp>

using namespace tt;
using namespace tt::tt_metal;

int main() {
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = mesh_device->mesh_command_queue();
    CoreCoord core = {0, 0};
    CoreRange core_range(core, core);
    Program program = CreateProgram();

    uint32_t tile_bytes = sizeof(bfloat16) * 1024;
    uint32_t stats_bytes = 64;

    CreateCircularBuffer(program, core_range,
        CircularBufferConfig(tile_bytes, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}}).set_page_size(tt::CBIndex::c_0, tile_bytes));
    CreateCircularBuffer(program, core_range,
        CircularBufferConfig(tile_bytes, {{tt::CBIndex::c_16, tt::DataFormat::Float16_b}}).set_page_size(tt::CBIndex::c_16, tile_bytes));
    CreateCircularBuffer(program, core_range,
        CircularBufferConfig(2048, {{tt::CBIndex::c_17, tt::DataFormat::RawUInt32}}).set_page_size(tt::CBIndex::c_17, 2048));

    distributed::DeviceLocalBufferConfig tile_dram_cfg{ .page_size = tile_bytes, .buffer_type = BufferType::DRAM };
    auto in_dram = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = tile_bytes}, tile_dram_cfg, mesh_device.get());
    auto out_dram = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = tile_bytes}, tile_dram_cfg, mesh_device.get());

    distributed::DeviceLocalBufferConfig stats_dram_cfg{ .page_size = stats_bytes, .buffer_type = BufferType::DRAM };
    auto stats_dram = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = stats_bytes}, stats_dram_cfg, mesh_device.get());

    std::vector<bfloat16> in_tile(1024, bfloat16(0.0f));
    distributed::EnqueueWriteMeshBuffer(cq, in_dram, in_tile, false);

    auto reader = CreateKernel(program, "tests/unit_sfpu/reader_unit_tile.cpp", core_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    auto writer = CreateKernel(program, "tests/unit_sfpu/writer_unit_tile.cpp", core_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    auto compute = CreateKernel(program, "tests/unit_sfpu/kernel_probe.cpp", core_range,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true});

    std::vector<uint32_t> r_args = { static_cast<uint32_t>(in_dram->address()), 0, tile_bytes };
    SetRuntimeArgs(program, reader, core, r_args);
    std::vector<uint32_t> w_args = { static_cast<uint32_t>(out_dram->address()), static_cast<uint32_t>(stats_dram->address()), 0, tile_bytes, stats_bytes };
    SetRuntimeArgs(program, writer, core, w_args);

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));

    distributed::EnqueueMeshWorkload(cq, workload, false);
    cq.finish();

    std::vector<bfloat16> hw_out_tile(1024);
    distributed::ReadShard(cq, hw_out_tile, out_dram, distributed::MeshCoordinate(0, 0), true);

    std::cout << "--- TILE ELEMENT TO REGISTER MAPPING ---" << std::endl;
    for (int f = 0; f < 4; ++f) {
        std::cout << "\n=== Face " << f << " (All 16 rows) ===" << std::endl;
        for (int r = 0; r < 16; ++r) {
            std::cout << "  Row " << std::setw(2) << r << ": ";
            for (int c = 0; c < 16; ++c) {
                int idx = f * 256 + r * 16 + c;
                int reg_val = static_cast<int>(static_cast<float>(hw_out_tile[idx]));
                std::cout << std::setw(3) << reg_val << " ";
            }
            std::cout << std::endl;
        }
    }

    return 0;
}
