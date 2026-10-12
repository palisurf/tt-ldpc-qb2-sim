#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <cstring>
#include <random>
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
    std::cout << "\n==================================================================" << std::endl;
    std::cout << " TENSTORRENT BLACKHOLE: DUAL-WORKER CONCURRENT GATHER BENCHMARK" << std::endl;
    std::cout << " Worker 0 (BRISC):  Checks  0..19 (60 Edge Pairs) -> Bytes   0..959" << std::endl;
    std::cout << " Worker 1 (NCRISC): Checks 20..39 (60 Edge Pairs) -> Bytes 960..1919" << std::endl;
    std::cout << " Total Workload:    120 Edge Pairs = FULL DEGREE-6 TILE (1,000 Iters)" << std::endl;
    std::cout << " Target: Physical Blackhole Tensix Core (0, 0)" << std::endl;
    std::cout << "==================================================================" << std::endl;

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = mesh_device->mesh_command_queue();
    Program program = CreateProgram();
    CoreCoord core(0, 0);
    CoreRange core_range(core, core);

    constexpr uint32_t vnode_bytes    = 20 * 1024; // 20 KB
    constexpr uint32_t h_bytes        = 16 * 1024; // 16 KB
    constexpr uint32_t scratch_bytes  = 4 * 1024;  // 4 KB (Unified 32x32 Tile)
    constexpr uint32_t stats_bytes    = 2 * 1024;  // 2 KB
    constexpr uint32_t telem_bytes    = 2 * 1024;  // 2 KB

    CreateCircularBuffer(program, core_range,
        CircularBufferConfig(vnode_bytes, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}}).set_page_size(tt::CBIndex::c_0, vnode_bytes));
    CreateCircularBuffer(program, core_range,
        CircularBufferConfig(h_bytes, {{tt::CBIndex::c_2, tt::DataFormat::RawUInt16}}).set_page_size(tt::CBIndex::c_2, h_bytes));
    CreateCircularBuffer(program, core_range,
        CircularBufferConfig(scratch_bytes, {{tt::CBIndex::c_3, tt::DataFormat::Float16_b}}).set_page_size(tt::CBIndex::c_3, scratch_bytes));
    CreateCircularBuffer(program, core_range,
        CircularBufferConfig(stats_bytes, {{tt::CBIndex::c_16, tt::DataFormat::RawUInt32}}).set_page_size(tt::CBIndex::c_16, stats_bytes));
    CreateCircularBuffer(program, core_range,
        CircularBufferConfig(telem_bytes, {{tt::CBIndex::c_17, tt::DataFormat::RawUInt32}}).set_page_size(tt::CBIndex::c_17, telem_bytes));

    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = 2048,
        .buffer_type = BufferType::DRAM
    };
    auto h_dram = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = h_bytes}, dram_config, mesh_device.get()
    );
    auto vnode_dram = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = vnode_bytes}, dram_config, mesh_device.get()
    );
    auto stats_dram = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = stats_bytes}, dram_config, mesh_device.get()
    );

    // Generate realistic h_offsets for all 120 edge pairs (full degree-6 tile)
    std::vector<uint32_t> h_data(h_bytes / sizeof(uint32_t));
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist_v(0, 2559);
    for (size_t i = 0; i < h_data.size(); ++i) {
        uint32_t v_even = dist_v(rng);
        uint32_t v_odd  = dist_v(rng);
        uint32_t off_even = (v_even * 8) & 0xFFFF;
        uint32_t off_odd  = (v_odd * 8) & 0xFFFF;
        h_data[i] = off_even | (off_odd << 16);
    }
    distributed::EnqueueWriteMeshBuffer(cq, h_dram, h_data, false);

    std::vector<uint16_t> v_data(vnode_bytes / sizeof(uint16_t), 0x3C00); // 1.0f in bf16
    distributed::EnqueueWriteMeshBuffer(cq, vnode_dram, v_data, false);

    std::vector<uint32_t> zero_stats(stats_bytes / sizeof(uint32_t), 0);
    distributed::EnqueueWriteMeshBuffer(cq, stats_dram, zero_stats, false);

    // Create Dual Data Movement Kernels:
    auto worker0 = CreateKernel(
        program, "test_dual_gather_worker0.cpp", core_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );
    auto worker1 = CreateKernel(
        program, "test_dual_gather_worker1.cpp", core_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default}
    );
    auto compute_kernel = CreateKernel(
        program, "test_empty_compute.cpp", core_range,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = false}
    );

    std::vector<uint32_t> w0_args = {
        static_cast<uint32_t>(h_dram->address()), 0,
        static_cast<uint32_t>(vnode_dram->address()), 0,
        static_cast<uint32_t>(stats_dram->address()), 0
    };
    SetRuntimeArgs(program, worker0, core, w0_args);

    std::vector<uint32_t> w1_args = {
        static_cast<uint32_t>(stats_dram->address()), 0
    };
    SetRuntimeArgs(program, worker1, core, w1_args);

    std::cout << "[INFO] Launching Dual-Worker Concurrent Gather on Blackhole Silicon..." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    cq.finish();

    auto t1 = std::chrono::high_resolution_clock::now();
    double host_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[INFO] Concurrent run completed in " << host_ms << " ms" << std::endl;

    std::vector<uint32_t> stats_res(stats_bytes / sizeof(uint32_t), 0);
    distributed::ReadShard(cq, stats_res, stats_dram, distributed::MeshCoordinate(0, 0), true);

    uint32_t cyc_worker0 = stats_res[0];
    uint32_t cyc_worker1 = stats_res[1];
    uint32_t concurrent_tile_cyc = std::max(cyc_worker0, cyc_worker1);

    std::cout << "\n==================================================================" << std::endl;
    std::cout << " DUAL-WORKER CONCURRENT GATHER SILICON MEASUREMENTS" << std::endl;
    std::cout << "==================================================================" << std::endl;
    std::cout << " Worker 0 (BRISC)  Latency (60 Edge Pairs): " << cyc_worker0 << " cycles" << std::endl;
    std::cout << " Worker 1 (NCRISC) Latency (60 Edge Pairs): " << cyc_worker1 << " cycles" << std::endl;
    std::cout << " Concurrent FULL TILE (120 Edge Pairs):     " << concurrent_tile_cyc << " cycles" << std::endl;
    std::cout << " Baseline Single-Core Full Tile:            6,402 cycles" << std::endl;
    std::cout << " Measured Dual-Worker Speedup:              " 
              << std::fixed << std::setprecision(2) << (6402.0 / concurrent_tile_cyc) << "x" << std::endl;
    std::cout << " Latency per Edge Pair (Overall):           " 
              << std::fixed << std::setprecision(1) << (static_cast<double>(concurrent_tile_cyc) / 120.0) << " cyc/pair" << std::endl;

    std::cout << "\n==================================================================" << std::endl;
    std::cout << " COMPARISON WITH SFPU MATH ENGINE (1,325 CYCLES)" << std::endl;
    std::cout << "==================================================================" << std::endl;
    std::cout << " SFPU Vector Approximate-Min* Math Budget: 1,325 cycles" << std::endl;
    std::cout << " Full-Tile Concurrent Gather Latency:       " << concurrent_tile_cyc << " cycles" << std::endl;
    std::cout << " Gap to Zero-Overhead Full Hiding:          " << (int32_t)concurrent_tile_cyc - 1325 << " cycles" << std::endl;
    std::cout << "==================================================================\n" << std::endl;

    mesh_device->close();
    return 0;
}
