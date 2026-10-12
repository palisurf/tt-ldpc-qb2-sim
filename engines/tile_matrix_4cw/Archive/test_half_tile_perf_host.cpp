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
    std::cout << " TENSTORRENT BLACKHOLE TENSIX: 1/2 TILE GATHER & SCATTER BENCHMARK" << std::endl;
    std::cout << " Workload: 60 Edge Pairs (120 Edges = 1/2 Tile) over 1,000 Iters" << std::endl;
    std::cout << " Target: Physical Blackhole Core (0, 0)" << std::endl;
    std::cout << "==================================================================" << std::endl;

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = mesh_device->mesh_command_queue();
    Program program = CreateProgram();
    CoreCoord core(0, 0);
    CoreRange core_range(core, core);

    constexpr uint32_t vnode_bytes    = 20 * 1024; // 20 KB
    constexpr uint32_t h_bytes        = 16 * 1024; // 16 KB
    constexpr uint32_t scratch_bytes  = 4 * 1024;  // 4 KB
    constexpr uint32_t stats_bytes    = 2 * 1024;  // 2 KB

    CreateCircularBuffer(program, core_range,
        CircularBufferConfig(vnode_bytes, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}}).set_page_size(tt::CBIndex::c_0, vnode_bytes));
    CreateCircularBuffer(program, core_range,
        CircularBufferConfig(h_bytes, {{tt::CBIndex::c_2, tt::DataFormat::RawUInt16}}).set_page_size(tt::CBIndex::c_2, h_bytes));
    CreateCircularBuffer(program, core_range,
        CircularBufferConfig(scratch_bytes, {{tt::CBIndex::c_3, tt::DataFormat::Float16_b}}).set_page_size(tt::CBIndex::c_3, scratch_bytes));
    CreateCircularBuffer(program, core_range,
        CircularBufferConfig(scratch_bytes, {{tt::CBIndex::c_16, tt::DataFormat::Float16_b}}).set_page_size(tt::CBIndex::c_16, scratch_bytes));
    CreateCircularBuffer(program, core_range,
        CircularBufferConfig(stats_bytes, {{tt::CBIndex::c_17, tt::DataFormat::RawUInt32}}).set_page_size(tt::CBIndex::c_17, stats_bytes));

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
    auto gather_stats_dram = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = stats_bytes}, dram_config, mesh_device.get()
    );
    auto scatter_stats_dram = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = stats_bytes}, dram_config, mesh_device.get()
    );

    // Generate realistic h_offsets (multiples of 8 in range [0, 2559 * 8])
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

    // Initial variable node test data
    std::vector<uint16_t> v_data(vnode_bytes / sizeof(uint16_t), 0x3C00); // 1.0f in bf16
    distributed::EnqueueWriteMeshBuffer(cq, vnode_dram, v_data, false);

    // Zero out stats buffers
    std::vector<uint32_t> zero_stats(stats_bytes / sizeof(uint32_t), 0);
    distributed::EnqueueWriteMeshBuffer(cq, gather_stats_dram, zero_stats, false);
    distributed::EnqueueWriteMeshBuffer(cq, scatter_stats_dram, zero_stats, false);

    // Create Kernels
    auto gather_kernel = CreateKernel(
        program, "test_gather_bench_kernel.cpp", core_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );
    auto scatter_kernel = CreateKernel(
        program, "test_scatter_bench_kernel.cpp", core_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default}
    );
    auto compute_kernel = CreateKernel(
        program, "test_empty_compute.cpp", core_range,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = false}
    );

    std::vector<uint32_t> g_args = {
        static_cast<uint32_t>(h_dram->address()), 0,
        static_cast<uint32_t>(vnode_dram->address()), 0,
        static_cast<uint32_t>(gather_stats_dram->address()), 0
    };
    SetRuntimeArgs(program, gather_kernel, core, g_args);

    std::vector<uint32_t> s_args = {
        static_cast<uint32_t>(scatter_stats_dram->address()), 0
    };
    SetRuntimeArgs(program, scatter_kernel, core, s_args);

    std::cout << "[INFO] Dispatching 1/2 Tile Benchmark Kernels to Blackhole Silicon..." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    cq.finish();

    auto t1 = std::chrono::high_resolution_clock::now();
    double host_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[INFO] Benchmark completed on device in " << host_ms << " ms" << std::endl;

    // Read back results
    std::vector<uint32_t> gather_res(stats_bytes / sizeof(uint32_t), 0);
    std::vector<uint32_t> scatter_res(stats_bytes / sizeof(uint32_t), 0);
    distributed::ReadShard(cq, gather_res, gather_stats_dram, distributed::MeshCoordinate(0, 0), true);
    distributed::ReadShard(cq, scatter_res, scatter_stats_dram, distributed::MeshCoordinate(0, 0), true);

    std::cout << "\n==================================================================" << std::endl;
    std::cout << " GATHER BENCHMARK RESULTS (60 Edge Pairs = 1/2 Tile = 120 Edges)" << std::endl;
    std::cout << "==================================================================" << std::endl;
    std::cout << std::left << std::setw(38) << "Implementation Variant" 
              << std::right << std::setw(14) << "1/2 Tile (cyc)" 
              << std::setw(14) << "Full Tile (cyc)" 
              << std::setw(12) << "cyc/pair" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;

    auto print_row = [](const std::string& name, uint32_t half_cyc) {
        uint32_t full_cyc = half_cyc * 2;
        double cyc_pair = static_cast<double>(half_cyc) / 60.0;
        std::cout << std::left << std::setw(38) << name 
                  << std::right << std::setw(14) << half_cyc 
                  << std::setw(14) << full_cyc 
                  << std::setw(12) << std::fixed << std::setprecision(1) << cyc_pair << std::endl;
    };

    print_row("Variant 0: Baseline Scalar (8 x 'sh')", gather_res[0]);
    print_row("Variant 1: 32-bit 'sw' (Shift/OR)", gather_res[1]);
    print_row("Variant 2: Hardware 'pack' + 'sw'", gather_res[2]);
    print_row("Variant 3: 2-Pair Unrolled + 'pack'", gather_res[3]);
    print_row("Variant 4: Direct Ptrs + 2-Pair Unroll", gather_res[4]);
    print_row("Variant 5: Direct Ptrs + 4-Pair Unroll", gather_res[5]);
    print_row("Variant 6: Direct Ptrs + 8-Pair Unroll", gather_res[6]);

    std::cout << "\n==================================================================" << std::endl;
    std::cout << " SCATTER BENCHMARK RESULTS (60 Edge Pairs = 1/2 Tile = 120 Edges)" << std::endl;
    std::cout << "==================================================================" << std::endl;
    std::cout << std::left << std::setw(38) << "Implementation Variant" 
              << std::right << std::setw(14) << "1/2 Tile (cyc)" 
              << std::setw(14) << "Full Tile (cyc)" 
              << std::setw(12) << "cyc/pair" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;

    print_row("Variant 0: Legacy 32-bit Shift/Mask", scatter_res[0]);
    print_row("Variant 1: Direct 16-bit 'sh'", scatter_res[1]);
    print_row("Variant 2: Hardware 'pack' + 'sw'", scatter_res[2]);
    print_row("Variant 3: 2-Pair Unrolled + 'pack'", scatter_res[3]);
    print_row("Variant 4: Direct Ptrs + 2-Pair 'sh'", scatter_res[4]);
    print_row("Variant 5: Direct Ptrs + 4-Pair 'sh'", scatter_res[5]);
    print_row("Variant 6: Direct Ptrs + 4-Pair Word Pack", scatter_res[6]);
    print_row("Variant 7: Direct Ptrs + 8-Pair 'sh'", scatter_res[7]);

    uint32_t min_gather = gather_res[0];
    for (int i = 1; i < 7; ++i) if (gather_res[i] < min_gather) min_gather = gather_res[i];
    uint32_t min_scatter = scatter_res[0];
    for (int i = 1; i < 8; ++i) if (scatter_res[i] < min_scatter) min_scatter = scatter_res[i];

    std::cout << "\n==================================================================" << std::endl;
    std::cout << " ARCHITECTURAL COMPARISON WITH SFPU MATH ENGINE (1,325 CYCLES)" << std::endl;
    std::cout << "==================================================================" << std::endl;
    std::cout << " SFPU Compute Clock Cycles per Tile:  1,325 cycles" << std::endl;
    std::cout << " Best 1/2 Tile Gather Latency:        " << min_gather << " cycles (" 
              << (min_gather <= 1325 ? "HIDDEN BEHIND MATH" : "Partially Exposed") << ")" << std::endl;
    std::cout << " Best 1/2 Tile Scatter Latency:       " << min_scatter << " cycles (" 
              << (min_scatter <= 1325 ? "HIDDEN BEHIND MATH" : "Partially Exposed") << ")" << std::endl;
    std::cout << "==================================================================\n" << std::endl;

    mesh_device->close();
    return 0;
}
