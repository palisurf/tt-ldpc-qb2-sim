#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <cstring>
#include <random>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/circular_buffer_config.hpp>

using namespace tt;
using namespace tt::tt_metal;

struct ParityCheckMatrix {
    uint32_t M = 0;
    uint32_t N = 0;
    uint32_t K = 0;
    std::vector<std::vector<uint32_t>> check_to_var;
};

bool load_ar4ja(const std::string& path, ParityCheckMatrix& pcm) {
    std::ifstream infile(path);
    if (!infile.is_open()) return false;

    struct Edge { uint32_t r, c; float val; };
    std::vector<Edge> edges;
    uint32_t r, c; float v;
    while (infile >> r >> c >> v) edges.push_back({r, c, v});
    if (edges.empty()) return false;

    pcm.M = edges.back().r;
    pcm.N = edges.back().c;
    pcm.K = pcm.N - pcm.M;
    pcm.check_to_var.resize(pcm.M);

    for (size_t i = 0; i + 1 < edges.size(); ++i) {
        if (std::abs(edges[i].val) > 1e-5f) {
            pcm.check_to_var[edges[i].r - 1].push_back(edges[i].c - 1);
        }
    }
    return true;
}

int main() {
    std::cout << "\n==================================================================" << std::endl;
    std::cout << " TENSTORRENT BLACKHOLE: DUAL SCATTER BANK-SCHEDULE BENCHMARK" << std::endl;
    std::cout << " Evaluating 8-Pair Unrolled Halfwords & Bank-Aware Scheduling" << std::endl;
    std::cout << " Target: Physical Blackhole Tensix Core (0, 0) | 1,000 Iters/Sched" << std::endl;
    std::cout << "==================================================================" << std::endl;

    // Load AR4JA matrix
    ParityCheckMatrix pcm;
    std::string pcm_path = "../../matrices/AR4JA_r45_4c_128c_r12.chinn.out";
    if (!load_ar4ja(pcm_path, pcm)) {
        pcm_path = "matrices/AR4JA_r45_4c_128c_r12.chinn.out";
        if (!load_ar4ja(pcm_path, pcm)) {
            std::cerr << "[ERROR] Could not load AR4JA matrix from " << pcm_path << std::endl;
            return 1;
        }
    }
    std::cout << "[INFO] Loaded CCSDS AR4JA Matrix: M=" << pcm.M << ", N=" << pcm.N << std::endl;

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = mesh_device->mesh_command_queue();
    Program program = CreateProgram();
    CoreCoord core(0, 0);
    CoreRange core_range(core, core);

    constexpr uint32_t vnode_bytes    = 20 * 1024; // 20 KB
    constexpr uint32_t h_bytes        = 16 * 1024; // 16 KB
    constexpr uint32_t scratch_bytes  = 4 * 1024;  // 4 KB (Scatter Input)
    constexpr uint32_t stats_bytes    = 2 * 1024;  // 2 KB
    constexpr uint32_t telem_bytes    = 2 * 1024;  // 2 KB

    CreateCircularBuffer(program, core_range,
        CircularBufferConfig(vnode_bytes, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}}).set_page_size(tt::CBIndex::c_0, vnode_bytes));
    CreateCircularBuffer(program, core_range,
        CircularBufferConfig(h_bytes, {{tt::CBIndex::c_2, tt::DataFormat::RawUInt16}}).set_page_size(tt::CBIndex::c_2, h_bytes));
    CreateCircularBuffer(program, core_range,
        CircularBufferConfig(scratch_bytes, {{tt::CBIndex::c_16, tt::DataFormat::Float16_b}}).set_page_size(tt::CBIndex::c_16, scratch_bytes));
    CreateCircularBuffer(program, core_range,
        CircularBufferConfig(stats_bytes, {{tt::CBIndex::c_17, tt::DataFormat::RawUInt32}}).set_page_size(tt::CBIndex::c_17, stats_bytes));
    CreateCircularBuffer(program, core_range,
        CircularBufferConfig(telem_bytes, {{tt::CBIndex::c_18, tt::DataFormat::RawUInt32}}).set_page_size(tt::CBIndex::c_18, telem_bytes));

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

    std::vector<uint32_t> h_data(h_bytes / sizeof(uint32_t), 0);
    constexpr uint32_t NUM_SCHEDULES = 5;
    constexpr uint32_t PAIRS_PER_WORKER = 60;
    constexpr uint32_t PAIRS_PER_TILE   = 120;

    auto make_pair = [](uint32_t ve, uint32_t vo) -> uint32_t {
        uint32_t oe = (ve * 8) & 0xFFFF;
        uint32_t oo = (vo * 8) & 0xFFFF;
        return oe | (oo << 16);
    };

    // --------------------------------------------------------------------------
    // Sched 0: True Single-Bank Bottleneck (All stores target Bank 0)
    // --------------------------------------------------------------------------
    {
        uint32_t* base = &h_data[0 * PAIRS_PER_TILE];
        for (uint32_t p = 0; p < PAIRS_PER_WORKER; ++p) {
            // (v >> 1) % 16 == 0 -> ALL accesses map strictly to physical Bank 0
            uint32_t ve = (32 * (p % 80)) % 2560;
            uint32_t vo = (32 * (p % 80) + 1) % 2560;
            uint32_t pair_val = make_pair(ve, vo);
            base[p]                    = pair_val; // W0 hits Bank 0
            base[PAIRS_PER_WORKER + p] = pair_val; // W1 hits Bank 0 simultaneously
        }
    }

    // --------------------------------------------------------------------------
    // Sched 1: Random / Unscheduled Variable Nodes (Current Baseline)
    // --------------------------------------------------------------------------
    {
        uint32_t* base = &h_data[1 * PAIRS_PER_TILE];
        std::mt19937 rng(42);
        std::uniform_int_distribution<uint32_t> dist_v(0, 2559);
        for (uint32_t p = 0; p < PAIRS_PER_TILE; ++p) {
            base[p] = make_pair(dist_v(rng), dist_v(rng));
        }
    }

    // --------------------------------------------------------------------------
    // Sched 2: Synthetic Disjoint Even/Odd Banks (Zero-Collision Orthogonal Phase)
    // --------------------------------------------------------------------------
    {
        uint32_t* base = &h_data[2 * PAIRS_PER_TILE];
        // W0 uses Even banks (v >> 1 is even): v = 4*k, 4*k+1
        // W1 uses Odd banks  (v >> 1 is odd):  v = 4*k+2, 4*k+3
        for (uint32_t p = 0; p < PAIRS_PER_WORKER; ++p) {
            uint32_t k = p % 8; // bank 2*k
            uint32_t rep = p / 8;
            uint32_t w0_ve = (4 * k + 32 * rep) % 2560;
            uint32_t w0_vo = (w0_ve + 1) % 2560;
            base[p] = make_pair(w0_ve, w0_vo);

            uint32_t w1_ve = (4 * k + 2 + 32 * rep) % 2560;
            uint32_t w1_vo = (w1_ve + 1) % 2560;
            base[PAIRS_PER_WORKER + p] = make_pair(w1_ve, w1_vo);
        }
    }

    // Extract Tile 0 edges from AR4JA matrix (Checks 0..39, each degree 6 = 3 pairs)
    struct EdgePair { uint32_t ve, vo; };
    std::vector<EdgePair> ar4ja_w0_pairs;
    std::vector<EdgePair> ar4ja_w1_pairs;

    for (uint32_t c = 0; c < 20; ++c) {
        const auto& vars = pcm.check_to_var[c];
        ar4ja_w0_pairs.push_back({vars[0], vars[1]});
        ar4ja_w0_pairs.push_back({vars[2], vars[3]});
        ar4ja_w0_pairs.push_back({vars[4], vars[5]});
    }
    for (uint32_t c = 20; c < 40; ++c) {
        const auto& vars = pcm.check_to_var[c];
        ar4ja_w1_pairs.push_back({vars[0], vars[1]});
        ar4ja_w1_pairs.push_back({vars[2], vars[3]});
        ar4ja_w1_pairs.push_back({vars[4], vars[5]});
    }

    // --------------------------------------------------------------------------
    // Sched 3: Real AR4JA Tile 0 Natural Order
    // --------------------------------------------------------------------------
    {
        uint32_t* base = &h_data[3 * PAIRS_PER_TILE];
        for (uint32_t p = 0; p < PAIRS_PER_WORKER; ++p) {
            base[p]                    = make_pair(ar4ja_w0_pairs[p].ve, ar4ja_w0_pairs[p].vo);
            base[PAIRS_PER_WORKER + p] = make_pair(ar4ja_w1_pairs[p].ve, ar4ja_w1_pairs[p].vo);
        }
    }

    // --------------------------------------------------------------------------
    // Sched 4: Real AR4JA Tile 0 Pre-Computed Bank-Aware Chunk Scheduling
    // --------------------------------------------------------------------------
    {
        uint32_t* base = &h_data[4 * PAIRS_PER_TILE];
        // W0 remains in natural order
        for (uint32_t p = 0; p < PAIRS_PER_WORKER; ++p) {
            base[p] = make_pair(ar4ja_w0_pairs[p].ve, ar4ja_w0_pairs[p].vo);
        }

        // Permute W1 pairs to minimize bank collisions with W0 per 8-pair chunk
        std::vector<bool> used_w1(PAIRS_PER_WORKER, false);
        std::vector<EdgePair> scheduled_w1;
        scheduled_w1.reserve(PAIRS_PER_WORKER);

        auto get_bank = [](uint32_t v) -> uint32_t {
            return (v >> 1) & 0xF;
        };

        // 7 chunks of 8 pairs
        for (uint32_t chunk = 0; chunk < 7; ++chunk) {
            // Compute bank occupancy of W0 in this chunk
            uint32_t bank_hist_w0[16] = {0};
            for (uint32_t k = 0; k < 8; ++k) {
                uint32_t p = chunk * 8 + k;
                bank_hist_w0[get_bank(ar4ja_w0_pairs[p].ve)]++;
                bank_hist_w0[get_bank(ar4ja_w0_pairs[p].vo)]++;
            }

            // Score remaining W1 pairs by overlap with W0 banks
            struct Cand { uint32_t idx; uint32_t score; };
            std::vector<Cand> candidates;
            for (uint32_t i = 0; i < PAIRS_PER_WORKER; ++i) {
                if (!used_w1[i]) {
                    uint32_t score = bank_hist_w0[get_bank(ar4ja_w1_pairs[i].ve)] +
                                     bank_hist_w0[get_bank(ar4ja_w1_pairs[i].vo)];
                    candidates.push_back({i, score});
                }
            }
            std::sort(candidates.begin(), candidates.end(), [](const Cand& a, const Cand& b) {
                return a.score < b.score;
            });

            // Pick top 8 candidates with lowest collision score
            for (uint32_t k = 0; k < 8; ++k) {
                uint32_t sel_idx = candidates[k].idx;
                used_w1[sel_idx] = true;
                scheduled_w1.push_back(ar4ja_w1_pairs[sel_idx]);
            }
        }
        // Tail 4 pairs
        for (uint32_t i = 0; i < PAIRS_PER_WORKER; ++i) {
            if (!used_w1[i]) {
                scheduled_w1.push_back(ar4ja_w1_pairs[i]);
            }
        }

        for (uint32_t p = 0; p < PAIRS_PER_WORKER; ++p) {
            base[PAIRS_PER_WORKER + p] = make_pair(scheduled_w1[p].ve, scheduled_w1[p].vo);
        }
    }

    // Write h_data and init buffers
    distributed::EnqueueWriteMeshBuffer(cq, h_dram, h_data, false);
    std::vector<uint16_t> v_data(vnode_bytes / sizeof(uint16_t), 0x3C00); // 1.0f in bf16
    distributed::EnqueueWriteMeshBuffer(cq, vnode_dram, v_data, false);
    std::vector<uint32_t> zero_stats(stats_bytes / sizeof(uint32_t), 0);
    distributed::EnqueueWriteMeshBuffer(cq, stats_dram, zero_stats, false);

    auto worker0 = CreateKernel(
        program, "test_dual_scatter_worker0.cpp", core_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );
    auto worker1 = CreateKernel(
        program, "test_dual_scatter_worker1.cpp", core_range,
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

    std::cout << "[INFO] Dispatching 5-Schedule Benchmark to Blackhole Silicon..." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    cq.finish();

    auto t1 = std::chrono::high_resolution_clock::now();
    double host_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[INFO] All 5 Schedules completed on device in " << host_ms << " ms\n" << std::endl;

    std::vector<uint32_t> stats_res(stats_bytes / sizeof(uint32_t), 0);
    distributed::ReadShard(cq, stats_res, stats_dram, distributed::MeshCoordinate(0, 0), true);

    const char* sched_names[NUM_SCHEDULES] = {
        "Sched 0: 100% Single-Bank Collisions (All in Bank 0)",
        "Sched 1: Random Unscheduled (Uniform Baseline)",
        "Sched 2: Synthetic Conflict-Free (Disjoint Banks)",
        "Sched 3: Real AR4JA Tile 0 (Natural Check Order)",
        "Sched 4: Real AR4JA Tile 0 (Bank-Aware Chunk Sched)"
    };

    std::cout << "==========================================================================================" << std::endl;
    std::cout << " TENSTORRENT BLACKHOLE: DUAL-WORKER SCATTER SILICON MEASUREMENTS (1,000 Iters)" << std::endl;
    std::cout << "==========================================================================================" << std::endl;
    std::cout << std::left << std::setw(52) << "Scatter Schedule Configuration"
              << std::right << std::setw(10) << "W0 (cyc)"
              << std::setw(10) << "W1 (cyc)"
              << std::setw(12) << "Tile (cyc)"
              << std::setw(11) << "cyc/pair"
              << std::setw(11) << "Speedup" << std::endl;
    std::cout << "------------------------------------------------------------------------------------------" << std::endl;

    for (uint32_t s = 0; s < NUM_SCHEDULES; ++s) {
        uint32_t w0 = stats_res[s];
        uint32_t w1 = stats_res[8 + s];
        uint32_t tile = std::max(w0, w1);
        double cyc_pair = static_cast<double>(tile) / 120.0;
        double speedup = 6402.0 / tile;

        std::cout << std::left << std::setw(52) << sched_names[s]
                  << std::right << std::setw(10) << w0
                  << std::setw(10) << w1
                  << std::setw(12) << tile
                  << std::setw(11) << std::fixed << std::setprecision(1) << cyc_pair
                  << std::setw(10) << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    }

    uint32_t syn_free = std::max(stats_res[2], stats_res[8 + 2]);
    uint32_t ar4ja_nat = std::max(stats_res[3], stats_res[8 + 3]);
    uint32_t ar4ja_opt = std::max(stats_res[4], stats_res[8 + 4]);

    std::cout << "\n==========================================================================================" << std::endl;
    std::cout << " ARCHITECTURAL CONTROLLABILITY & ANALYSIS" << std::endl;
    std::cout << "==========================================================================================" << std::endl;
    std::cout << " SFPU Compute Clock Cycles Budget:           1,325 cycles" << std::endl;
    std::cout << " Single-Worker 8-Pair Unrolled Halfwords:    1,737 cycles (Half Tile)" << std::endl;
    std::cout << " Dual-Worker Synthetic Conflict-Free Tile:   " << syn_free << " cycles" << std::endl;
    std::cout << " Real AR4JA Natural Order Tile Latency:      " << ar4ja_nat << " cycles" << std::endl;
    std::cout << " Real AR4JA Bank-Aware Scheduled Latency:    " << ar4ja_opt << " cycles" << std::endl;
    std::cout << " Schedule Optimization Latency Reduction:    " << (int32_t)ar4ja_nat - (int32_t)ar4ja_opt << " cycles" << std::endl;
    std::cout << "==========================================================================================\n" << std::endl;

    mesh_device->close();
    return 0;
}
