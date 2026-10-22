// ==============================================================================
// engines/circulant_dma_4cw/ldpc_sim_circulant_dma.cpp
//
// Tenstorrent QuietBox 2: 4-Codeword Circulant DMA LDPC Simulator & Harness
// Architecture: Line-Rate 512-bit DMA Bursts with Zero-Transpose SFPU Math
// Target: 2x2 Blackhole Mesh Device (4 Chips x 110 Tensix Cores = 440 Cores)
// Target: Single Tensix Core (0, 0) / ttsim Software Simulator Gatekeeper
// ==============================================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <unistd.h>
#include <thread>
#include <atomic>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/circular_buffer_config.hpp>

#include "ldpc_circulant_dma_core.h"

using namespace tt;
using namespace tt::tt_metal;
namespace fs = std::filesystem;

static std::string resolve_filepath(const std::string& filename) {
    if (fs::exists(filename)) return filename;
    if (fs::exists("engines/circulant_dma_4cw/" + filename)) return "engines/circulant_dma_4cw/" + filename;
    if (fs::exists("../../" + filename)) return "../../" + filename;
    if (fs::exists("../" + filename)) return "../" + filename;
    return filename;
}

// ------------------------------------------------------------------------------
// Run Simulation Workload on Blackhole Hardware / ttsim
static uint32_t get_unique_invocation_id() {
    uint32_t id = 1;
    const char* path = "/tmp/tt_circulant_invocation_id";
    std::ifstream fin(path);
    if (fin.is_open()) {
        fin >> id;
        fin.close();
    } else {
        auto now_sec = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        id = static_cast<uint32_t>(now_sec) & 0x3FFF;
    }
    id = (id + 1) & 0x7FFF;
    if (id <= 1) id = 2;
    std::ofstream fout(path);
    if (fout.is_open()) {
        fout << id;
        fout.close();
    }
    return id;
}

// ------------------------------------------------------------------------------
int run_circulant_dma_simulation(
    double eb_n0_db,
    uint64_t max_blocks,
    uint32_t max_iterations,
    bool single_core,
    bool benchmark_mode = false,
    bool passthrough_mode = false,
    uint32_t invocation_id = 1,
    uint64_t min_fe = 25,
    uint32_t chunk_batches_per_core = 2500,
    const std::string& csv_filename = "",
    uint32_t max_steps_override = 0
) {
    if (invocation_id <= 1) {
        invocation_id = get_unique_invocation_id();
    }

    auto mesh_device = single_core 
        ? distributed::MeshDevice::create_unit_mesh(0)
        : distributed::MeshDevice::create(distributed::MeshDeviceConfig(distributed::MeshShape{2, 2}));
    auto& cq = mesh_device->mesh_command_queue();

    CoreCoord grid_size = mesh_device->compute_with_storage_grid_size();
    uint32_t cores_per_chip = single_core ? 1 : (grid_size.x * grid_size.y);
    uint32_t num_chips = single_core ? 1 : (mesh_device->num_rows() * mesh_device->num_cols());
    uint32_t total_mesh_cores = cores_per_chip * num_chips;

    std::cout << "\n=======================================================" << std::endl;
    std::cout << " RUNNING 4-CODEWORD CIRCULANT DMA ENGINE ON QUIETBOX 2" << std::endl;
    std::cout << " Target: " << (single_core ? "Single Tensix Core (0, 0)" : "2x2 Blackhole Mesh (440 Cores)") << std::endl;
    std::cout << " Target Max Blocks: " << max_blocks << " CWs" << std::endl;
    std::cout << " Eb/N0: " << eb_n0_db << " dB | Max Iterations: " << max_iterations 
              << " | Target FE: " << min_fe << std::endl;
    std::cout << " Invocation ID: " << invocation_id << " (Epoch: 0x" 
              << std::hex << (invocation_id << 16) << std::dec << ")" << std::endl;
    if (benchmark_mode)  std::cout << " Mode: SILICON THROUGHPUT BENCHMARK (sigma=0)" << std::endl;
    if (passthrough_mode) std::cout << " Mode: PASSTHROUGH MEMORY GATHER/SCATTER VERIFICATION" << std::endl;
    std::cout << "=======================================================" << std::endl;

    CoreCoord core = {0, 0};
    CoreRange core_range = single_core 
        ? CoreRange(core, core)
        : CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});

    // Compute Channel Noise Parameters
    double rate = 0.5;
    double snr_lin = std::pow(10.0, eb_n0_db / 10.0);
    float mu_llr = static_cast<float>(4.0 * rate * snr_lin);
    float sigma_llr = benchmark_mode ? 0.0f : static_cast<float>(std::sqrt(8.0 * rate * snr_lin));
    uint32_t mu_u32 = 0;
    uint32_t sigma_bits = 0;
    std::memcpy(&mu_u32, &mu_llr, sizeof(float));
    std::memcpy(&sigma_bits, &sigma_llr, sizeof(float));

    // DRAM Buffer Allocation for Telemetry + Initial & Final Noise Dump (64 KB/core)
    uint32_t stats_per_core_bytes = 64 * 1024;
    uint32_t stats_cb_bytes = single_core ? stats_per_core_bytes : (cores_per_chip * stats_per_core_bytes);

    distributed::DeviceLocalBufferConfig stats_dram_config{
        .page_size = stats_cb_bytes,
        .buffer_type = BufferType::DRAM
    };
    auto stats_dram = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = stats_cb_bytes}, stats_dram_config, mesh_device.get()
    );

    // Compile Device Kernels:
    std::string reader_path  = resolve_filepath("reader_brisc.cpp");
    std::string writer_path  = resolve_filepath("writer_ncrisc.cpp");
    std::string compute_path = resolve_filepath("compute_trisc.cpp");

    DataMovementConfig reader_config{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default};
    DataMovementConfig writer_config{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default};
    ComputeConfig compute_config{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = false};
#if defined(LDPC_USE_SFPU_LUT)
    compute_config.defines["LDPC_USE_SFPU_LUT"] = "1";
#endif

    if (passthrough_mode) {
        reader_config.defines["LDPC_TEST_PASSTHROUGH_MODE"] = "1";
        writer_config.defines["LDPC_TEST_PASSTHROUGH_MODE"] = "1";
        compute_config.defines["LDPC_TEST_PASSTHROUGH_MODE"] = "1";
    }

    uint32_t num_tiles = QC_LDPC_4CW::TOTAL_TILES;

    distributed::MeshWorkload workload;

    struct ChipHandles {
        KernelHandle reader;
        KernelHandle writer;
        KernelHandle compute;
        distributed::MeshCoordinateRange coord_range;
    };
    std::vector<ChipHandles> chip_handles;

    for (uint32_t chip_row = 0; chip_row < (single_core ? 1 : mesh_device->num_rows()); ++chip_row) {
        for (uint32_t chip_col = 0; chip_col < (single_core ? 1 : mesh_device->num_cols()); ++chip_col) {
            uint32_t chip_idx = chip_row * (single_core ? 1 : mesh_device->num_cols()) + chip_col;
            Program program = CreateProgram();

            // Circular Buffers in L1 SRAM (Ping-Pong Double Buffering: 132 KB Total)
            CreateCircularBuffer(program, core_range, CircularBufferConfig(20 * 1024, {{TileMatrixCB::CB_VNODE_A, tt::DataFormat::Float16_b}}).set_page_size(TileMatrixCB::CB_VNODE_A, 2 * 1024));
            CreateCircularBuffer(program, core_range, CircularBufferConfig(20 * 1024, {{TileMatrixCB::CB_VNODE_B, tt::DataFormat::Float16_b}}).set_page_size(TileMatrixCB::CB_VNODE_B, 2 * 1024));
            CreateCircularBuffer(program, core_range, CircularBufferConfig(num_tiles * 2 * 1024, {{TileMatrixCB::CB_R_MSG, tt::DataFormat::Float16_b}}).set_page_size(TileMatrixCB::CB_R_MSG, 2 * 1024));
            CreateCircularBuffer(program, core_range, CircularBufferConfig(4 * 1024, {{TileMatrixCB::CB_SCRATCH_GATHER, tt::DataFormat::Float16_b}}).set_page_size(TileMatrixCB::CB_SCRATCH_GATHER, 2 * 1024));
            CreateCircularBuffer(program, core_range, CircularBufferConfig(4 * 1024, {{TileMatrixCB::CB_SCRATCH_SCATTER, tt::DataFormat::Float16_b}}).set_page_size(TileMatrixCB::CB_SCRATCH_SCATTER, 2 * 1024));
            CreateCircularBuffer(program, core_range, CircularBufferConfig(4 * 1024, {{TileMatrixCB::CB_TELEMETRY, tt::DataFormat::RawUInt32}}).set_page_size(TileMatrixCB::CB_TELEMETRY, 4 * 1024));

            auto reader  = CreateKernel(program, reader_path, core_range, reader_config);
            auto writer  = CreateKernel(program, writer_path, core_range, writer_config);
            auto compute = CreateKernel(program, compute_path, core_range, compute_config);

            for (uint32_t local_core = 0; local_core < cores_per_chip; ++local_core) {
                CoreCoord c = single_core ? CoreCoord{0, 0} : CoreCoord{local_core % grid_size.x, local_core / grid_size.x};
                uint32_t global_core_id = chip_idx * cores_per_chip + local_core;
                uint32_t core_stats_addr = static_cast<uint32_t>(stats_dram->address()) + local_core * stats_per_core_bytes;

                std::vector<uint32_t> r_args = {
                    chunk_batches_per_core, max_iterations, num_tiles, invocation_id, global_core_id, max_steps_override
                };
                SetRuntimeArgs(program, reader, c, r_args);

                std::vector<uint32_t> w_args = {
                    chunk_batches_per_core, max_iterations, num_tiles,
                    core_stats_addr, 0, stats_cb_bytes,
                    mu_u32, sigma_bits, 0x12345678U, invocation_id,
                    max_steps_override, 0u, global_core_id
                };
                SetRuntimeArgs(program, writer, c, w_args);

                std::vector<uint32_t> c_args = {
                    chunk_batches_per_core, max_iterations, num_tiles,
                    mu_u32, sigma_bits, invocation_id,
                    max_steps_override, 0u, global_core_id
                };
                SetRuntimeArgs(program, compute, c, c_args);
            }

            auto coord_range = distributed::MeshCoordinateRange(
                distributed::MeshCoordinate{chip_row, chip_col},
                distributed::MeshCoordinate{chip_row, chip_col}
            );
            workload.add_program(coord_range, std::move(program));
            chip_handles.push_back({reader, writer, compute, coord_range});
        }
    }

    uint64_t total_cws = 0;
    uint64_t total_fe = 0;
    uint64_t total_be = 0;
    uint64_t total_iters = 0;
    double total_sec = 0.0;
    std::vector<uint32_t> core0_stats(stats_per_core_bytes / sizeof(uint32_t), 0);

    // Setup CSV output if specified or default
    std::string out_csv = csv_filename;
    if (out_csv.empty() && !passthrough_mode && !benchmark_mode) {
        fs::create_directories("Results");
        char buf[256];
        if (max_blocks >= 1000000000ULL) {
            std::snprintf(buf, sizeof(buf), "Results/results_circulant_dma_4cw_ar4ja_r12_ebn0_%.2f_%lluB.csv",
                          eb_n0_db, static_cast<unsigned long long>(max_blocks / 1000000000ULL));
        } else if (max_blocks >= 1000000ULL) {
            std::snprintf(buf, sizeof(buf), "Results/results_circulant_dma_4cw_ar4ja_r12_ebn0_%.2f_%lluM.csv",
                          eb_n0_db, static_cast<unsigned long long>(max_blocks / 1000000ULL));
        } else {
            std::snprintf(buf, sizeof(buf), "Results/results_circulant_dma_4cw_ar4ja_r12_ebn0_%.2f_%lluk.csv",
                          eb_n0_db, static_cast<unsigned long long>(max_blocks / 1000ULL));
        }
        out_csv = buf;
    }

    std::ofstream csv_out;
    if (!out_csv.empty()) {
        bool file_exists = fs::exists(out_csv);
        csv_out.open(out_csv, std::ios::app);
        if (!file_exists) {
            csv_out << "EbN0_dB,Total_Blocks,Total_Bit_Errors,Total_Block_Errors,BER,FER,Mean_Iterations,Throughput_Msps,Host_Wall_s\n";
            csv_out.flush();
        }
        std::cout << " [CSV] Logging to: " << out_csv << std::endl;
    }

    std::cout << "[WORKLOAD] Commencing chunked execution across " << total_mesh_cores << " cores..." << std::endl;

    for (uint32_t chunk_idx = 0; total_fe < min_fe && total_cws < max_blocks; ++chunk_idx) {
        uint64_t remaining_blocks = max_blocks - total_cws;
        uint32_t cur_batches = chunk_batches_per_core;
        uint64_t cws_per_chunk = static_cast<uint64_t>(cur_batches) * 4 * total_mesh_cores;

        if (cws_per_chunk > remaining_blocks) {
            cur_batches = static_cast<uint32_t>((remaining_blocks + (4 * total_mesh_cores - 1)) / (4 * total_mesh_cores));
            if (cur_batches == 0) cur_batches = 1;
        }

        uint32_t cur_invocation_id = (invocation_id + chunk_idx) & 0x7FFF;
        if (cur_invocation_id <= 1) cur_invocation_id = 2;

        auto& programs = workload.get_programs();

        for (size_t chip_idx = 0; chip_idx < chip_handles.size(); ++chip_idx) {
            auto& ch = chip_handles[chip_idx];
            Program& program = programs.at(ch.coord_range);

            for (uint32_t local_core = 0; local_core < cores_per_chip; ++local_core) {
                CoreCoord c = single_core ? CoreCoord{0, 0} : CoreCoord{local_core % grid_size.x, local_core / grid_size.x};
                uint32_t global_core_id = chip_idx * cores_per_chip + local_core;
                uint32_t core_stats_addr = static_cast<uint32_t>(stats_dram->address()) + local_core * stats_per_core_bytes;

                std::vector<uint32_t> r_args = {
                    cur_batches, max_iterations, num_tiles, cur_invocation_id, global_core_id, max_steps_override
                };
                SetRuntimeArgs(program, ch.reader, c, r_args);

                std::vector<uint32_t> w_args = {
                    cur_batches, max_iterations, num_tiles,
                    core_stats_addr, 0, stats_cb_bytes,
                    mu_u32, sigma_bits, 0x12345678U, cur_invocation_id,
                    max_steps_override, 0u, global_core_id
                };
                SetRuntimeArgs(program, ch.writer, c, w_args);

                std::vector<uint32_t> c_args = {
                    cur_batches, max_iterations, num_tiles,
                    mu_u32, sigma_bits, cur_invocation_id,
                    max_steps_override, 0u, global_core_id
                };
                SetRuntimeArgs(program, ch.compute, c, c_args);
            }
        }

        auto t_chunk_start = std::chrono::high_resolution_clock::now();
        distributed::EnqueueMeshWorkload(cq, workload, false);
        cq.finish();
        auto t_chunk_end = std::chrono::high_resolution_clock::now();
        double chunk_sec = std::chrono::duration<double>(t_chunk_end - t_chunk_start).count();
        total_sec += chunk_sec;

        uint32_t chunk_be = 0;
        uint32_t chunk_fe = 0;
        uint64_t chunk_iters = 0;
        uint32_t chunk_cws = 0;

        for (uint32_t r = 0; r < (single_core ? 1 : mesh_device->num_rows()); ++r) {
            for (uint32_t c = 0; c < (single_core ? 1 : mesh_device->num_cols()); ++c) {
                std::vector<uint32_t> chip_stats(stats_cb_bytes / sizeof(uint32_t), 0);
                distributed::ReadShard(cq, chip_stats, stats_dram, distributed::MeshCoordinate(r, c), true);
                if (r == 0 && c == 0 && chunk_idx == 0) {
                    std::copy(chip_stats.begin(), chip_stats.begin() + core0_stats.size(), core0_stats.begin());
                }
                for (uint32_t core = 0; core < cores_per_chip; ++core) {
                    uint32_t off = core * (stats_per_core_bytes / sizeof(uint32_t));
                    chunk_be += chip_stats[off + TelemSlot::TOTAL_BE];
                    chunk_fe += chip_stats[off + TelemSlot::TOTAL_FE];
                    chunk_iters += chip_stats[off + TelemSlot::TOTAL_ITERS];
                    chunk_cws += chip_stats[off + TelemSlot::TOTAL_CWS];
                }
            }
        }

        total_be += chunk_be;
        total_fe += chunk_fe;
        total_iters += chunk_iters;
        total_cws += chunk_cws;

        double cur_fer = (total_cws > 0) ? (static_cast<double>(total_fe) / total_cws) : 0.0;
        double cur_ber = (total_cws > 0) ? (static_cast<double>(total_be) / (total_cws * 1024.0)) : 0.0;
        double cur_mean_iter = (total_cws > 0) ? (static_cast<double>(total_iters) / total_cws) : 0.0;
        double cur_msps = (total_cws * 2048.0) / (total_sec * 1e6);

        std::cout << "[CHUNK " << std::setw(4) << chunk_idx + 1 << "] "
                  << "Blocks: " << std::setw(11) << total_cws << " / " << max_blocks
                  << " | FE: " << std::setw(4) << total_fe << " / " << min_fe
                  << " | BE: " << std::setw(6) << total_be
                  << " | FER: " << std::scientific << std::setprecision(2) << cur_fer
                  << " | MeanIter: " << std::fixed << std::setprecision(2) << cur_mean_iter
                  << " | Rate: " << std::setprecision(1) << cur_msps << " Msps"
                  << " | Wall: " << std::setprecision(1) << total_sec << "s"
                  << std::endl;
        // Explicitly flush stdout after each chunk to ensure unbuffered real-time streaming
        // over pipes, tmux sessions, and background log monitors (e.g. tail -f).
        std::fflush(stdout);

        if (csv_out.is_open()) {
            csv_out << std::fixed << std::setprecision(2) << eb_n0_db << ","
                    << total_cws << ","
                    << total_be << ","
                    << total_fe << ","
                    << std::scientific << std::setprecision(6) << cur_ber << ","
                    << cur_fer << ","
                    << std::fixed << std::setprecision(3) << cur_mean_iter << ","
                    << std::setprecision(2) << cur_msps << ","
                    << total_sec << "\n";
            csv_out.flush();
        }

        if (passthrough_mode) {
            break;
        }
    }

    double fer = (total_cws > 0) ? (static_cast<double>(total_fe) / total_cws) : 0.0;
    double ber = (total_cws > 0) ? (static_cast<double>(total_be) / (total_cws * 1024.0)) : 0.0;
    double mean_iter = (total_cws > 0) ? (static_cast<double>(total_iters) / total_cws) : 0.0;
    double msps = (total_cws * 2048.0) / (total_sec * 1e6);
    double mbps_info = (total_cws * 1024.0) / (total_sec * 1e6);

    std::cout << "\n=======================================================" << std::endl;
    std::cout << " CIRCULANT DMA DECODER SILICON RESULTS" << std::endl;
    std::cout << "=======================================================" << std::endl;
    std::cout << "  Active Cores:     " << total_mesh_cores << std::endl;
    std::cout << "  Total Blocks:     " << total_cws << std::endl;
    std::cout << "  Frame Errors:     " << total_fe << std::endl;
    std::cout << "  Bit Errors:       " << total_be << std::endl;
    std::cout << "  FER:              " << std::scientific << std::setprecision(4) << fer << std::endl;
    std::cout << "  BER:              " << std::scientific << std::setprecision(4) << ber << std::endl;
    std::cout << "  Mean Iterations:  " << std::fixed << std::setprecision(3) << mean_iter << std::endl;
    std::cout << "  Elapsed Time:     " << std::setprecision(2) << total_sec << " s" << std::endl;
    std::cout << "  Raw Throughput:   " << std::setprecision(2) << msps << " Msps ("
              << (total_cws / total_sec) << " blocks/sec)" << std::endl;
    std::cout << "  Info Throughput:  " << std::setprecision(2) << mbps_info << " Mbps ("
              << (mbps_info / 1000.0) << " Gbps)" << std::endl;

    uint32_t trapped_batch = core0_stats[TelemSlot::FIRST_FAILING_BATCH];
    uint32_t trapped_cw    = core0_stats[TelemSlot::FIRST_FAILING_CW];
    uint32_t trapped_be    = core0_stats[TelemSlot::FIRST_FAILING_BE];
    uint32_t trapped_mask  = core0_stats[TelemSlot::FIRST_FAILING_MASK];

    if (trapped_batch != 0xFFFFFFFF || max_blocks <= 100) {
        std::cout << "-------------------------------------------------------" << std::endl;
        if (trapped_batch != 0xFFFFFFFF) {
            std::cout << " >>> [FIRST BLOCK ERROR TRAPPED ON CORE (0,0)] <<<" << std::endl;
            std::cout << "  Failing Batch:     " << trapped_batch << std::endl;
            std::cout << "  Failing Codeword:  " << trapped_cw << std::endl;
            std::cout << "  Bit Errors:        " << trapped_be << std::endl;
            std::cout << "  Active Mask:       0x" << std::hex << trapped_mask << std::dec << std::endl;
        } else {
            std::cout << " >>> [CAPTURING BATCH 0 VNODE STATE ON CORE (0,0)] <<<" << std::endl;
        }

        const char* vnode_init_ptr = reinterpret_cast<const char*>(core0_stats.data()) + 4096;
        std::string init_dump_path = "tensix_initial_vnodes.bin";
        std::ofstream fout_init(init_dump_path, std::ios::binary);
        if (fout_init.is_open()) {
            fout_init.write(vnode_init_ptr, 20480);
            fout_init.close();
            std::cout << "  [SUCCESS] Saved Initial Noise Dump: " << init_dump_path << " (20,480 Bytes)" << std::endl;
        } else {
            std::cerr << "  [ERROR] Failed to write " << init_dump_path << std::endl;
        }

        const char* vnode_final_ptr = reinterpret_cast<const char*>(core0_stats.data()) + 4096 + 20480;
        std::string final_dump_path = "tensix_final_vnodes.bin";
        std::ofstream fout_final(final_dump_path, std::ios::binary);
        if (fout_final.is_open()) {
            fout_final.write(vnode_final_ptr, 20480);
            fout_final.close();
            std::cout << "  [SUCCESS] Saved Final Vnodes Dump:  " << final_dump_path << " (20,480 Bytes)" << std::endl;
        } else {
            std::cerr << "  [ERROR] Failed to write " << final_dump_path << std::endl;
        }
    }

    std::cout << "-------------------------------------------------------" << std::endl;
    uint32_t tiles_sim = core0_stats[TelemSlot::TOTAL_TILES_SIM];
    double tiles_d = (tiles_sim > 0) ? static_cast<double>(tiles_sim) : 1.0;

    double cyc_stage_gather  = core0_stats[TelemSlot::STAGE_GATHER_CYCLES]  / tiles_d;
    double cyc_stage_unpack  = core0_stats[TelemSlot::STAGE_UNPACK_CYCLES]  / tiles_d;
    double cyc_stage_math    = core0_stats[TelemSlot::STAGE_MATH_CYCLES]    / tiles_d;
    double cyc_stage_pack    = core0_stats[TelemSlot::STAGE_PACK_CYCLES]    / tiles_d;
    double cyc_stage_scatter = core0_stats[TelemSlot::STAGE_SCATTER_CYCLES] / tiles_d;
    double cyc_stage_total   = core0_stats[TelemSlot::STAGE_TOTAL_CYCLES]   / tiles_d;

    std::cout << "-------------------------------------------------------" << std::endl;
    std::cout << " HARDWARE CYCLE TELEMETRY (Core 0,0 Timers)" << std::endl;
    std::cout << "  Total Elapsed Cycles:  " << core0_stats[TelemSlot::ELAPSED_CYCLES] << std::endl;
    std::cout << "  Cycles Per Batch:      " << core0_stats[TelemSlot::CYCLES_PER_BATCH] << std::endl;
    std::cout << "  Channel Gen Cycles:    " << core0_stats[TelemSlot::CYCLES_CHANNEL] << std::endl;
    std::cout << "  Error Scan Cycles:     " << core0_stats[TelemSlot::CYCLES_ERROR_SCAN] << std::endl;
    std::cout << "-------------------------------------------------------" << std::endl;
    std::cout << " GAPLESS SEQUENTIAL PER-TILE BREAKDOWN (T0 -> T5)" << std::endl;
    std::cout << "  Stage 1: Gather  (W0+W1 512-bit DMA) : " << std::fixed << std::setprecision(1) << cyc_stage_gather  << " cyc/tile  (T1 - T0)" << std::endl;
    std::cout << "  Stage 2: Unpack  (TRISC0 CB -> DEST) : " << std::fixed << std::setprecision(1) << cyc_stage_unpack  << " cyc/tile  (T2 - T1)" << std::endl;
    std::cout << "  Stage 3: Math    (TRISC1 Pure SFPU)  : " << std::fixed << std::setprecision(1) << cyc_stage_math    << " cyc/tile  (T3 - T2)" << std::endl;
    std::cout << "  Stage 4: Pack    (TRISC2 DEST -> CB) : " << std::fixed << std::setprecision(1) << cyc_stage_pack    << " cyc/tile  (T4 - T3)" << std::endl;
    std::cout << "  Stage 5: Scatter (W0+W1 512-bit DMA) : " << std::fixed << std::setprecision(1) << cyc_stage_scatter << " cyc/tile  (T5 - T4)" << std::endl;
    std::cout << " ------------------------------------------------------" << std::endl;
    std::cout << "  Exact Tile Turnaround (Sum of 1..5)  : " << std::fixed << std::setprecision(1) << cyc_stage_total   << " cyc/tile  (T5 - T0)" << std::endl;
    std::cout << "=======================================================" << std::endl;

    if (passthrough_mode) {
        std::cout << "\n=======================================================" << std::endl;
        std::cout << " PASSTHROUGH MEMORY GATHER/SCATTER VERIFICATION" << std::endl;
        std::cout << "=======================================================" << std::endl;
        std::cout << "  Total Simulated Tiles: " << core0_stats[TelemSlot::TOTAL_TILES_SIM] << std::endl;
        std::cout << "  Total Word Mismatches: " << total_be << std::endl;
        std::cout << "  Failing Batches:       " << total_fe << " / " << (total_cws / 4) << std::endl;
        if (total_be == 0) {
            std::cout << "  [SUCCESS] 100% BIT-EXACT MATCH! 0 mismatches across all variable nodes." << std::endl;
            std::cout << "  Circulant 512-bit DMA transfers and gather/scatter symmetry CONFIRMED!" << std::endl;
        } else {
            std::cout << "  [FAILURE] " << total_be << " mismatches detected in vnode LLRs!" << std::endl;
        }
        std::cout << "=======================================================\n" << std::endl;
    }

    mesh_device->close();
    if (passthrough_mode) {
        return (total_be == 0) ? 0 : 1;
    }
    return (total_fe <= min_fe) ? 0 : 1;
}

int main(int argc, char** argv) {
    // Disable stdout stream buffering to prevent glibc block-buffering delays (4KB/64KB)
    // when output is redirected to pipes, tee, or background log files.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    double eb_n0_db = 2.5;
    uint64_t max_blocks = 10000000000ULL; // 10 Billion codewords default
    uint32_t max_iterations = 200;
    bool single_core = false;     // Default to full 440-core mesh
    bool benchmark = false;
    bool passthrough = false;
    uint64_t min_fe = 25;
    uint32_t chunk_batches = 2500; // 2,500 batches/core = 10k CWs/core = 4.4M CWs/chunk across 440 cores
    std::string csv_path = "";
    uint32_t max_steps = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--single-core")  single_core = true;
        else if (arg == "--mesh")    single_core = false;
        else if (arg == "--benchmark") benchmark = true;
        else if (arg == "--passthrough") passthrough = true;
        else if (arg == "--min-fe" && i + 1 < argc) min_fe = std::stoull(argv[++i]);
        else if (arg == "--chunk-batches" && i + 1 < argc) chunk_batches = std::stoul(argv[++i]);
        else if (arg == "--out" && i + 1 < argc) csv_path = argv[++i];
        else if (arg == "--max-steps" && i + 1 < argc) max_steps = std::stoul(argv[++i]);
    }

    if (argc > 1 && argv[1][0] != '-') eb_n0_db = std::stod(argv[1]);
    if (argc > 2 && argv[2][0] != '-') max_blocks = std::stoull(argv[2]);
    if (argc > 3 && argv[3][0] != '-') max_iterations = std::stoul(argv[3]);

    return run_circulant_dma_simulation(
        eb_n0_db, max_blocks, max_iterations, single_core, benchmark, passthrough,
        1, min_fe, chunk_batches, csv_path, max_steps
    );
}
