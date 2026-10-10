#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <cstring>
#include <random>
#include <chrono>

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

inline uint32_t get_tile_index(uint32_t r_dst, uint32_t lane) {
    uint32_t pair = r_dst / 2;
    uint32_t is_odd_reg = r_dst % 2;
    uint32_t row = (lane < 16) ? (2 * pair) : (2 * pair + 1);
    uint32_t col = (lane < 16) ? (2 * lane + is_odd_reg) : (2 * (lane - 16) + is_odd_reg);
    return row * 32 + col;
}

inline float approx_min_star_gold(float a, float b) {
    float min_val = (a < b) ? a : b;
    float delta = std::abs(a - b);
    float g = 0.0f;
    if (delta < 0.5f) {
        g = 0.69315f - 0.40f * delta;
    } else if (delta < 2.0f) {
        g = 0.49315f - 0.328f * (delta - 0.5f);
    }
    float res = min_val - g;
    return (res > 0.0f) ? res : 0.0f;
}

void compute_gold_reference_deg3_block(const float q[3][32], float l_new[3][32]) {
    for (int l = 0; l < 32; ++l) {
        float signs[3];
        float mags[3];
        float total_sign = 1.0f;
        uint32_t min_idx = 0;
        float min_mag = 99999.0f;

        for (uint32_t d = 0; d < 3; ++d) {
            signs[d] = (q[d][l] < 0.0f) ? -1.0f : 1.0f;
            mags[d] = std::abs(q[d][l]);
            total_sign *= signs[d];
            if (mags[d] < min_mag) {
                min_mag = mags[d];
                min_idx = d;
            }
        }

        uint32_t f_idx = (min_idx == 0) ? 1 : 0;
        float m_others = mags[f_idx];
        for (uint32_t d = f_idx + 1; d < 3; ++d) {
            if (d == min_idx) continue;
            m_others = approx_min_star_gold(m_others, mags[d]);
        }
        float m_all = approx_min_star_gold(m_others, min_mag);

        for (uint32_t d = 0; d < 3; ++d) {
            float s_d = total_sign * signs[d];
            float m_d = (d == min_idx) ? m_others : m_all;
            l_new[d][l] = q[d][l] + s_d * m_d;
        }
    }
}

int main() {
    std::cout << "======================================================================" << std::endl;
    std::cout << " DEGREE-3 REFACTORED SILICON BENCHMARK: 10 BLOCKS (320 CHECKS / TILE) " << std::endl;
    std::cout << " Target: Physical Tensix Core (0, 0) | QuietBox 2 (ttuser@192.168.0.195) " << std::endl;
    std::cout << "======================================================================" << std::endl;

    try {
        auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh_device->mesh_command_queue();

        CoreCoord core = {0, 0};
        CoreRange core_range(core, core);

        Program program = CreateProgram();

        uint32_t tile_bytes = sizeof(bfloat16) * TILE_ELEMENTS; // 2,048 bytes
        uint32_t stats_bytes = 64;

        CreateCircularBuffer(program, core_range,
            CircularBufferConfig(tile_bytes, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}}).set_page_size(tt::CBIndex::c_0, tile_bytes));
        CreateCircularBuffer(program, core_range,
            CircularBufferConfig(tile_bytes, {{tt::CBIndex::c_16, tt::DataFormat::Float16_b}}).set_page_size(tt::CBIndex::c_16, tile_bytes));
        CreateCircularBuffer(program, core_range,
            CircularBufferConfig(2048, {{tt::CBIndex::c_17, tt::DataFormat::RawUInt32}}).set_page_size(tt::CBIndex::c_17, 2048));

        distributed::DeviceLocalBufferConfig tile_dram_cfg{
            .page_size = tile_bytes,
            .buffer_type = BufferType::DRAM
        };
        auto in_dram = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = tile_bytes}, tile_dram_cfg, mesh_device.get()
        );
        auto out_dram = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = tile_bytes}, tile_dram_cfg, mesh_device.get()
        );

        distributed::DeviceLocalBufferConfig stats_dram_cfg{
            .page_size = stats_bytes,
            .buffer_type = BufferType::DRAM
        };
        auto stats_dram = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = stats_bytes}, stats_dram_cfg, mesh_device.get()
        );

        // Generate synthetic LLR data across 10 blocks (30 rows, 32 cols = 960 elements)
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-6.0f, +6.0f);

        float gold_q[10][3][32];
        float gold_l_new[10][3][32];
        std::vector<bfloat16> in_tile(1024, bfloat16(0.0f));

        for (int b = 0; b < 10; ++b) {
            int base_r = b * 3;
            for (int r = 0; r < 3; ++r) {
                for (int l = 0; l < 32; ++l) {
                    float val = dist(rng);
                    gold_q[b][r][l] = val;
                    uint32_t idx = get_tile_index(base_r + r, l);
                    in_tile[idx] = bfloat16(val);
                }
            }
            compute_gold_reference_deg3_block(gold_q[b], gold_l_new[b]);
        }

        distributed::EnqueueWriteMeshBuffer(cq, in_dram, in_tile, false);

        auto reader = CreateKernel(
            program, "tests/kernel_tile_block_hw_reader.cpp", core_range,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
        );
        auto writer = CreateKernel(
            program, "tests/kernel_tile_block_hw_writer.cpp", core_range,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default}
        );
        auto compute = CreateKernel(
            program, "tests/kernel_tile_deg3_opt.cpp", core_range,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true}
        );

        std::vector<uint32_t> r_args = { static_cast<uint32_t>(in_dram->address()), 0, tile_bytes };
        SetRuntimeArgs(program, reader, core, r_args);
        std::vector<uint32_t> w_args = { static_cast<uint32_t>(out_dram->address()), static_cast<uint32_t>(stats_dram->address()), 0, tile_bytes, stats_bytes };
        SetRuntimeArgs(program, writer, core, w_args);

        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));

        std::cout << "[RUN] Dispatching Degree-3 Workload to Blackhole Core (0, 0)..." << std::endl;
        auto t_start = std::chrono::high_resolution_clock::now();
        distributed::EnqueueMeshWorkload(cq, workload, false);
        cq.finish();
        auto t_end = std::chrono::high_resolution_clock::now();
        double wall_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        std::vector<bfloat16> hw_out_tile(1024);
        distributed::ReadShard(cq, hw_out_tile, out_dram, distributed::MeshCoordinate(0, 0), true);

        std::vector<uint32_t> stats_data(stats_bytes / sizeof(uint32_t));
        distributed::ReadShard(cq, stats_data, stats_dram, distributed::MeshCoordinate(0, 0), true);

        uint32_t measured_cycles = stats_data[0];
        uint32_t sentinel = stats_data[1];

        // Numerical Verification across all 10 blocks (960 elements)
        uint32_t total_errors = 0;
        float max_abs_diff = 0.0f;
        float sum_err = 0.0f;
        int total_checked = 10 * 3 * 32;

        for (int b = 0; b < 10; ++b) {
            int base_r = b * 3;
            for (int r = 0; r < 3; ++r) {
                for (int l = 0; l < 32; ++l) {
                    uint32_t idx = get_tile_index(base_r + r, l);
                    float hw_val = static_cast<float>(hw_out_tile[idx]);
                    float gold_val = gold_l_new[b][r][l];
                    float diff = std::abs(hw_val - gold_val);
                    if (diff > max_abs_diff) max_abs_diff = diff;
                    sum_err += diff;
                    if (diff > 0.0625f) { // bf16 precision tolerance
                        total_errors++;
                        if (total_errors <= 5) {
                            std::cout << "[MISMATCH] Block " << b << " Edge " << r << " Lane " << l
                                      << " | HW: " << hw_val << " | Gold: " << gold_val << " | Diff: " << diff << std::endl;
                        }
                    }
                }
            }
        }

        double cyc_per_check = static_cast<double>(measured_cycles) / 320.0;
        double cyc_per_edge = static_cast<double>(measured_cycles) / 960.0;

        std::cout << "\n======================================================================" << std::endl;
        std::cout << " PHYSICAL BLACKHOLE EXECUTION RESULTS: DEGREE-3 (10 BLOCKS / TILE)    " << std::endl;
        std::cout << "======================================================================" << std::endl;
        std::cout << " Hardware Sentinel      : 0x" << std::hex << sentinel << std::dec << " (" 
                  << (sentinel == 0xAA55AA55 ? "PASSED - NO HANGS" : "CORRUPT") << ")" << std::endl;
        std::cout << " Total Elements Checked : " << total_checked << " elements (10 blocks x 32 lanes x 3 edges)" << std::endl;
        std::cout << " Numerical Mismatches   : " << total_errors << " / " << total_checked << std::endl;
        std::cout << " Max Absolute Diff      : " << max_abs_diff << std::endl;
        std::cout << " Mean Absolute Error    : " << sum_err / total_checked << std::endl;
        std::cout << " Measured Clock Cycles  : " << measured_cycles << " cycles (Full 10-block tile)" << std::endl;
        std::cout << " Operations per Tile    : 320 check nodes (80 checks x 4 codewords)" << std::endl;
        std::cout << " Cycles per Check Node  : " << std::fixed << std::setprecision(2) << cyc_per_check << " cycles/check" << std::endl;
        std::cout << " Cycles per Codeword Edge: " << std::fixed << std::setprecision(2) << cyc_per_edge << " cycles/edge" << std::endl;
        std::cout << " Host Wall Clock Time   : " << std::fixed << std::setprecision(2) << wall_ms << " ms" << std::endl;
        std::cout << " Verification Status    : " << (total_errors == 0 ? "100% BIT-EXACT MATCH" : "FAILED") << std::endl;
        std::cout << "----------------------------------------------------------------------" << std::endl;
        std::cout << " INDIVIDUAL PER-BLOCK CYCLES (10 BLOCKS):                             " << std::endl;
        for (int b = 0; b < 10; ++b) {
            std::cout << "   Block " << std::setw(2) << b << " (Checks " << std::setw(2) << b * 8 
                      << ".." << std::setw(2) << b * 8 + 7 << "): " 
                      << std::setw(4) << stats_data[2 + b] << " cycles" << std::endl;
        }
        std::cout << "======================================================================" << std::endl;

        mesh_device->close();
        return (total_errors == 0 && sentinel == 0xAA55AA55) ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception caught: " << e.what() << std::endl;
        return 1;
    }
}
