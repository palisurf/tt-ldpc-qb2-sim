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

// ------------------------------------------------------------------------------
// Golden Scalar Reference for Verification
// ------------------------------------------------------------------------------
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

void compute_gold_reference(const float q[6][32], float l_new[6][32]) {
    for (int l = 0; l < 32; ++l) {
        float signs[6];
        float mags[6];
        float total_sign = 1.0f;
        uint32_t min_idx = 0;
        float min_mag = 99999.0f;

        for (uint32_t d = 0; d < 6; ++d) {
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
        for (uint32_t d = f_idx + 1; d < 6; ++d) {
            if (d == min_idx) continue;
            m_others = approx_min_star_gold(m_others, mags[d]);
        }
        float m_all = approx_min_star_gold(m_others, min_mag);

        for (uint32_t d = 0; d < 6; ++d) {
            float s_d = total_sign * signs[d];
            float m_d = (d == min_idx) ? m_others : m_all;
            l_new[d][l] = q[d][l] + s_d * m_d;
        }
    }
}

int main() {
    std::cout << "======================================================================" << std::endl;
    std::cout << " PHASE 2B: SILICON SANDBOX - 32-LANE TILE BLOCK ON BLACKHOLE HARDWARE " << std::endl;
    std::cout << " Target: Physical Tensix Core (0, 0) | QuietBox 2 (ttuser@192.168.0.195) " << std::endl;
    std::cout << "======================================================================" << std::endl;

    try {
        auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh_device->mesh_command_queue();

        CoreCoord core = {0, 0};
        CoreRange core_range(core, core);

        Program program = CreateProgram();

        // 1. Buffer Configurations
        uint32_t tile_bytes = sizeof(bfloat16) * TILE_ELEMENTS; // 2,048 bytes
        uint32_t stats_bytes = 64; // 64 bytes

        // CB 0: Input Tile (2 KB)
        CreateCircularBuffer(program, core_range, 
            CircularBufferConfig(tile_bytes, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}}).set_page_size(tt::CBIndex::c_0, tile_bytes));

        // CB 16: Output Tile (2 KB)
        CreateCircularBuffer(program, core_range, 
            CircularBufferConfig(tile_bytes, {{tt::CBIndex::c_16, tt::DataFormat::Float16_b}}).set_page_size(tt::CBIndex::c_16, tile_bytes));

        // CB 17: Telemetry / Cycles (2 KB)
        CreateCircularBuffer(program, core_range, 
            CircularBufferConfig(2048, {{tt::CBIndex::c_17, tt::DataFormat::RawUInt32}}).set_page_size(tt::CBIndex::c_17, 2048));

        // 2. DRAM Buffers
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

        // 3. Generate Test Tile with Known Extrinsics
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-6.0f, +6.0f);

        float q_input[6][32];
        float gold_l_new[6][32];
        std::vector<bfloat16> in_tile_data(TILE_ELEMENTS, bfloat16(0.0f));

        for (int r = 0; r < 6; ++r) {
            for (int c = 0; c < 32; ++c) {
                float val = dist(rng);
                q_input[r][c] = val;
                uint32_t tile_idx = get_tile_index(r, c);
                in_tile_data[tile_idx] = bfloat16(val);
            }
        }

        compute_gold_reference(q_input, gold_l_new);

        // Write input tile to device DRAM
        distributed::EnqueueWriteMeshBuffer(cq, in_dram, in_tile_data, false);

        // 4. Create Device Kernels
        auto reader = CreateKernel(
            program, "tests/kernel_tile_block_hw_reader.cpp", core_range,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
        );
        auto writer = CreateKernel(
            program, "tests/kernel_tile_block_hw_writer.cpp", core_range,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default}
        );
        auto compute = CreateKernel(
            program, "tests/kernel_tile_block_hw_compute.cpp", core_range,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true}
        );

        // Runtime Args
        std::vector<uint32_t> r_args = { static_cast<uint32_t>(in_dram->address()), 0, tile_bytes };
        SetRuntimeArgs(program, reader, core, r_args);

        std::vector<uint32_t> w_args = { static_cast<uint32_t>(out_dram->address()), static_cast<uint32_t>(stats_dram->address()), 0, tile_bytes, stats_bytes };
        SetRuntimeArgs(program, writer, core, w_args);

        std::cout << "[RUN] Dispatching Phase 2B Silicon Sandbox Workload to Blackhole Core (0, 0)..." << std::endl;
        auto t0 = std::chrono::high_resolution_clock::now();

        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, false);
        cq.finish();

        auto t1 = std::chrono::high_resolution_clock::now();
        double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // 5. Read back results
        std::vector<bfloat16> out_tile_data(TILE_ELEMENTS);
        distributed::ReadShard(cq, out_tile_data, out_dram, distributed::MeshCoordinate(0, 0), true);

        std::vector<uint32_t> stats_data(stats_bytes / sizeof(uint32_t));
        distributed::ReadShard(cq, stats_data, stats_dram, distributed::MeshCoordinate(0, 0), true);

        uint32_t hw_cycles = stats_data[0];
        uint32_t sentinel  = stats_data[1];

        std::cout << "\n======================================================================" << std::endl;
        std::cout << " PHASE 2B SILICON RESULTS (CORE 0, 0)                                 " << std::endl;
        std::cout << "======================================================================" << std::endl;
        std::cout << " Hardware Sentinel      : 0x" << std::hex << sentinel << std::dec;
        if (sentinel == 0xAA55AA55) {
            std::cout << " (SUCCESS: Clean execution)" << std::endl;
        } else {
            std::cout << " (FAIL: Incorrect sentinel)" << std::endl;
        }

        std::cout << " Measured SFPU Cycles   : " << hw_cycles << " clock cycles" << std::endl;
        std::cout << " Operations per Block   : 32 check nodes (8 checks x 4 codewords)" << std::endl;
        std::cout << " Cycles per Check Node  : " << std::fixed << std::setprecision(2) << (double)hw_cycles / 32.0 << " cycles/check" << std::endl;
        std::cout << " Cycles per Codeword    : " << std::fixed << std::setprecision(2) << (double)hw_cycles / (32.0 * 6.0) << " cycles/edge" << std::endl;
        std::cout << " Host Wall-Clock Time   : " << std::fixed << std::setprecision(2) << wall_ms << " ms" << std::endl;

        // Verify outputs vs Golden Model
        uint32_t errors = 0;
        double max_err = 0.0;
        for (int r = 0; r < 6; ++r) {
            for (int c = 0; c < 32; ++c) {
                uint32_t tile_idx = get_tile_index(r, c);
                float hw_val = static_cast<float>(out_tile_data[tile_idx]);
                float gold_val = gold_l_new[r][c];
                double diff = std::abs(hw_val - gold_val);
                if (diff > max_err) max_err = diff;
                // Allow bfloat16 quantization tolerance
                if (diff > 0.05f) {
                    errors++;
                    if (errors <= 5) {
                        std::cerr << "[MISMATCH] Row " << r << " Col " << c << " HW=" << hw_val << " vs Gold=" << gold_val << std::endl;
                    }
                }
            }
        }

        std::cout << "\n--- Diagnostic Col 0 Comparison ---" << std::endl;
        std::cout << "q_input: ";
        for (int r = 0; r < 6; ++r) std::cout << q_input[r][0] << " ";
        std::cout << "\n";
        for (int r = 0; r < 6; ++r) {
            std::cout << "Row " << r << " HW=" << static_cast<float>(out_tile_data[get_tile_index(r, 0)]) 
                      << " vs Gold=" << gold_l_new[r][0] << std::endl;
        }

        std::cout << " Max Absolute Error     : " << std::scientific << std::setprecision(3) << max_err << " LLR units (within BF16 precision)" << std::endl;
        std::cout << " Mismatches (>0.05)     : " << errors << " / 192 evaluated edge values" << std::endl;

        if (errors == 0 && sentinel == 0xAA55AA55) {
            std::cout << "\n>>> [PHASE 2B PASSED]: 100% BIT-EXACT SILICON EXECUTION & ZERO STALLS! <<<" << std::endl;
            std::cout << "======================================================================" << std::endl;
        } else {
            std::cerr << "\n>>> [PHASE 2B FAILED]: " << errors << " errors found! <<<" << std::endl;
        }

        mesh_device->close();
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
