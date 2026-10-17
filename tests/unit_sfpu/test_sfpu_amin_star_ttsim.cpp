#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <cstring>
#include <chrono>
#include <algorithm>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/circular_buffer_config.hpp>

#include "../../engines/tile_matrix_4cw/ldpc_tile_matrix_core.h"

using namespace tt;
using namespace tt::tt_metal;

// Physically-verified mapping from destination register index (0..31) and vector lane (0..31)
// to standard 32x32 tile element index in DRAM / L1:
//   - 4 Faces (f = reg_idx / 8)
//   - 4 Blocks per face (b = (reg_idx % 8) / 2), each 4 rows high
//   - 2 interleaved registers per block: even cols (reg_idx % 2 == 0) and odd cols (reg_idx % 2 == 1)
//   - 32 lanes = 4 rows x 8 columns (row_in_block = lane / 8, col_in_row = lane % 8)
inline uint32_t dst_reg_to_element_index(uint32_t reg_idx, uint32_t lane) {
    uint32_t f = reg_idx / 8;
    uint32_t b = (reg_idx % 8) / 2;
    uint32_t is_odd = reg_idx % 2;
    uint32_t row_in_block = lane / 8;
    uint32_t col_in_row   = lane % 8;
    uint32_t row_in_face  = b * 4 + row_in_block;
    uint32_t col_in_face  = 2 * col_in_row + is_odd;
    return f * 256 + row_in_face * 16 + col_in_face;
}

// Golden scalar Approximate-Min* function (matches SFPU branchless spline or LUT)
inline float gold_approx_min_star(float a, float b) {
    float min_v = std::min(a, b);
    float max_v = std::max(a, b);
    float diff  = max_v - min_v;
#if defined(LDPC_USE_SFPU_LUT)
    float slope = 0.0f;
    float intercept = 0.0f;
    if (diff < 0.5f) {
        slope = -0.4000f;
        intercept = 0.69315f;
    } else if (diff < 2.0f) {
        slope = -0.3280f;
        intercept = 0.65715f;
    } else {
        slope = 0.0f;
        intercept = 0.0f;
    }
    float corr = slope * diff + intercept;
    return std::max(min_v - corr, 0.0f);
#else
    float line1 = 0.69315f - 0.4000f * diff;
    float line2 = 0.65715f - 0.3280f * diff;
    float corr  = std::max(std::max(line1, line2), 0.0f);
    return std::max(min_v - corr, 0.0f);
#endif
}

// Golden scalar Degree-6 check node
struct CheckDeg6Gold {
    float r_out[6];
    float l_out[6];
    uint32_t min_idx;
    float m_others;
    float m_all;
};

CheckDeg6Gold compute_gold_deg6(const float q[6]) {
    CheckDeg6Gold res{};
    float signs[6], mags[6];
    float total_sign = 1.0f;
    uint32_t min_idx = 0;
    float min_mag = 99999.0f;

    for (int r = 0; r < 6; ++r) {
        signs[r] = (q[r] < 0.0f) ? -1.0f : 1.0f;
        mags[r] = std::fabs(q[r]);
        total_sign *= signs[r];
        if (mags[r] < min_mag) {
            min_mag = mags[r];
            min_idx = r;
        }
    }

    float m_others = (min_idx == 0) ? 1000.0f : mags[0];
    for (int r = 1; r < 6; ++r) {
        float m = (min_idx == r) ? 1000.0f : mags[r];
        m_others = gold_approx_min_star(m_others, m);
    }
    float m_all = gold_approx_min_star(m_others, min_mag);

    res.min_idx = min_idx;
    res.m_others = m_others;
    res.m_all = m_all;

    for (int r = 0; r < 6; ++r) {
        float edge_sign = total_sign * signs[r];
        float m_d = (r == min_idx) ? m_others : m_all;
        res.r_out[r] = edge_sign * m_d;
        res.l_out[r] = q[r] + res.r_out[r];
    }
    return res;
}

// Golden scalar Degree-3 check node
struct CheckDeg3Gold {
    float r_out[3];
    float l_out[3];
};

CheckDeg3Gold compute_gold_deg3(const float q[3]) {
    CheckDeg3Gold res{};
    float signs[3], mags[3];
    float total_sign = 1.0f;
    uint32_t min_idx = 0;
    float min_mag = 99999.0f;

    for (int r = 0; r < 3; ++r) {
        signs[r] = (q[r] < 0.0f) ? -1.0f : 1.0f;
        mags[r] = std::fabs(q[r]);
        total_sign *= signs[r];
        if (mags[r] < min_mag) {
            min_mag = mags[r];
            min_idx = r;
        }
    }

    float m_others = (min_idx == 0) ? 1000.0f : mags[0];
    for (int r = 1; r < 3; ++r) {
        float m = (min_idx == r) ? 1000.0f : mags[r];
        m_others = gold_approx_min_star(m_others, m);
    }
    float m_all = gold_approx_min_star(m_others, min_mag);

    for (int r = 0; r < 3; ++r) {
        float edge_sign = total_sign * signs[r];
        float m_d = (r == min_idx) ? m_others : m_all;
        res.r_out[r] = edge_sign * m_d;
        res.l_out[r] = q[r] + res.r_out[r];
    }
    return res;
}

int main() {
    std::cout << "======================================================================" << std::endl;
    std::cout << " TENSTORRENT BLACKHOLE TRISC SFPU APPROXIMATE-MIN* UNIT TEST (ttsim)" << std::endl;
    std::cout << " Unit Tests: Binary AMIN*, Degree-6 Check Node, Degree-3 Check Node" << std::endl;
    std::cout << "======================================================================" << std::endl;

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

    std::vector<bfloat16> in_tile(1024, bfloat16(0.0f));

    // --------------------------------------------------------------------------
    // Populate Test 1: Binary Approximate-Min* inputs across 32 lanes
    // --------------------------------------------------------------------------
    float test_a[32];
    float test_b[32];
    float expected_binary[32];

    const struct TestCase { float a; float b; const char* desc; } cases[32] = {
        {4.0f, 4.0f, "Delta = 0.0 (Equal Inputs)"},
        {4.0f, 4.1f, "Delta = 0.1 (Region 1)"},
        {4.0f, 4.2f, "Delta = 0.2 (Region 1)"},
        {4.0f, 4.3f, "Delta = 0.3 (Region 1)"},
        {4.0f, 4.4f, "Delta = 0.4 (Region 1)"},
        {4.0f, 4.5f, "Delta = 0.5 (Boundary 1)"},
        {4.0f, 4.6f, "Delta = 0.6 (Region 2)"},
        {4.0f, 4.8f, "Delta = 0.8 (Region 2)"},
        {4.0f, 5.0f, "Delta = 1.0 (Region 2)"},
        {4.0f, 5.2f, "Delta = 1.2 (Region 2)"},
        {4.0f, 5.5f, "Delta = 1.5 (Region 2)"},
        {4.0f, 5.8f, "Delta = 1.8 (Region 2)"},
        {4.0f, 6.0f, "Delta = 2.0 (Boundary 2)"},
        {4.0f, 6.5f, "Delta = 2.5 (Region 3 / Zero Corr)"},
        {4.0f, 7.0f, "Delta = 3.0 (Zero Corr)"},
        {2.0f, 7.0f, "Delta = 5.0 (Zero Corr)"},
        {0.3f, 0.3f, "Clamping: 0.3 - 0.693 < 0 -> 0.0"},
        {0.1f, 0.3f, "Clamping: 0.1 - 0.613 < 0 -> 0.0"},
        {0.0f, 0.0f, "Zeros: a=0, b=0 -> 0.0"},
        {4.1f, 4.0f, "Inverse order: a > b (Delta=0.1)"},
        {4.5f, 4.0f, "Inverse order: a > b (Delta=0.5)"},
        {5.0f, 4.0f, "Inverse order: a > b (Delta=1.0)"},
        {6.5f, 4.0f, "Inverse order: a > b (Delta=2.5)"},
        {20.0f, 20.25f, "Large inputs: Delta=0.25"},
        {50.0f, 55.0f,  "Large inputs: Delta=5.0"},
        {0.8f, 0.9f,    "Small inputs: Delta=0.1"},
        {3.0f, 1000.0f, "Neutral Element: b = 1000.0f"},
        {5.5f, 1000.0f, "Neutral Element: b = 1000.0f"},
        {1000.0f, 2.5f, "Neutral Element: a = 1000.0f"},
        {1.2f, 1.3f,    "Fine grain Delta = 0.1"},
        {2.4f, 3.4f,    "Fine grain Delta = 1.0"},
        {7.0f, 9.5f,    "Fine grain Delta = 2.5"}
    };

    for (int l = 0; l < 32; ++l) {
        test_a[l] = cases[l].a;
        test_b[l] = cases[l].b;
        expected_binary[l] = gold_approx_min_star(test_a[l], test_b[l]);

        in_tile[dst_reg_to_element_index(0, l)] = bfloat16(test_a[l]);
        in_tile[dst_reg_to_element_index(1, l)] = bfloat16(test_b[l]);
    }

    // --------------------------------------------------------------------------
    // Populate Test 2: Degree-6 Check Nodes across 32 lanes
    // --------------------------------------------------------------------------
    float deg6_q[6][32];
    CheckDeg6Gold deg6_expected[32];

    for (int l = 0; l < 32; ++l) {
        float base_vals[6] = {2.5f, -1.8f, 3.2f, -4.0f, 1.2f, -2.9f};
        if (l % 6 == 0) base_vals[0] = 0.5f;
        if (l % 6 == 1) base_vals[1] = -0.4f;
        if (l % 6 == 2) base_vals[2] = 0.6f;
        if (l % 6 == 3) base_vals[3] = -0.3f;
        if (l % 6 == 4) base_vals[4] = 0.7f;
        if (l % 6 == 5) base_vals[5] = -0.5f;

        if (l >= 16) {
            for (int r = 0; r < 6; ++r) base_vals[r] = -base_vals[r];
        }

        for (int r = 0; r < 6; ++r) {
            deg6_q[r][l] = base_vals[r];
            in_tile[dst_reg_to_element_index(2 + r, l)] = bfloat16(deg6_q[r][l]);
        }
        float col_q[6] = {deg6_q[0][l], deg6_q[1][l], deg6_q[2][l], deg6_q[3][l], deg6_q[4][l], deg6_q[5][l]};
        deg6_expected[l] = compute_gold_deg6(col_q);
    }

    // --------------------------------------------------------------------------
    // Populate Test 3: Degree-3 Check Nodes across 32 lanes
    // --------------------------------------------------------------------------
    float deg3_q[3][32];
    CheckDeg3Gold deg3_expected[32];

    for (int l = 0; l < 32; ++l) {
        float base_vals[3] = {1.5f, -2.0f, 3.5f};
        if (l % 3 == 0) base_vals[0] = 0.8f;
        if (l % 3 == 1) base_vals[1] = -0.7f;
        if (l % 3 == 2) base_vals[2] = 0.6f;

        for (int r = 0; r < 3; ++r) {
            deg3_q[r][l] = base_vals[r];
            in_tile[dst_reg_to_element_index(8 + r, l)] = bfloat16(deg3_q[r][l]);
        }
        float col_q[3] = {deg3_q[0][l], deg3_q[1][l], deg3_q[2][l]};
        deg3_expected[l] = compute_gold_deg3(col_q);
    }

    // Send input tile to simulator device DRAM
    distributed::EnqueueWriteMeshBuffer(cq, in_dram, in_tile, false);

    // Compile and configure kernels
    auto reader = CreateKernel(
        program, "tests/unit_sfpu/reader_unit_tile.cpp", core_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );
    auto writer = CreateKernel(
        program, "tests/unit_sfpu/writer_unit_tile.cpp", core_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default}
    );
    ComputeConfig compute_config{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true};
#if defined(LDPC_USE_SFPU_LUT)
    compute_config.defines["LDPC_USE_SFPU_LUT"] = "1";
#endif
    auto compute = CreateKernel(
        program, "tests/unit_sfpu/compute_unit_sfpu_amin_star.cpp", core_range,
        compute_config
    );

    std::vector<uint32_t> r_args = { static_cast<uint32_t>(in_dram->address()), 0, tile_bytes };
    SetRuntimeArgs(program, reader, core, r_args);

    std::vector<uint32_t> w_args = { static_cast<uint32_t>(out_dram->address()), static_cast<uint32_t>(stats_dram->address()), 0, tile_bytes, stats_bytes };
    SetRuntimeArgs(program, writer, core, w_args);

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));

    std::cout << "[SIM] Dispatching TRISC SFPU Unit Test Program to ttsim Simulator..." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    distributed::EnqueueMeshWorkload(cq, workload, false);
    cq.finish();
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[SIM] Execution completed in " << elapsed_ms << " ms." << std::endl;

    // Read back output tile and stats
    std::vector<bfloat16> hw_out_tile(1024);
    distributed::ReadShard(cq, hw_out_tile, out_dram, distributed::MeshCoordinate(0, 0), true);

    std::vector<uint32_t> stats_data(stats_bytes / sizeof(uint32_t));
    distributed::ReadShard(cq, stats_data, stats_dram, distributed::MeshCoordinate(0, 0), true);

    uint32_t hw_cycles = stats_data[0];
    uint32_t sentinel = stats_data[1];

    std::cout << "Hardware Sentinel: 0x" << std::hex << sentinel << std::dec;
    if (sentinel == 0xAA55AA55) std::cout << " (PASSED - NO HANGS)" << std::endl;
    else std::cout << " (FAILED - HUNG/INVALID)" << std::endl;
    std::cout << "SFPU Elapsed Cycles: " << hw_cycles << " cycles" << std::endl;

    // ==========================================================================
    // Verification Part 1: Binary Approximate-Min*
    // ==========================================================================
    std::cout << "\n----------------------------------------------------------------------" << std::endl;
    std::cout << " TEST 1: BINARY APPROXIMATE-MIN* OPERATOR VERIFICATION" << std::endl;
    std::cout << "----------------------------------------------------------------------" << std::endl;
    std::cout << std::left << std::setw(6) << "Lane"
              << std::setw(8) << "Input A"
              << std::setw(8) << "Input B"
              << std::setw(12) << "SFPU Out"
              << std::setw(12) << "Golden"
              << std::setw(10) << "AbsDiff"
              << std::setw(8) << "Status"
              << "Description" << std::endl;

    uint32_t bin_pass_count = 0;
    float max_bin_diff = 0.0f;

    for (int l = 0; l < 32; ++l) {
        float sfpu_val = static_cast<float>(hw_out_tile[dst_reg_to_element_index(0, l)]);
        float gold_val = expected_binary[l];
        float diff = std::abs(sfpu_val - gold_val);
        if (diff > max_bin_diff) max_bin_diff = diff;

        // bfloat16 has 7-8 bits of mantissa (~0.78% relative resolution).
        // Tolerance: max of 0.03125 (for small values) or 0.5% of magnitude (for larger values).
        float tol = std::max(0.03125f, 0.005f * std::fabs(gold_val));
        bool pass = (diff <= tol);
        if (pass) bin_pass_count++;

        std::cout << std::left << std::setw(6) << l
                  << std::setw(8) << std::fixed << std::setprecision(2) << test_a[l]
                  << std::setw(8) << test_b[l]
                  << std::setw(12) << std::setprecision(4) << sfpu_val
                  << std::setw(12) << gold_val
                  << std::setw(10) << std::scientific << std::setprecision(2) << diff
                  << std::setw(8) << (pass ? "PASS" : "FAIL")
                  << cases[l].desc << std::endl;
    }
    std::cout << "Test 1 Summary: " << bin_pass_count << " / 32 Passed. Max Abs Diff = "
              << std::scientific << std::setprecision(4) << max_bin_diff << std::endl;

    // ==========================================================================
    // Verification Part 2: Degree-6 Check Node Updates
    // ==========================================================================
    std::cout << "\n----------------------------------------------------------------------" << std::endl;
    std::cout << " TEST 2: DEGREE-6 CHECK NODE SFPU OPERATOR VERIFICATION (32 Checks)" << std::endl;
    std::cout << "----------------------------------------------------------------------" << std::endl;

    uint32_t deg6_r_passes = 0, deg6_r_total = 32 * 6;
    uint32_t deg6_l_passes = 0, deg6_l_total = 32 * 6;
    float max_deg6_r_diff = 0.0f;
    float max_deg6_l_diff = 0.0f;

    for (int l = 0; l < 32; ++l) {
        for (int r = 0; r < 6; ++r) {
            float sfpu_r = static_cast<float>(hw_out_tile[dst_reg_to_element_index(2 + r, l)]);
            float gold_r = deg6_expected[l].r_out[r];
            float diff_r = std::abs(sfpu_r - gold_r);
            if (diff_r > max_deg6_r_diff) max_deg6_r_diff = diff_r;
            if (diff_r <= 0.03125f) deg6_r_passes++;

            float sfpu_l = static_cast<float>(hw_out_tile[dst_reg_to_element_index(16 + r, l)]);
            float gold_l = deg6_expected[l].l_out[r];
            float diff_l = std::abs(sfpu_l - gold_l);
            if (diff_l > max_deg6_l_diff) max_deg6_l_diff = diff_l;
            if (diff_l <= 0.03125f) deg6_l_passes++;
        }
    }
    std::cout << "Test 2 Extrinsic Messages (R_out): " << deg6_r_passes << " / " << deg6_r_total
              << " Passed. Max Diff = " << std::scientific << std::setprecision(4) << max_deg6_r_diff << std::endl;
    std::cout << "Test 2 Posterior LLRs (L_out)    : " << deg6_l_passes << " / " << deg6_l_total
              << " Passed. Max Diff = " << std::scientific << std::setprecision(4) << max_deg6_l_diff << std::endl;

    // ==========================================================================
    // Verification Part 3: Degree-3 Check Node Updates
    // ==========================================================================
    std::cout << "\n----------------------------------------------------------------------" << std::endl;
    std::cout << " TEST 3: DEGREE-3 CHECK NODE SFPU OPERATOR VERIFICATION (32 Checks)" << std::endl;
    std::cout << "----------------------------------------------------------------------" << std::endl;

    uint32_t deg3_r_passes = 0, deg3_r_total = 32 * 3;
    uint32_t deg3_l_passes = 0, deg3_l_total = 32 * 3;
    float max_deg3_r_diff = 0.0f;
    float max_deg3_l_diff = 0.0f;

    for (int l = 0; l < 32; ++l) {
        for (int r = 0; r < 3; ++r) {
            float sfpu_r = static_cast<float>(hw_out_tile[dst_reg_to_element_index(8 + r, l)]);
            float gold_r = deg3_expected[l].r_out[r];
            float diff_r = std::abs(sfpu_r - gold_r);
            if (diff_r > max_deg3_r_diff) max_deg3_r_diff = diff_r;
            if (diff_r <= 0.03125f) deg3_r_passes++;

            float sfpu_l = static_cast<float>(hw_out_tile[dst_reg_to_element_index(24 + r, l)]);
            float gold_l = deg3_expected[l].l_out[r];
            float diff_l = std::abs(sfpu_l - gold_l);
            if (diff_l > max_deg3_l_diff) max_deg3_l_diff = diff_l;
            if (diff_l <= 0.03125f) deg3_l_passes++;
        }
    }
    std::cout << "Test 3 Extrinsic Messages (R_out): " << deg3_r_passes << " / " << deg3_r_total
              << " Passed. Max Diff = " << std::scientific << std::setprecision(4) << max_deg3_r_diff << std::endl;
    std::cout << "Test 3 Posterior LLRs (L_out)    : " << deg3_l_passes << " / " << deg3_l_total
              << " Passed. Max Diff = " << std::scientific << std::setprecision(4) << max_deg3_l_diff << std::endl;

    bool all_passed = (bin_pass_count == 32) &&
                      (deg6_r_passes == deg6_r_total) && (deg6_l_passes == deg6_l_total) &&
                      (deg3_r_passes == deg3_r_total) && (deg3_l_passes == deg3_l_total) &&
                      (sentinel == 0xAA55AA55);

    std::cout << "\n======================================================================" << std::endl;
    if (all_passed) {
        std::cout << " >>> [ALL UNIT TESTS PASSED] TRISC SFPU AMIN* OPERATIONS VERIFIED! <<<" << std::endl;
    } else {
        std::cout << " >>> [UNIT TEST FAILURES DETECTED] Review individual mismatches above! <<<" << std::endl;
    }
    std::cout << "======================================================================" << std::endl;

    return all_passed ? 0 : 1;
}
