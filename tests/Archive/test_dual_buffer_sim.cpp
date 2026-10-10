#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>

#include "../../engines/scalar_5w/ldpc_decoder_core.h"
#include "../../engines/scalar_5w/Archive/ldpc_decoder_core_dual_buffer.h"
#include "../../engines/scalar_5w/Archive/ldpc_decoder_core_fixed_point.h"

struct ChinnMatrix {
    uint32_t N = 0;
    uint32_t M = 0;
    uint32_t max_var_deg = 0;
    uint32_t max_check_deg = 0;
    std::vector<uint16_t> flattened_indices;
};

ChinnMatrix parse_matrix_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) throw std::runtime_error("Could not open matrix: " + filepath);

    struct Edge { uint32_t r, c; float val; };
    std::vector<Edge> raw_lines;
    raw_lines.reserve(100000);

    std::string line;
    while (std::getline(file, line)) {
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos || line[start] == '#') continue;
        std::stringstream ss(line);
        uint32_t r = 0, c = 0;
        float val = 0.0f;
        if (ss >> r >> c >> val) {
            raw_lines.push_back({r, c, val});
        }
    }
    if (raw_lines.empty()) throw std::runtime_error("Empty matrix file");

    Edge dim_line = raw_lines.back();
    ChinnMatrix chinn;
    chinn.M = dim_line.r;
    chinn.N = dim_line.c;

    std::vector<std::vector<uint16_t>> check_nodes(chinn.M);
    std::vector<uint32_t> var_deg(chinn.N, 0);

    for (size_t i = 0; i + 1 < raw_lines.size(); i++) {
        const auto& edge = raw_lines[i];
        if (std::abs(edge.val) > 1e-6f) {
            uint32_t m_idx = edge.r - 1;
            uint32_t vn_idx = edge.c - 1;
            check_nodes[m_idx].push_back(static_cast<uint16_t>(vn_idx));
            var_deg[vn_idx]++;
        }
    }

    chinn.max_check_deg = 0;
    for (uint32_t m = 0; m < chinn.M; m++) {
        if (check_nodes[m].size() > chinn.max_check_deg) {
            chinn.max_check_deg = static_cast<uint32_t>(check_nodes[m].size());
        }
    }

    uint32_t stride = chinn.max_check_deg + 1;
    chinn.flattened_indices.assign(chinn.M * stride, 0xFFFF);
    for (uint32_t m = 0; m < chinn.M; m++) {
        uint32_t deg = static_cast<uint32_t>(check_nodes[m].size());
        chinn.flattened_indices[m * stride] = static_cast<uint16_t>(deg);
        for (size_t d = 0; d < deg && d < chinn.max_check_deg; d++) {
            chinn.flattened_indices[m * stride + 1 + d] = check_nodes[m][d];
        }
    }
    return chinn;
}

int main(int argc, char** argv) {
    std::string matrix_path = "matrices/AR4JA_r45_4c_128c_r12.chinn.out";
    if (argc > 1) matrix_path = argv[1];

    std::cout << "==========================================================" << std::endl;
    std::cout << "  Dual-Buffer Bfloat8 vs Single-Buffer Bfloat16 Benchmark " << std::endl;
    std::cout << "==========================================================" << std::endl;

    ChinnMatrix chinn = parse_matrix_file(matrix_path);
    std::cout << "Matrix: N=" << chinn.N << ", M=" << chinn.M 
              << ", max_check_deg=" << chinn.max_check_deg << std::endl;

    uint32_t P_punctured = 512;
    uint32_t max_iter = 16;
    uint32_t test_blocks = 3000; // 1500 pairs of codewords

    std::vector<double> snr_list = {1.6, 1.8, 2.0, 2.2, 2.4};

    // Allocate memory buffers
    std::vector<uint32_t> bf16_channel_llrs(chinn.N, 0);
    std::vector<uint32_t> bf16_r_msg(chinn.M * MAX_DEG_STRIDE, 0);

    // BF8 buffers
    std::vector<uint16_t> fp8_channel_llrs(chinn.N, 0);
    std::vector<uint32_t> prop_post_llrs(chinn.N, 0);
    std::vector<uint16_t> prop_r_msg(chinn.M * DUAL_BUF_MAX_DEG_STRIDE, 0);

    // Fixed-Point buffers
    std::vector<uint16_t> fixed_channel_llrs(chinn.N, 0);
    std::vector<uint16_t> fixed_post_all_q(chinn.N, 0);
    std::vector<uint32_t> fixed_post_hybrid(chinn.N, 0);
    std::vector<uint16_t> fixed_r_msg(chinn.M * DUAL_BUF_MAX_DEG_STRIDE, 0);

    uint32_t s0 = 0x12345678, s1 = 0x9ABCDEF0, s2 = 0x0FEDCBA9, s3 = 0x87654321;

    for (double eb_n0_db : snr_list) {
        double rate = static_cast<double>(chinn.N - chinn.M) / static_cast<double>(chinn.N - P_punctured);
        double snr_lin = std::pow(10.0, eb_n0_db / 10.0);
        float mu_llr = static_cast<float>(4.0 * rate * snr_lin);
        float sigma_llr = static_cast<float>(std::sqrt(8.0 * rate * snr_lin));
        float clip_val = mu_llr + 2.0f * sigma_llr; // ~2-sigma bound

        std::cout << "\n------------------------------------------------------------------------------------------------------" << std::endl;
        std::cout << "Eb/N0 = " << std::fixed << std::setprecision(2) << eb_n0_db 
                  << " dB (mu=" << mu_llr << ", sigma=" << sigma_llr << ", clip=" << clip_val << ", blocks=" << test_blocks << ")" << std::endl;
        std::cout << "------------------------------------------------------------------------------------------------------" << std::endl;

        // 1. Baseline: Single-buffer Bfloat16
        uint32_t stats_bf16[4] = {0};
        auto t0 = std::chrono::high_resolution_clock::now();
        run_ldpc_worker(
            0, test_blocks, chinn.N, chinn.M, chinn.max_check_deg, P_punctured, max_iter,
            mu_llr, sigma_llr, 0.0f, 0.0f, 0.0f,
            s0, s1, s2, s3,
            chinn.flattened_indices.data(),
            bf16_channel_llrs.data(),
            bf16_r_msg.data(),
            stats_bf16
        );
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms_bf16 = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // 2. BF8 Proposal: Channel BF8, Post BF16, Edges BF8 (Unclipped)
        uint32_t stats_bf8[4] = {0};
        auto t2 = std::chrono::high_resolution_clock::now();
        decode_ldpc_dual_buffer_proposal(
            test_blocks, 0, chinn.N, chinn.M, chinn.max_check_deg, P_punctured, max_iter,
            mu_llr, sigma_llr, 0.0f, 0.0f, 0.0f,
            s0, s1, s2, s3,
            chinn.flattened_indices.data(),
            fp8_channel_llrs.data(),
            prop_post_llrs.data(),
            prop_r_msg.data(),
            stats_bf8
        );
        auto t3 = std::chrono::high_resolution_clock::now();
        double ms_bf8 = std::chrono::duration<double, std::milli>(t3 - t2).count();

        // 3. Fixed-Point Hybrid: Channel Q4.3, Post BF16, Edges Q3.4 (Unclipped)
        uint32_t stats_fx_hyb_unclip[4] = {0};
        auto t4 = std::chrono::high_resolution_clock::now();
        decode_ldpc_fixed_point_hybrid(
            test_blocks, 0, chinn.N, chinn.M, chinn.max_check_deg, P_punctured, max_iter,
            mu_llr, sigma_llr, 0.0f, 0.0f, 0.0f,
            s0, s1, s2, s3,
            chinn.flattened_indices.data(),
            fixed_channel_llrs.data(),
            fixed_post_hybrid.data(),
            fixed_r_msg.data(),
            stats_fx_hyb_unclip
        );
        auto t5 = std::chrono::high_resolution_clock::now();
        double ms_fx_hyb_unclip = std::chrono::duration<double, std::milli>(t5 - t4).count();

        // 4. Fixed-Point Hybrid: Channel Q4.3, Post BF16, Edges Q3.4 (Clipped L_ch_max = mu + 2*sigma)
        uint32_t stats_fx_hyb_clip[4] = {0};
        auto t6 = std::chrono::high_resolution_clock::now();
        decode_ldpc_fixed_point_hybrid(
            test_blocks, 0, chinn.N, chinn.M, chinn.max_check_deg, P_punctured, max_iter,
            mu_llr, sigma_llr, clip_val, 0.0f, 0.0f,
            s0, s1, s2, s3,
            chinn.flattened_indices.data(),
            fixed_channel_llrs.data(),
            fixed_post_hybrid.data(),
            fixed_r_msg.data(),
            stats_fx_hyb_clip
        );
        auto t7 = std::chrono::high_resolution_clock::now();
        double ms_fx_hyb_clip = std::chrono::duration<double, std::milli>(t7 - t6).count();

        // 5. Fixed-Point All-Q: Channel Q4.3, Post Q4.3, Edges Q3.4 (Unclipped)
        uint32_t stats_fx_all_unclip[4] = {0};
        auto t8 = std::chrono::high_resolution_clock::now();
        decode_ldpc_fixed_point_all_q43_q34(
            test_blocks, 0, chinn.N, chinn.M, chinn.max_check_deg, P_punctured, max_iter,
            mu_llr, sigma_llr, 0.0f, 0.0f, 0.0f,
            s0, s1, s2, s3,
            chinn.flattened_indices.data(),
            fixed_channel_llrs.data(),
            fixed_post_all_q.data(),
            fixed_r_msg.data(),
            stats_fx_all_unclip
        );
        auto t9 = std::chrono::high_resolution_clock::now();
        double ms_fx_all_unclip = std::chrono::duration<double, std::milli>(t9 - t8).count();

        // 6. Fixed-Point All-Q: Channel Q4.3, Post Q4.3, Edges Q3.4 (Clipped L_ch_max = mu + 2*sigma)
        uint32_t stats_fx_all_clip[4] = {0};
        auto t10 = std::chrono::high_resolution_clock::now();
        decode_ldpc_fixed_point_all_q43_q34(
            test_blocks, 0, chinn.N, chinn.M, chinn.max_check_deg, P_punctured, max_iter,
            mu_llr, sigma_llr, clip_val, 0.0f, 0.0f,
            s0, s1, s2, s3,
            chinn.flattened_indices.data(),
            fixed_channel_llrs.data(),
            fixed_post_all_q.data(),
            fixed_r_msg.data(),
            stats_fx_all_clip
        );
        auto t11 = std::chrono::high_resolution_clock::now();
        double ms_fx_all_clip = std::chrono::duration<double, std::milli>(t11 - t10).count();

        std::cout << std::left << std::setw(38) << "Decoder Configuration"
                  << std::setw(12) << "Bit Errs"
                  << std::setw(12) << "Frame Errs"
                  << std::setw(14) << "FER"
                  << std::setw(12) << "Avg Iters"
                  << std::setw(12) << "Time (ms)" << std::endl;
        std::cout << std::string(100, '-') << std::endl;

        std::cout << std::left << std::setw(38) << "Baseline Bfloat16 (Single-Buf)"
                  << std::setw(12) << stats_bf16[0]
                  << std::setw(12) << stats_bf16[1]
                  << std::setw(14) << std::scientific << std::setprecision(2) << (double)stats_bf16[1] / test_blocks
                  << std::setw(12) << std::fixed << std::setprecision(1) << (double)stats_bf16[2]
                  << std::setw(12) << std::fixed << std::setprecision(1) << ms_bf16 << std::endl;

        std::cout << std::left << std::setw(38) << "BF8 Proposal (Ch:BF8, Post:BF16, E:BF8)"
                  << std::setw(12) << stats_bf8[0]
                  << std::setw(12) << stats_bf8[1]
                  << std::setw(14) << std::scientific << std::setprecision(2) << (double)stats_bf8[1] / test_blocks
                  << std::setw(12) << std::fixed << std::setprecision(1) << (double)stats_bf8[2]
                  << std::setw(12) << std::fixed << std::setprecision(1) << ms_bf8 << std::endl;

        std::cout << std::left << std::setw(38) << "Fixed-Pt Hybrid (Ch:Q4.3, Post:BF16, E:Q3.4)"
                  << std::setw(12) << stats_fx_hyb_unclip[0]
                  << std::setw(12) << stats_fx_hyb_unclip[1]
                  << std::setw(14) << std::scientific << std::setprecision(2) << (double)stats_fx_hyb_unclip[1] / test_blocks
                  << std::setw(12) << std::fixed << std::setprecision(1) << (double)stats_fx_hyb_unclip[2]
                  << std::setw(12) << std::fixed << std::setprecision(1) << ms_fx_hyb_unclip << std::endl;

        std::cout << std::left << std::setw(38) << "Fixed-Pt Hybrid Clipped (2-sigma)"
                  << std::setw(12) << stats_fx_hyb_clip[0]
                  << std::setw(12) << stats_fx_hyb_clip[1]
                  << std::setw(14) << std::scientific << std::setprecision(2) << (double)stats_fx_hyb_clip[1] / test_blocks
                  << std::setw(12) << std::fixed << std::setprecision(1) << (double)stats_fx_hyb_clip[2]
                  << std::setw(12) << std::fixed << std::setprecision(1) << ms_fx_hyb_clip << std::endl;

        std::cout << std::left << std::setw(38) << "Fixed-Pt All-Q (Ch:Q4.3, Post:Q4.3, E:Q3.4)"
                  << std::setw(12) << stats_fx_all_unclip[0]
                  << std::setw(12) << stats_fx_all_unclip[1]
                  << std::setw(14) << std::scientific << std::setprecision(2) << (double)stats_fx_all_unclip[1] / test_blocks
                  << std::setw(12) << std::fixed << std::setprecision(1) << (double)stats_fx_all_unclip[2]
                  << std::setw(12) << std::fixed << std::setprecision(1) << ms_fx_all_unclip << std::endl;

        std::cout << std::left << std::setw(38) << "Fixed-Pt All-Q Clipped (2-sigma)"
                  << std::setw(12) << stats_fx_all_clip[0]
                  << std::setw(12) << stats_fx_all_clip[1]
                  << std::setw(14) << std::scientific << std::setprecision(2) << (double)stats_fx_all_clip[1] / test_blocks
                  << std::setw(12) << std::fixed << std::setprecision(1) << (double)stats_fx_all_clip[2]
                  << std::setw(12) << std::fixed << std::setprecision(1) << ms_fx_all_clip << std::endl;
    }

    std::cout << "\n==========================================================" << std::endl;
    std::cout << "                 Benchmark Complete                       " << std::endl;
    std::cout << "==========================================================" << std::endl;
    return 0;
}
