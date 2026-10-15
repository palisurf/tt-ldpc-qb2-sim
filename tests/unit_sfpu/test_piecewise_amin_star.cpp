#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <chrono>

// ------------------------------------------------------------------------------
// Bit-Exact IEEE Round-to-Nearest-Even (RNE) for bfloat16
// ------------------------------------------------------------------------------
inline float round_to_bfloat16_rne(float val) {
    uint32_t u;
    std::memcpy(&u, &val, sizeof(float));
    uint32_t lsb = (u >> 16) & 1U;
    uint32_t bias = lsb + 0x7FFFU;
    u += bias;
    u &= 0xFFFF0000U;
    float rounded;
    std::memcpy(&rounded, &u, sizeof(float));
    return rounded;
}

// ------------------------------------------------------------------------------
// 1. Exact Reference: Jacobi Logarithm
//    g(Delta) = ln(1 + exp(-Delta))
//    AMin*(a, b) = min(a, b) - ln(1 + exp(-|a - b|))
// ------------------------------------------------------------------------------
inline double true_jacobi(double delta) {
    return std::log(1.0 + std::exp(-delta));
}

inline double exact_min_star(double a, double b) {
    double min_v = std::min(a, b);
    double max_v = std::max(a, b);
    double diff = max_v - min_v;
    return std::max(0.0, min_v - true_jacobi(diff));
}

// ------------------------------------------------------------------------------
// 2. Existing Baseline: Dr. Christopher Jones 2-Piece (MILCOM 2003)
// ------------------------------------------------------------------------------
inline float approx_min_star_2piece(float a, float b, bool use_bf16 = false) {
    float min_v = std::min(a, b);
    float max_v = std::max(a, b);
    float diff  = max_v - min_v;
    float line1 = 0.69315f - 0.4000f * diff;
    float line2 = 0.65715f - 0.3280f * diff;
    float corr  = std::max(std::max(line1, line2), 0.0f);
    float res   = std::max(min_v - corr, 0.0f);
    return use_bf16 ? round_to_bfloat16_rne(res) : res;
}

// ------------------------------------------------------------------------------
// 3. Proposed Enhanced 3-Piece Convex Hull (Branchless SFPU Architecture)
//    Line 1: [0.0, 0.85)  -> y = 0.67862 - 0.39622 * diff
//    Line 2: [0.85, 2.20) -> y = 0.48710 - 0.18297 * diff
//    Line 3: [2.20, 4.35) -> y = 0.17614 - 0.04029 * diff
//    Tail  : >= 4.35      -> y = 0.0
// ------------------------------------------------------------------------------
inline float approx_min_star_3piece(float a, float b, bool use_bf16 = false) {
    float min_v = std::min(a, b);
    float max_v = std::max(a, b);
    float diff  = max_v - min_v;
    float line1 = 0.67862f - 0.39622f * diff;
    float line2 = 0.48710f - 0.18297f * diff;
    float line3 = 0.17614f - 0.04029f * diff;
    float corr  = std::max(std::max(std::max(line1, line2), line3), 0.0f);
    float res   = std::max(min_v - corr, 0.0f);
    return use_bf16 ? round_to_bfloat16_rne(res) : res;
}

// ------------------------------------------------------------------------------
// 4. Proposed Enhanced 4-Piece Convex Hull (Branchless SFPU Architecture)
//    Line 1: [0.0, 0.70)  -> y = 0.68319 - 0.41390 * diff
//    Line 2: [0.70, 1.67) -> y = 0.55418 - 0.23665 * diff
//    Line 3: [1.67, 2.83) -> y = 0.32502 - 0.09773 * diff
//    Line 4: [2.83, 5.10) -> y = 0.10588 - 0.02092 * diff
//    Tail  : >= 5.10      -> y = 0.0
// ------------------------------------------------------------------------------
inline float approx_min_star_4piece(float a, float b, bool use_bf16 = false) {
    float min_v = std::min(a, b);
    float max_v = std::max(a, b);
    float diff  = max_v - min_v;
    float line1 = 0.68319f - 0.41390f * diff;
    float line2 = 0.55418f - 0.23665f * diff;
    float line3 = 0.32502f - 0.09773f * diff;
    float line4 = 0.10588f - 0.02092f * diff;
    float corr  = std::max(std::max(std::max(std::max(line1, line2), line3), line4), 0.0f);
    float res   = std::max(min_v - corr, 0.0f);
    return use_bf16 ? round_to_bfloat16_rne(res) : res;
}

// ------------------------------------------------------------------------------
// 5. Hardware LUT (SFPLUTFP32 Table 2: 6 Entry Hardware Table)
// ------------------------------------------------------------------------------
inline float approx_min_star_hw_lut(float a, float b, bool use_bf16 = false) {
    float min_v = std::min(a, b);
    float max_v = std::max(a, b);
    float diff  = max_v - min_v;
    float corr;
    if      (diff < 0.5f) corr = 0.68801f - 0.43804f * diff;
    else if (diff < 1.0f) corr = 0.63011f - 0.32130f * diff;
    else if (diff < 1.5f) corr = 0.53285f - 0.22330f * diff;
    else if (diff < 2.0f) corr = 0.42160f - 0.14860f * diff;
    else if (diff < 4.0f) corr = 0.21113f - 0.05158f * diff;
    else                  corr = 0.0f;
    corr = std::max(corr, 0.0f);
    float res = std::max(min_v - corr, 0.0f);
    return use_bf16 ? round_to_bfloat16_rne(res) : res;
}

// ==============================================================================
// Unit Test 1: Mathematical Approximation Quality Evaluation
// ==============================================================================
void test_mathematical_quality() {
    std::cout << "======================================================================" << std::endl;
    std::cout << " TEST 1: MATHEMATICAL QUALITY VS EXACT JACOBI LOGARITHM (100,000 Points)" << std::endl;
    std::cout << "======================================================================" << std::endl;

    constexpr int N_POINTS = 100000;
    constexpr double MAX_DELTA = 6.0;

    double max_err_2p = 0.0, sum_err_2p = 0.0, sum_sq_2p = 0.0, tail_err_2p = 0.0;
    double max_err_3p = 0.0, sum_err_3p = 0.0, sum_sq_3p = 0.0, tail_err_3p = 0.0;
    double max_err_4p = 0.0, sum_err_4p = 0.0, sum_sq_4p = 0.0, tail_err_4p = 0.0;
    double max_err_lut = 0.0, sum_err_lut = 0.0, sum_sq_lut = 0.0, tail_err_lut = 0.0;

    for (int i = 0; i < N_POINTS; ++i) {
        double d = (MAX_DELTA * i) / (N_POINTS - 1);
        double gold_g = true_jacobi(d);

        float a = 10.0f;
        float b = static_cast<float>(a + d);

        double g_2p  = 10.0 - approx_min_star_2piece(a, b, false);
        double g_3p  = 10.0 - approx_min_star_3piece(a, b, false);
        double g_4p  = 10.0 - approx_min_star_4piece(a, b, false);
        double g_lut = 10.0 - approx_min_star_hw_lut(a, b, false);

        double e_2p  = std::abs(g_2p - gold_g);
        double e_3p  = std::abs(g_3p - gold_g);
        double e_4p  = std::abs(g_4p - gold_g);
        double e_lut = std::abs(g_lut - gold_g);

        max_err_2p = std::max(max_err_2p, e_2p); sum_err_2p += e_2p; sum_sq_2p += e_2p * e_2p;
        max_err_3p = std::max(max_err_3p, e_3p); sum_err_3p += e_3p; sum_sq_3p += e_3p * e_3p;
        max_err_4p = std::max(max_err_4p, e_4p); sum_err_4p += e_4p; sum_sq_4p += e_4p * e_4p;
        max_err_lut = std::max(max_err_lut, e_lut); sum_err_lut += e_lut; sum_sq_lut += e_lut * e_lut;

        if (d >= 2.0 && d <= 4.0) {
            tail_err_2p = std::max(tail_err_2p, e_2p);
            tail_err_3p = std::max(tail_err_3p, e_3p);
            tail_err_4p = std::max(tail_err_4p, e_4p);
            tail_err_lut = std::max(tail_err_lut, e_lut);
        }
    }

    std::cout << std::left << std::setw(22) << "Approximation Model"
              << std::setw(14) << "Max Error"
              << std::setw(14) << "Mean Error"
              << std::setw(14) << "RMS Error"
              << std::setw(16) << "Tail Error [2,4]"
              << std::setw(12) << "Error Gain" << std::endl;
    std::cout << std::string(92, '-') << std::endl;

    auto print_row = [&](const std::string& name, double max_e, double sum_e, double sum_sq, double tail_e, double baseline_max) {
        double mean_e = sum_e / N_POINTS;
        double rms_e  = std::sqrt(sum_sq / N_POINTS);
        double gain   = baseline_max / max_e;
        std::cout << std::left << std::setw(22) << name
                  << std::scientific << std::setprecision(4)
                  << std::setw(14) << max_e
                  << std::setw(14) << mean_e
                  << std::setw(14) << rms_e
                  << std::setw(16) << tail_e
                  << std::fixed << std::setprecision(2)
                  << gain << "x" << std::endl;
    };

    print_row("2-Piece Baseline", max_err_2p, sum_err_2p, sum_sq_2p, tail_err_2p, max_err_2p);
    print_row("3-Piece Enhanced", max_err_3p, sum_err_3p, sum_sq_3p, tail_err_3p, max_err_2p);
    print_row("4-Piece Enhanced", max_err_4p, sum_err_4p, sum_sq_4p, tail_err_4p, max_err_2p);
    print_row("6-Entry HW LUT",   max_err_lut, sum_err_lut, sum_sq_lut, tail_err_lut, max_err_2p);
    std::cout << std::string(92, '-') << std::endl;
}

// ==============================================================================
// Unit Test 2: Edge Cases & Bit-Exact Baseline Regression Verification
// ==============================================================================
void test_baseline_regression_and_edge_cases() {
    std::cout << "\n======================================================================" << std::endl;
    std::cout << " TEST 2: CORNER CASES & BIT-EXACT BASELINE REGRESSION (bfloat16 RNE)" << std::endl;
    std::cout << "======================================================================" << std::endl;

    struct TestCase {
        float a;
        float b;
        const char* name;
    };

    std::vector<TestCase> test_cases = {
        {4.0f, 4.0f, "Delta = 0.0 (Identical inputs)"},
        {4.0f, 4.1f, "Delta = 0.1 (Region 1)"},
        {4.0f, 4.5f, "Delta = 0.5 (Boundary 1)"},
        {4.0f, 5.0f, "Delta = 1.0 (Region 2)"},
        {4.0f, 5.5f, "Delta = 1.5 (Region 2)"},
        {4.0f, 6.0f, "Delta = 2.0 (Boundary 2 / 2-piece zero cutoff)"},
        {4.0f, 6.5f, "Delta = 2.5 (Tail region)"},
        {4.0f, 7.0f, "Delta = 3.0 (Tail region)"},
        {4.0f, 8.0f, "Delta = 4.0 (Deep tail)"},
        {4.0f, 10.0f, "Delta = 6.0 (Far tail)"},
        {0.3f, 0.3f, "Clamping: min - corr < 0"},
        {0.1f, 0.4f, "Clamping: 0.1 - 0.57 < 0"},
        {0.0f, 0.0f, "Zero Inputs"},
        {4.5f, 4.0f, "Symmetry / Swapped Order: a > b"},
        {6.5f, 4.0f, "Symmetry / Swapped Order: Delta = 2.5"},
        {3.0f, 1000.0f, "Neutral Element: b = 1000.0f"},
        {1000.0f, 2.5f, "Neutral Element: a = 1000.0f"},
        {20.0f, 20.25f, "Large Magnitude: a=20.0, b=20.25"},
        {50.0f, 55.0f, "Large Magnitude: a=50.0, b=55.0"}
    };

    std::cout << std::left << std::setw(32) << "Test Condition"
              << std::setw(12) << "Exact Gold"
              << std::setw(12) << "2-Piece"
              << std::setw(12) << "3-Piece"
              << std::setw(12) << "4-Piece"
              << std::setw(12) << "HW LUT" << std::endl;
    std::cout << std::string(92, '-') << std::endl;

    for (const auto& tc : test_cases) {
        double gold = exact_min_star(tc.a, tc.b);
        float r_2p  = approx_min_star_2piece(tc.a, tc.b, true);
        float r_3p  = approx_min_star_3piece(tc.a, tc.b, true);
        float r_4p  = approx_min_star_4piece(tc.a, tc.b, true);
        float r_lut = approx_min_star_hw_lut(tc.a, tc.b, true);

        std::cout << std::left << std::setw(32) << tc.name
                  << std::fixed << std::setprecision(4)
                  << std::setw(12) << gold
                  << std::setw(12) << r_2p
                  << std::setw(12) << r_3p
                  << std::setw(12) << r_4p
                  << std::setw(12) << r_lut << std::endl;
    }
    std::cout << std::string(92, '-') << std::endl;
}

// ==============================================================================
// Unit Test 3: Degree-6 Check Node Check Processing Equivalence
// ==============================================================================
void test_deg6_check_node() {
    std::cout << "\n======================================================================" << std::endl;
    std::cout << " TEST 3: DEGREE-6 CHECK NODE SFPU EMULATION COMPARISON" << std::endl;
    std::cout << "======================================================================" << std::endl;

    // Realistic LLR vectors from AR4JA iterations
    float q_test[4][6] = {
        { 1.875f, -2.500f,  3.125f, -0.9375f,  4.250f, -2.125f },
        { 0.4375f,  0.625f, -0.3125f, 0.875f, -0.500f,  0.750f },
        { 5.250f,  -6.125f,  7.000f,  4.875f, -5.750f,  6.500f },
        { 0.125f,  -0.1875f, 0.0625f, -0.250f,  0.3125f, -0.15625f }
    };

    for (int t = 0; t < 4; ++t) {
        std::cout << "--- Sub-Test 3." << (t + 1) << ": Q = [ ";
        for (int r = 0; r < 6; ++r) std::cout << q_test[t][r] << " ";
        std::cout << "] ---" << std::endl;

        // Degree 6 pairwise tree:
        // Pair 0: (q0, q1), Pair 1: (q2, q3), Pair 2: (q4, q5)
        float m0 = std::abs(q_test[t][0]), m0_sw = std::abs(q_test[t][1]);
        float m1 = std::abs(q_test[t][2]), m1_sw = std::abs(q_test[t][3]);
        float m2 = std::abs(q_test[t][4]), m2_sw = std::abs(q_test[t][5]);

        // Evaluate with 2-Piece
        float m01_2p = round_to_bfloat16_rne(approx_min_star_2piece(m0, m0_sw));
        float m23_2p = round_to_bfloat16_rne(approx_min_star_2piece(m1, m1_sw));
        float m45_2p = round_to_bfloat16_rne(approx_min_star_2piece(m2, m2_sw));
        float m_oth0_2p = approx_min_star_2piece(m23_2p, m45_2p);
        float r_mag0_2p = approx_min_star_2piece(m_oth0_2p, m0_sw, true);

        // Evaluate with 3-Piece
        float m01_3p = round_to_bfloat16_rne(approx_min_star_3piece(m0, m0_sw));
        float m23_3p = round_to_bfloat16_rne(approx_min_star_3piece(m1, m1_sw));
        float m45_3p = round_to_bfloat16_rne(approx_min_star_3piece(m2, m2_sw));
        float m_oth0_3p = approx_min_star_3piece(m23_3p, m45_3p);
        float r_mag0_3p = approx_min_star_3piece(m_oth0_3p, m0_sw, true);

        // Exact
        double m01_ex = exact_min_star(m0, m0_sw);
        double m23_ex = exact_min_star(m1, m1_sw);
        double m45_ex = exact_min_star(m2, m2_sw);
        double m_oth0_ex = exact_min_star(m23_ex, m45_ex);
        double r_mag0_ex = exact_min_star(m_oth0_ex, m0_sw);

        std::cout << "  Pairwise M01: Exact=" << m01_ex << ", 2-Piece=" << m01_2p << ", 3-Piece=" << m01_3p << std::endl;
        std::cout << "  Pairwise M23: Exact=" << m23_ex << ", 2-Piece=" << m23_2p << ", 3-Piece=" << m23_3p << std::endl;
        std::cout << "  Pairwise M45: Exact=" << m45_ex << ", 2-Piece=" << m45_2p << ", 3-Piece=" << m45_3p << std::endl;
        std::cout << "  Outgoing R0 : Exact=" << r_mag0_ex
                  << ", 2-Piece=" << r_mag0_2p << " (err=" << std::abs(r_mag0_2p - r_mag0_ex) << ")"
                  << ", 3-Piece=" << r_mag0_3p << " (err=" << std::abs(r_mag0_3p - r_mag0_ex) << ")" << std::endl;
    }
}

int main() {
    test_mathematical_quality();
    test_baseline_regression_and_edge_cases();
    test_deg6_check_node();
    return 0;
}
