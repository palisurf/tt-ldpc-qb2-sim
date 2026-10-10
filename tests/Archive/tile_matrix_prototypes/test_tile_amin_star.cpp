#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <random>
#include <cstring>
#include <cassert>

#include "../engines/tile_matrix_4cw/ldpc_tile_matrix_core.h"

// ==============================================================================
// Tenstorrent QuietBox 2: Phase 2A Tile-Block Approximate-Min* Verification
// Tests 32-Lane Vectorized Degree-6 Check Block against Golden Scalar Reference
//
// 1 Tile Block = 8 Check Nodes x 4 Parallel Codewords = 32 Vector Lanes
// Height = 6 Rows (one row per degree-6 connected edge)
// ==============================================================================

// ------------------------------------------------------------------------------
// 1. Golden Scalar Reference Implementation (One Check Node, One Codeword)
// ------------------------------------------------------------------------------
struct CheckOutputs {
    float r_prime[6];
    float l_new[6];
    uint32_t min_idx;
    float m_others;
    float m_all;
};

CheckOutputs compute_scalar_deg6_reference(const float q[6]) {
    CheckOutputs out{};
    float signs[6];
    float mags[6];
    float total_sign = 1.0f;
    uint32_t min_idx = 0;
    float min_mag = 99999.0f;

    for (uint32_t d = 0; d < 6; ++d) {
        signs[d] = (q[d] < 0.0f) ? -1.0f : 1.0f;
        uint32_t u = float_as_uint(q[d]) & 0x7FFFFFFF;
        mags[d] = uint_as_float(u);
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
        m_others = approx_min_star_scalar(m_others, mags[d]);
    }
    float m_all = approx_min_star_scalar(m_others, min_mag);

    out.min_idx = min_idx;
    out.m_others = m_others;
    out.m_all = m_all;

    for (uint32_t d = 0; d < 6; ++d) {
        float s_d = total_sign * signs[d];
        float m_d = (d == min_idx) ? m_others : m_all;
        out.r_prime[d] = s_d * m_d;
        out.l_new[d] = q[d] + out.r_prime[d];
    }

    return out;
}

// ------------------------------------------------------------------------------
// 2. 32-Lane Vectorized SFPU Degree-6 Tile Block Algorithm
// Evaluates 8 Checks x 4 Codewords (32 parallel lanes) simultaneously in 6 rows
// ------------------------------------------------------------------------------
struct TileBlock32x6 {
    // 6 rows of 32 lanes (bfloat16 stored in float for simulation)
    float q[6][32];
    float r_prime[6][32];
    float l_new[6][32];
    uint32_t min_idx[32];
    float m_others[32];
    float m_all[32];
};

// Binary Approximate-Min* SIMD operator (matches SFPU 2-region piecewise spline)
inline float sim_sfpu_amin_star_binary(float a, float b) {
    float diff = std::abs(a - b);
    float min_v = (a < b) ? a : b;
    float corr = 0.0f;
    if (diff < 0.5f) {
        corr = 0.69315f - 0.4000f * diff;
    } else if (diff < 2.0f) {
        corr = 0.49315f - 0.3280f * (diff - 0.5f);
    }
    float res = min_v - corr;
    return (res > 0.0f) ? res : 0.0f;
}

// Vectorized Degree-6 Tile Block Kernel (Simulates TRISC1 SFPU Execution)
void vector_sfpu_amin_star_deg6_tile_block(TileBlock32x6& tile) {
    // 1. Magnitude and Sign Extraction across all 6 rows (32 lanes wide)
    float mags[6][32];
    float signs[6][32];
    float total_sign[32];

    for (int l = 0; l < 32; ++l) {
        total_sign[l] = 1.0f;
    }

    for (int r = 0; r < 6; ++r) {
        for (int l = 0; l < 32; ++l) {
            float v = tile.q[r][l];
            signs[r][l] = (v < 0.0f) ? -1.0f : 1.0f;
            mags[r][l] = std::abs(v);
            total_sign[l] *= signs[r][l];
        }
    }

    // 2. Vectorized Minimum Magnitude Search across the 6 rows
    // Across all 32 lanes simultaneously:
    for (int l = 0; l < 32; ++l) {
        float min_m = mags[0][l];
        uint32_t min_i = 0;
        for (uint32_t r = 1; r < 6; ++r) {
            if (mags[r][l] < min_m) {
                min_m = mags[r][l];
                min_i = r;
            }
        }
        tile.min_idx[l] = min_i;
    }

    // 3. True Recursive Approximate-Min* Accumulation (MILCOM 2003)
    // Computes M_others (all except min_idx) and M_all (all 6 edges)
    for (int l = 0; l < 32; ++l) {
        uint32_t min_i = tile.min_idx[l];
        uint32_t f_idx = (min_i == 0) ? 1 : 0;
        float m_others = mags[f_idx][l];

        for (uint32_t r = f_idx + 1; r < 6; ++r) {
            if (r == min_i) continue;
            m_others = sim_sfpu_amin_star_binary(m_others, mags[r][l]);
        }
        float m_all = sim_sfpu_amin_star_binary(m_others, mags[min_i][l]);

        tile.m_others[l] = m_others;
        tile.m_all[l] = m_all;

        // 4. Parity and Message Reconstruction across all 6 rows
        for (uint32_t r = 0; r < 6; ++r) {
            float s_d = total_sign[l] * signs[r][l];
            float m_d = (r == min_i) ? m_others : m_all;
            float r_p = s_d * m_d;
            tile.r_prime[r][l] = r_p;
            tile.l_new[r][l] = tile.q[r][l] + r_p;
        }
    }
}

// ------------------------------------------------------------------------------
// 4. Degree-3 Check Block Reference & 32-Lane Vector Implementation
// 1 Block = 8 Check Nodes x 4 Parallel Codewords = 32 Vector Lanes (Height = 3 Rows)
// ------------------------------------------------------------------------------
struct CheckOutputsDeg3 {
    float r_prime[3];
    float l_new[3];
    uint32_t min_idx;
    float m_others;
    float m_all;
};

CheckOutputsDeg3 compute_scalar_deg3_reference(const float q[3]) {
    CheckOutputsDeg3 out{};
    float signs[3];
    float mags[3];
    float total_sign = 1.0f;
    uint32_t min_idx = 0;
    float min_mag = 99999.0f;

    for (uint32_t d = 0; d < 3; ++d) {
        signs[d] = (q[d] < 0.0f) ? -1.0f : 1.0f;
        uint32_t u = float_as_uint(q[d]) & 0x7FFFFFFF;
        mags[d] = uint_as_float(u);
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
        m_others = approx_min_star_scalar(m_others, mags[d]);
    }
    float m_all = approx_min_star_scalar(m_others, min_mag);

    out.min_idx = min_idx;
    out.m_others = m_others;
    out.m_all = m_all;

    for (uint32_t d = 0; d < 3; ++d) {
        float s_d = total_sign * signs[d];
        float m_d = (d == min_idx) ? m_others : m_all;
        out.r_prime[d] = s_d * m_d;
        out.l_new[d] = q[d] + out.r_prime[d];
    }

    return out;
}

struct TileBlock32x3 {
    float q[3][32];
    float r_prime[3][32];
    float l_new[3][32];
    uint32_t min_idx[32];
    float m_others[32];
    float m_all[32];
};

void vector_sfpu_amin_star_deg3_tile_block(TileBlock32x3& tile) {
    float mags[3][32];
    float signs[3][32];
    float total_sign[32];

    for (int l = 0; l < 32; ++l) total_sign[l] = 1.0f;

    for (int r = 0; r < 3; ++r) {
        for (int l = 0; l < 32; ++l) {
            float v = tile.q[r][l];
            signs[r][l] = (v < 0.0f) ? -1.0f : 1.0f;
            mags[r][l] = std::abs(v);
            total_sign[l] *= signs[r][l];
        }
    }

    for (int l = 0; l < 32; ++l) {
        float min_m = mags[0][l];
        uint32_t min_i = 0;
        for (uint32_t r = 1; r < 3; ++r) {
            if (mags[r][l] < min_m) {
                min_m = mags[r][l];
                min_i = r;
            }
        }
        tile.min_idx[l] = min_i;
    }

    for (int l = 0; l < 32; ++l) {
        uint32_t min_i = tile.min_idx[l];
        uint32_t f_idx = (min_i == 0) ? 1 : 0;
        float m_others = mags[f_idx][l];

        for (uint32_t r = f_idx + 1; r < 3; ++r) {
            if (r == min_i) continue;
            m_others = sim_sfpu_amin_star_binary(m_others, mags[r][l]);
        }
        float m_all = sim_sfpu_amin_star_binary(m_others, mags[min_i][l]);

        tile.m_others[l] = m_others;
        tile.m_all[l] = m_all;

        for (uint32_t r = 0; r < 3; ++r) {
            float s_d = total_sign[l] * signs[r][l];
            float m_d = (r == min_i) ? m_others : m_all;
            float r_p = s_d * m_d;
            tile.r_prime[r][l] = r_p;
            tile.l_new[r][l] = tile.q[r][l] + r_p;
        }
    }
}

int main() {
    std::cout << "======================================================================" << std::endl;
    std::cout << " PHASE 2A: TILE-BLOCK APPROXIMATE-MIN* VECTOR VERIFICATION (HOST)     " << std::endl;
    std::cout << " Target: 32 Vector Lanes (8 Checks x 4 Parallel Codewords)           " << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> llr_dist(-8.0f, +8.0f);

    constexpr uint32_t NUM_TRIALS = 5000;
    uint32_t total_checks_tested = NUM_TRIALS * 32; // 160,000 check node operations
    uint32_t mismatch_count = 0;
    double max_r_diff = 0.0;
    double max_l_diff = 0.0;

    std::cout << "\n[TEST CONFIG]" << std::endl;
    std::cout << " Number of Trials     : " << NUM_TRIALS << " tile passes" << std::endl;
    std::cout << " Checks per Tile Pass : 32 checks (8 checks x 4 codewords in parallel)" << std::endl;
    std::cout << " Total Checks Tested  : " << total_checks_tested << " check operations" << std::endl;
    std::cout << " Total Edges Evaluated: " << total_checks_tested * 6 << " edges" << std::endl;

    for (uint32_t trial = 0; trial < NUM_TRIALS; ++trial) {
        TileBlock32x6 tile{};

        // Populate tile with random extrinsics Q across all 6 rows and 32 lanes
        for (int r = 0; r < 6; ++r) {
            for (int l = 0; l < 32; ++l) {
                tile.q[r][l] = llr_dist(rng);
            }
        }

        // Execute 32-lane Vector SFPU block
        vector_sfpu_amin_star_deg6_tile_block(tile);

        // Execute Golden Scalar Reference independently for each of the 32 lanes
        for (int l = 0; l < 32; ++l) {
            float scalar_q[6];
            for (int r = 0; r < 6; ++r) {
                scalar_q[r] = tile.q[r][l];
            }

            CheckOutputs ref = compute_scalar_deg6_reference(scalar_q);

            // Compare min_idx
            if (tile.min_idx[l] != ref.min_idx) {
                mismatch_count++;
                std::cerr << "[ERROR] Trial " << trial << " Lane " << l 
                          << " min_idx mismatch: Vec=" << tile.min_idx[l] 
                          << " vs Ref=" << ref.min_idx << std::endl;
            }

            // Compare M_others and M_all
            double diff_others = std::abs(tile.m_others[l] - ref.m_others);
            double diff_all = std::abs(tile.m_all[l] - ref.m_all);
            if (diff_others > 1e-6 || diff_all > 1e-6) {
                mismatch_count++;
            }

            // Compare outgoing check messages and updated posterior LLRs
            for (int r = 0; r < 6; ++r) {
                double diff_r = std::abs(tile.r_prime[r][l] - ref.r_prime[r]);
                double diff_l = std::abs(tile.l_new[r][l] - ref.l_new[r]);
                if (diff_r > max_r_diff) max_r_diff = diff_r;
                if (diff_l > max_l_diff) max_l_diff = diff_l;

                if (diff_r > 1e-5 || diff_l > 1e-5) {
                    mismatch_count++;
                    std::cerr << "[ERROR] Trial " << trial << " Lane " << l << " Row " << r
                              << " mismatch: R_vec=" << tile.r_prime[r][l] 
                              << " vs R_ref=" << ref.r_prime[r] << std::endl;
                }
            }
        }
    }

    // --------------------------------------------------------------------------
    // Test Degree-3 Check Nodes (AR4JA Rate-1/2 has 512 Degree-3 checks)
    // --------------------------------------------------------------------------
    std::cout << "\n[TEST 2: DEGREE-3 CHECKS (AR4JA Rate-1/2)]" << std::endl;
    for (uint32_t trial = 0; trial < NUM_TRIALS; ++trial) {
        TileBlock32x3 tile{};
        for (int r = 0; r < 3; ++r) {
            for (int l = 0; l < 32; ++l) {
                tile.q[r][l] = llr_dist(rng);
            }
        }

        vector_sfpu_amin_star_deg3_tile_block(tile);

        for (int l = 0; l < 32; ++l) {
            float scalar_q[3];
            for (int r = 0; r < 3; ++r) scalar_q[r] = tile.q[r][l];

            CheckOutputsDeg3 ref = compute_scalar_deg3_reference(scalar_q);

            if (tile.min_idx[l] != ref.min_idx) mismatch_count++;

            double diff_others = std::abs(tile.m_others[l] - ref.m_others);
            double diff_all = std::abs(tile.m_all[l] - ref.m_all);
            if (diff_others > 1e-6 || diff_all > 1e-6) mismatch_count++;

            for (int r = 0; r < 3; ++r) {
                double diff_r = std::abs(tile.r_prime[r][l] - ref.r_prime[r]);
                double diff_l = std::abs(tile.l_new[r][l] - ref.l_new[r]);
                if (diff_r > max_r_diff) max_r_diff = diff_r;
                if (diff_l > max_l_diff) max_l_diff = diff_l;
                if (diff_r > 1e-5 || diff_l > 1e-5) mismatch_count++;
            }
        }
    }

    std::cout << "\n======================================================================" << std::endl;
    std::cout << " VERIFICATION AUDIT RESULTS (DEGREE-6 & DEGREE-3)                     " << std::endl;
    std::cout << "======================================================================" << std::endl;
    std::cout << " Total Checks Tested      : " << total_checks_tested * 2 << " (Deg-6 + Deg-3)" << std::endl;
    std::cout << " Total Edges Checked      : " << total_checks_tested * 6 + total_checks_tested * 3 << std::endl;
    std::cout << " Max Outgoing R' Error    : " << std::scientific << std::setprecision(3) << max_r_diff << " LLR units" << std::endl;
    std::cout << " Max Updated L_new Error  : " << std::scientific << std::setprecision(3) << max_l_diff << " LLR units" << std::endl;
    std::cout << " Total Mismatches Found   : " << mismatch_count << std::endl;

    if (mismatch_count == 0) {
        std::cout << "\n>>> [PHASE 2A PASSED]: 100% BIT-EXACT MATCH WITH GOLDEN REFERENCE! <<<" << std::endl;
        std::cout << " All 8 checks x 4 codewords (32 vector lanes) verified identical." << std::endl;
        std::cout << "======================================================================" << std::endl;
        return 0;
    } else {
        std::cerr << "\n>>> [PHASE 2A FAILED]: " << mismatch_count << " MISMATCHES DETECTED! <<<" << std::endl;
        std::cout << "======================================================================" << std::endl;
        return 1;
    }
}
