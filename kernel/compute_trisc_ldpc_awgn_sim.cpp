#include "api/compute/eltwise_binary.h"
#include "api/compute/cb_api.h"
#include "ldpc_decoder_core.h"

void kernel_main() {
    // Runtime Arguments:
    // 0: num_codewords_trisc0
    // 1: num_codewords_trisc1
    // 2: num_codewords_trisc2
    // 3: N_var_nodes
    // 4: M_check_nodes
    // 5: P_punctured
    // 6: max_check_deg
    // 7: mu_bits
    // 8: sigma_bits
    // 9: s0
    // 10: s1
    // 11: s2
    // 12: s3
    // 13: max_iter
    // 14: l_max_bits
    // 15: r_max_bits
    uint32_t num_cw_trisc0 = get_arg_val<uint32_t>(0);
    uint32_t num_cw_trisc1 = get_arg_val<uint32_t>(1);
    uint32_t num_cw_trisc2 = get_arg_val<uint32_t>(2);
    uint32_t N_var_nodes   = get_arg_val<uint32_t>(3);
    uint32_t M_check_nodes = get_arg_val<uint32_t>(4);
    uint32_t P_punctured   = get_arg_val<uint32_t>(5);
    uint32_t max_check_deg = get_arg_val<uint32_t>(6);
    uint32_t mu_bits       = get_arg_val<uint32_t>(7);
    uint32_t sigma_bits    = get_arg_val<uint32_t>(8);
    uint32_t s0            = get_arg_val<uint32_t>(9);
    uint32_t s1            = get_arg_val<uint32_t>(10);
    uint32_t s2            = get_arg_val<uint32_t>(11);
    uint32_t s3            = get_arg_val<uint32_t>(12);
    uint32_t max_iter      = get_arg_val<uint32_t>(13);
    uint32_t l_max_bits    = get_arg_val<uint32_t>(14);
    uint32_t r_max_bits    = get_arg_val<uint32_t>(15);

    float mu_llr = uint_as_float(mu_bits);
    float sigma_llr = uint_as_float(sigma_bits);
    float l_max = uint_as_float(l_max_bits);
    float r_max = uint_as_float(r_max_bits);

    uint32_t cb0_base = get_tile_address(tt::CBIndex::c_0, 0);
    uint32_t cb1_base = get_tile_address(tt::CBIndex::c_1, 0);
    uint32_t cb2_base = get_tile_address(tt::CBIndex::c_2, 0);
    uint32_t cb16_base = get_tile_address(tt::CBIndex::c_16, 0);

    // =========================================================================
    // TRISC0: UNPACK Thread (Worker 2)
    // =========================================================================
    UNPACK({
        constexpr uint32_t cb_h = tt::CBIndex::c_2;
        // Wait for H matrix pushed by BRISC
        cb_wait_front(cb_h, 1);

        // Notify MATH and PACK that H is ready in L1
        mailbox_write(ckernel::ThreadId::MathThreadId, 1);
        mailbox_write(ckernel::ThreadId::PackThreadId, 1);

        // Execute Worker 2 (TRISC0)
        uint32_t* channel_llrs = reinterpret_cast<uint32_t*>(cb0_base + 2 * LLR_WORKER_STRIDE_BYTES);
        uint32_t* r_msg = reinterpret_cast<uint32_t*>(cb1_base + 2 * R_WORKER_STRIDE_BYTES);
        const uint16_t* h_col_idx = reinterpret_cast<const uint16_t*>(cb2_base);
        uint32_t* stats_out = reinterpret_cast<uint32_t*>(cb16_base + WORKER_STATS_OFFSET + 2 * 16);

        run_ldpc_worker(
            2, // worker_id 2 (TRISC0)
            num_cw_trisc0,
            N_var_nodes,
            M_check_nodes,
            max_check_deg,
            P_punctured,
            max_iter,
            mu_llr,
            sigma_llr,
            l_max,
            r_max,
            s0, s1, s2, s3,
            h_col_idx,
            channel_llrs,
            r_msg,
            stats_out
        );

        cb_pop_front(cb_h, 1);

        // Notify PACK that UNPACK has completed
        mailbox_write(ckernel::ThreadId::PackThreadId, 1);
    })

    // =========================================================================
    // TRISC1: MATH Thread (Worker 3)
    // =========================================================================
    MATH({
        // Wait for UNPACK confirmation that H is in L1
        mailbox_read(ckernel::ThreadId::UnpackThreadId);

        // Execute Worker 3 (TRISC1)
        uint32_t* channel_llrs = reinterpret_cast<uint32_t*>(cb0_base + 3 * LLR_WORKER_STRIDE_BYTES);
        uint32_t* r_msg = reinterpret_cast<uint32_t*>(cb1_base + 3 * R_WORKER_STRIDE_BYTES);
        const uint16_t* h_col_idx = reinterpret_cast<const uint16_t*>(cb2_base);
        uint32_t* stats_out = reinterpret_cast<uint32_t*>(cb16_base + WORKER_STATS_OFFSET + 3 * 16);

        run_ldpc_worker(
            3, // worker_id 3 (TRISC1)
            num_cw_trisc1,
            N_var_nodes,
            M_check_nodes,
            max_check_deg,
            P_punctured,
            max_iter,
            mu_llr,
            sigma_llr,
            l_max,
            r_max,
            s0, s1, s2, s3,
            h_col_idx,
            channel_llrs,
            r_msg,
            stats_out
        );

        // Notify PACK that MATH has completed
        mailbox_write(ckernel::ThreadId::PackThreadId, 1);
    })

    // =========================================================================
    // TRISC2: PACK Thread (Worker 4)
    // =========================================================================
    PACK({
        // Wait for UNPACK confirmation that H is in L1
        mailbox_read(ckernel::ThreadId::UnpackThreadId);

        // Execute Worker 4 (TRISC2)
        uint32_t* channel_llrs = reinterpret_cast<uint32_t*>(cb0_base + 4 * LLR_WORKER_STRIDE_BYTES);
        uint32_t* r_msg = reinterpret_cast<uint32_t*>(cb1_base + 4 * R_WORKER_STRIDE_BYTES);
        const uint16_t* h_col_idx = reinterpret_cast<const uint16_t*>(cb2_base);
        uint32_t* stats_out = reinterpret_cast<uint32_t*>(cb16_base + WORKER_STATS_OFFSET + 4 * 16);

        run_ldpc_worker(
            4, // worker_id 4 (TRISC2)
            num_cw_trisc2,
            N_var_nodes,
            M_check_nodes,
            max_check_deg,
            P_punctured,
            max_iter,
            mu_llr,
            sigma_llr,
            l_max,
            r_max,
            s0, s1, s2, s3,
            h_col_idx,
            channel_llrs,
            r_msg,
            stats_out
        );

        // Wait for UNPACK and MATH to complete before signaling NCRISC
        mailbox_read(ckernel::ThreadId::UnpackThreadId);
        mailbox_read(ckernel::ThreadId::MathThreadId);

        // Signal NCRISC (Writer) that all TRISC workers have completed via CB 16
        constexpr uint32_t cb_stats = tt::CBIndex::c_16;
        cb_reserve_back(cb_stats, 1);
        cb_push_back(cb_stats, 1);
    })
}