#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ldpc_decoder_core.h"

void kernel_main() {
    // Arguments:
    // 0: h_dram_addr
    // 1: dram_bank
    // 2: h_bytes
    // 3: num_tiles
    // 4: num_codewords_brisc
    // 5: N_var_nodes
    // 6: M_check_nodes
    // 7: P_punctured
    // 8: max_check_deg
    // 9: mu_bits
    // 10: sigma_bits
    // 11: s0
    // 12: s1
    // 13: s2
    // 14: s3
    // 15: max_iter
    // 16: l_max_bits
    // 17: r_max_bits
    uint32_t h_dram_addr          = get_arg_val<uint32_t>(0);
    uint32_t dram_bank            = get_arg_val<uint32_t>(1);
    uint32_t h_bytes              = get_arg_val<uint32_t>(2);
    uint32_t num_tiles            = get_arg_val<uint32_t>(3);
    uint32_t num_codewords_brisc  = get_arg_val<uint32_t>(4);
    uint32_t N_var_nodes          = get_arg_val<uint32_t>(5);
    uint32_t M_check_nodes        = get_arg_val<uint32_t>(6);
    uint32_t P_punctured          = get_arg_val<uint32_t>(7);
    uint32_t max_check_deg        = get_arg_val<uint32_t>(8);
    uint32_t mu_bits              = get_arg_val<uint32_t>(9);
    uint32_t sigma_bits           = get_arg_val<uint32_t>(10);
    uint32_t s0                   = get_arg_val<uint32_t>(11);
    uint32_t s1                   = get_arg_val<uint32_t>(12);
    uint32_t s2                   = get_arg_val<uint32_t>(13);
    uint32_t s3                   = get_arg_val<uint32_t>(14);
    uint32_t max_iter             = get_arg_val<uint32_t>(15);
    uint32_t l_max_bits           = get_arg_val<uint32_t>(16);
    uint32_t r_max_bits           = get_arg_val<uint32_t>(17);

    float mu_llr = uint_as_float(mu_bits);
    float sigma_llr = uint_as_float(sigma_bits);
    float l_max = uint_as_float(l_max_bits);
    float r_max = uint_as_float(r_max_bits);

    // STEP 1: Load shared H matrix from DRAM into CB 2
    constexpr uint32_t cb_h = tt::CBIndex::c_2;
    cb_reserve_back(cb_h, 1);
    uint32_t l1_write_addr = get_write_ptr(cb_h);

    uint64_t src_noc_addr = get_noc_addr_from_bank_id<true>(dram_bank, h_dram_addr);
    noc_async_read(src_noc_addr, l1_write_addr, h_bytes);
    noc_async_read_barrier();

    // Signal UNPACK (TRISC0) that H is ready in L1
    cb_push_back(cb_h, 1);

    // Signal NCRISC (Worker 1) that H is ready in L1 via CB 3
    constexpr uint32_t cb_sync_hn = tt::CBIndex::c_3;
    cb_reserve_back(cb_sync_hn, 1);
    cb_push_back(cb_sync_hn, 1);

    // STEP 2: Execute Worker 0 (BRISC) LDPC Decoder
    uint32_t cb0_base = get_read_ptr(tt::CBIndex::c_0);
    uint32_t cb1_base = get_read_ptr(tt::CBIndex::c_1);
    uint32_t cb2_base = get_read_ptr(tt::CBIndex::c_2);
    uint32_t cb16_base = get_read_ptr(tt::CBIndex::c_16);

    uint32_t* channel_llrs = reinterpret_cast<uint32_t*>(cb0_base + 0 * LLR_WORKER_STRIDE_BYTES);
    uint32_t* r_msg = reinterpret_cast<uint32_t*>(cb1_base + 0 * R_WORKER_STRIDE_BYTES);
    const uint16_t* h_col_idx = reinterpret_cast<const uint16_t*>(cb2_base);
    uint32_t* stats_out = reinterpret_cast<uint32_t*>(cb16_base + WORKER_STATS_OFFSET + 0 * 16);

    run_ldpc_worker(
        0, // worker_id 0 (BRISC)
        num_codewords_brisc,
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

    // STEP 3: Signal NCRISC that Worker 0 is finished via CB 4
    constexpr uint32_t cb_sync_bd = tt::CBIndex::c_4;
    cb_reserve_back(cb_sync_bd, 1);
    cb_push_back(cb_sync_bd, 1);
}