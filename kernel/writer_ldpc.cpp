#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ldpc_decoder_core.h"

void kernel_main() {
    // Arguments:
    // 0: stats_dram_addr
    // 1: dram_bank
    // 2: num_codewords_ncrisc
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
    uint32_t stats_dram_addr       = get_arg_val<uint32_t>(0);
    uint32_t dram_bank             = get_arg_val<uint32_t>(1);
    uint32_t num_codewords_ncrisc = get_arg_val<uint32_t>(2);
    uint32_t N_var_nodes           = get_arg_val<uint32_t>(3);
    uint32_t M_check_nodes         = get_arg_val<uint32_t>(4);
    uint32_t P_punctured           = get_arg_val<uint32_t>(5);
    uint32_t max_check_deg         = get_arg_val<uint32_t>(6);
    uint32_t mu_bits               = get_arg_val<uint32_t>(7);
    uint32_t sigma_bits            = get_arg_val<uint32_t>(8);
    uint32_t s0                    = get_arg_val<uint32_t>(9);
    uint32_t s1                    = get_arg_val<uint32_t>(10);
    uint32_t s2                    = get_arg_val<uint32_t>(11);
    uint32_t s3                    = get_arg_val<uint32_t>(12);
    uint32_t max_iter              = get_arg_val<uint32_t>(13);
    uint32_t l_max_bits            = get_arg_val<uint32_t>(14);
    uint32_t r_max_bits            = get_arg_val<uint32_t>(15);

    float mu_llr = uint_as_float(mu_bits);
    float sigma_llr = uint_as_float(sigma_bits);
    float l_max = uint_as_float(l_max_bits);
    float r_max = uint_as_float(r_max_bits);

    // STEP 1: Wait for BRISC to load shared H matrix into L1 via CB 3
    constexpr uint32_t cb_sync_hn = tt::CBIndex::c_3;
    cb_wait_front(cb_sync_hn, 1);
    cb_pop_front(cb_sync_hn, 1);

    // STEP 2: Execute Worker 1 (NCRISC) LDPC Decoder
    uint32_t cb0_base = get_read_ptr(tt::CBIndex::c_0);
    uint32_t cb1_base = get_read_ptr(tt::CBIndex::c_1);
    uint32_t cb2_base = get_read_ptr(tt::CBIndex::c_2);
    uint32_t cb16_base = get_read_ptr(tt::CBIndex::c_16);

    uint16_t* channel_llrs = reinterpret_cast<uint16_t*>(cb0_base + 1 * LLR_WORKER_STRIDE_BYTES);
    uint16_t* r_msg = reinterpret_cast<uint16_t*>(cb1_base + 1 * R_WORKER_STRIDE_BYTES);
    const uint16_t* h_col_idx = reinterpret_cast<const uint16_t*>(cb2_base);
    uint32_t* stats_out = reinterpret_cast<uint32_t*>(cb16_base + WORKER_STATS_OFFSET + 1 * 16);

    run_ldpc_worker(
        1, // worker_id 1 (NCRISC)
        num_codewords_ncrisc,
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

    // STEP 3: Wait for BRISC Worker 0 to finish via CB 4
    constexpr uint32_t cb_sync_bd = tt::CBIndex::c_4;
    cb_wait_front(cb_sync_bd, 1);
    cb_pop_front(cb_sync_bd, 1);

    // STEP 4: Wait for TRISC Workers (2, 3, 4) to finish via CB 16
    constexpr uint32_t cb_stats = tt::CBIndex::c_16;
    cb_wait_front(cb_stats, 1);

    // STEP 5: Aggregate statistics across all 5 concurrent workers
    volatile uint32_t* all_worker_stats = reinterpret_cast<volatile uint32_t*>(cb16_base + WORKER_STATS_OFFSET);
    uint32_t total_bit_errs   = 0;
    uint32_t total_frame_errs = 0;
    for (uint32_t w = 0; w < NUM_WORKERS; w++) {
        total_bit_errs   += all_worker_stats[w * 4 + 0];
        total_frame_errs += all_worker_stats[w * 4 + 1];
    }
    uint32_t sample_iters = all_worker_stats[2]; // Worker 0 cw0 iters
    uint32_t sample_llr   = all_worker_stats[3]; // Worker 0 sample LLR

    uint32_t* final_stats = reinterpret_cast<uint32_t*>(cb16_base);
    final_stats[0] = total_bit_errs;
    final_stats[1] = total_frame_errs;
    final_stats[2] = sample_iters;
    final_stats[3] = sample_llr;

    // STEP 6: Write aggregated 16-byte summary packet to DRAM
    uint64_t dst_noc_addr = get_noc_addr_from_bank_id<true>(dram_bank, stats_dram_addr);
    noc_async_write(cb16_base, dst_noc_addr, 4 * sizeof(uint32_t));
    noc_async_write_barrier();

    cb_pop_front(cb_stats, 1);
}