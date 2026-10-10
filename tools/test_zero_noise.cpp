#include <iostream>
#include <vector>
#include <cassert>
#include <cstring>
#include "engines/tile_matrix_4cw/ldpc_tile_matrix_core.h"
#include "tests/test_tile_matrix_40tile_lockstep_stepper.cpp"

// Just test convergence with 0 noise
int main() {
    ParityCheckMatrix pcm;
    if (!parse_chinn_out("matrices/AR4JA_r45_4c_128c_r12.chinn.out", pcm)) return 1;
    auto schedule = build_circulant_aware_tile_schedule(pcm);

    std::vector<uint8_t> vnode_l1(20480, 0);
    std::vector<std::vector<uint16_t>> r_msg_fifo(schedule.size(), std::vector<uint16_t>(1024, 0));
    std::vector<uint16_t> scratch_in(1024, 0);
    std::vector<uint16_t> scratch_out(1024, 0);

    Xoshiro128Plus rng;
    rng.init(12345);
    init_vnodes_host(vnode_l1.data(), 4.0f, 0.0f, rng); // sigma = 0 (bench)

    BatchStats stats = decode_tile_matrix_batch(
        pcm, schedule, vnode_l1.data(), r_msg_fifo, scratch_in, scratch_out, 20, 0.0f
    );

    std::cout << "Convergence test with ZERO NOISE:" << std::endl;
    for (int c = 0; c < 4; ++c) {
        std::cout << "  CW " << c << ": Converged=" << stats.converged[c] 
                  << " Iters=" << stats.iters_taken[c] 
                  << " BitErrors=" << stats.bit_errors[c] << std::endl;
    }
    return 0;
}
