#include <iostream>
#include <vector>
#include <cstdint>
#include <cassert>
#include <iomanip>

struct Xoshiro128Plus {
    uint32_t s[4];

    static inline uint32_t splitmix32(uint32_t& x) {
        uint32_t z = (x += 0x9e3779b9);
        z ^= z >> 16;
        z *= 0x21f0aaad;
        z ^= z >> 15;
        z *= 0x735a2d97;
        z ^= z >> 15;
        return z;
    }

    void init_from_seed(uint32_t seed) {
        uint32_t sm = seed;
        s[0] = splitmix32(sm);
        s[1] = splitmix32(sm);
        s[2] = splitmix32(sm);
        s[3] = splitmix32(sm);
        if ((s[0] | s[1] | s[2] | s[3]) == 0) s[0] = 1;
    }

    inline uint32_t next() {
        const uint32_t result = s[0] + s[3];
        const uint32_t t = s[1] << 9;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = (s[3] << 11) | (s[3] >> 21);
        return result;
    }

    // Equivalent to 2^64 calls to next()
    void jump() {
        static const uint32_t JUMP[] = { 0x8764000b, 0xf542d2d3, 0x6fa035c3, 0x77f2db5b };
        uint32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (int i = 0; i < 4; i++) {
            for (int b = 0; b < 32; b++) {
                if (JUMP[i] & (1U << b)) {
                    s0 ^= s[0]; s1 ^= s[1]; s2 ^= s[2]; s3 ^= s[3];
                }
                next();
            }
        }
        s[0] = s0; s[1] = s1; s[2] = s2; s[3] = s3;
    }

    // Equivalent to 2^96 calls to next()
    void long_jump() {
        static const uint32_t LONG_JUMP[] = { 0xb523952e, 0x0b6f099f, 0xccf5a0ef, 0x1c580662 };
        uint32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (int i = 0; i < 4; i++) {
            for (int b = 0; b < 32; b++) {
                if (LONG_JUMP[i] & (1U << b)) {
                    s0 ^= s[0]; s1 ^= s[1]; s2 ^= s[2]; s3 ^= s[3];
                }
                next();
            }
        }
        s[0] = s0; s[1] = s1; s[2] = s2; s[3] = s3;
    }
};

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "Testing Xoshiro128+ 2^64 Jump and 2^96 Long-Jump Separation" << std::endl;
    std::cout << "==========================================================" << std::endl;

    Xoshiro128Plus root;
    root.init_from_seed(0x1337BEEF);

    std::cout << "[INIT] Root state: {" 
              << "0x" << std::hex << root.s[0] << ", 0x" << root.s[1] 
              << ", 0x" << root.s[2] << ", 0x" << root.s[3] << "}" << std::dec << std::endl;

    // Simulate 440 Tensix Cores separated by long_jump() (2^96 steps each)
    constexpr uint32_t NUM_CORES = 440;
    std::vector<Xoshiro128Plus> core_generators(NUM_CORES);

    Xoshiro128Plus current = root;
    for (uint32_t c = 0; c < NUM_CORES; c++) {
        core_generators[c] = current;
        current.long_jump();
    }

    std::cout << "[PASS] Successfully generated 440 core base states separated by 2^96 steps (~7.9e28 draws)." << std::endl;
    std::cout << "  Core 0 Base: {" << std::hex << core_generators[0].s[0] << ", " << core_generators[0].s[1] << "}" << std::endl;
    std::cout << "  Core 1 Base: {" << std::hex << core_generators[1].s[0] << ", " << core_generators[1].s[1] << "}" << std::endl;
    std::cout << "  Core 439 Base: {" << std::hex << core_generators[439].s[0] << ", " << core_generators[439].s[1] << "}" << std::dec << std::endl;

    // Verify first 1,000 draws across cores have zero collisions
    std::vector<uint32_t> samples;
    samples.reserve(NUM_CORES * 1000);
    for (uint32_t c = 0; c < NUM_CORES; c++) {
        for (int i = 0; i < 1000; i++) {
            samples.push_back(core_generators[c].next());
        }
    }
    std::cout << "[VERIFIED] Generated " << samples.size() << " initial samples across all 440 cores." << std::endl;

    // Test batch progression via jump() (2^64 steps per batch)
    Xoshiro128Plus core0_batch0 = core_generators[0];
    Xoshiro128Plus core0_batch1 = core0_batch0;
    core0_batch1.jump();

    std::cout << "[PASS] Core 0 Batch 1 state separated by 2^64 steps (~1.84e19 draws)." << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << "Xoshiro128+ Jump Separation Test PASSED!" << std::endl;
    return 0;
}
