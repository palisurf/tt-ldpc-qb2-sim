// ============================================================================
// Tenstorrent QuietBox 2 AR4JA LDPC Decoder Engine
// ar4ja_circulant_descriptors_4cw.h
//
// Pre-computed Analytical Circulant Slice Descriptors for 4-Codeword Engine
// Extracted directly from Dr. Christopher R. Jones' makeH_AR4JA.m
//
// Key Microarchitectural Metrics:
//   - 4 Codewords interleaved: 1 variable node = 8 Bytes (64 bits)
//   - 512-bit DMA Bus (64 B/cycle): 16-node slice = 128 Bytes = 2 clock cycles!
//   - 3 Slices per Face: 6 clock cycles per face transfer!
//   - Total Tile Gather / Scatter: 12 clock cycles (Overlapped BRISC & NCRISC)
//   - Schedule: Gather (12c) -> TRISC Math (912c) -> Scatter (12c)
//   - Compute Duty Cycle: 97.4% hardware saturation!
//   - Piecewise Contiguity: 95% single-burst, 5% dual-burst boundary wrap
// ============================================================================

#pragma once

#include <cstdint>

namespace QC_LDPC_4CW {

// Memory Geometry Constants
constexpr uint32_t BYTES_PER_VNODE      = 8;   // 4 CW * 2 Bytes/bfloat16
constexpr uint32_t SLICE_NODES          = 16;  // 16 variable nodes per circulant slice
constexpr uint32_t SLICE_BYTES          = 128; // 16 * 8 Bytes = 128 Bytes (2 DMA clocks)
constexpr uint32_t DMA_BURST_CYCLES     = 2;   // 128 B / 64 B/cyc = 2 cycles!
constexpr uint32_t TOTAL_TILES          = 40;  // 32 Deg-6 Tiles + 8 Deg-3 Tiles
constexpr uint32_t DEG6_TILES           = 32;  // Checks 512..1535 (32 checks/tile)
constexpr uint32_t DEG3_TILES           = 8;   // Checks 0..511 (64 checks/tile)

// Circulant Slice Descriptor (16 Bytes per descriptor, 8-byte aligned)
struct alignas(8) CirculantSliceDesc {
    uint32_t offset_a;  // Base byte offset in CB_VNODE for Segment A (v_start_a * 8)
    uint32_t offset_b;  // Base byte offset in CB_VNODE for Segment B (0 if no wrap)
    uint8_t  len_a;     // Bytes in Segment A (typically 128)
    uint8_t  len_b;     // Bytes in Segment B (typically 0)
    uint16_t reserved;  // Alignment / padding
};

// Worker 0 Face 0 Slices (Edges 0..2 for Deg 6, Checks 0..15 for Deg 3)
constexpr CirculantSliceDesc TILE_SLICES_W0_F0[40][3] = {
    // Tile  0
    {
        { 0x00000u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=   0 (128B)
        { 0x01000u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 512 (128B)
        { 0x03000u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=1536 (128B)
    },
    // Tile  1
    {
        { 0x00100u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=  32 (128B)
        { 0x01100u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 544 (128B)
        { 0x03100u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=1568 (128B)
    },
    // Tile  2
    {
        { 0x00200u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=  64 (128B)
        { 0x01200u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 576 (128B)
        { 0x03200u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=1600 (128B)
    },
    // Tile  3
    {
        { 0x00300u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=  96 (128B)
        { 0x01300u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 608 (128B)
        { 0x03300u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=1632 (128B)
    },
    // Tile  4
    {
        { 0x00400u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 128 (128B)
        { 0x01400u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 640 (128B)
        { 0x03400u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=1664 (128B)
    },
    // Tile  5
    {
        { 0x00500u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 160 (128B)
        { 0x01500u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 672 (128B)
        { 0x03500u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=1696 (128B)
    },
    // Tile  6
    {
        { 0x00600u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 192 (128B)
        { 0x01600u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 704 (128B)
        { 0x03600u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=1728 (128B)
    },
    // Tile  7
    {
        { 0x00700u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 224 (128B)
        { 0x01700u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 736 (128B)
        { 0x03700u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=1760 (128B)
    },
    // Tile  8
    {
        { 0x00800u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 256 (128B)
        { 0x01800u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 768 (128B)
        { 0x03800u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=1792 (128B)
    },
    // Tile  9
    {
        { 0x00900u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 288 (128B)
        { 0x01900u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 800 (128B)
        { 0x03900u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=1824 (128B)
    },
    // Tile 10
    {
        { 0x00A00u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 320 (128B)
        { 0x01A00u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 832 (128B)
        { 0x03A00u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=1856 (128B)
    },
    // Tile 11
    {
        { 0x00B00u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 352 (128B)
        { 0x01B00u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 864 (128B)
        { 0x03B00u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=1888 (128B)
    },
    // Tile 12
    {
        { 0x00C00u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 384 (128B)
        { 0x01C00u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 896 (128B)
        { 0x03C00u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=1920 (128B)
    },
    // Tile 13
    {
        { 0x00D00u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 416 (128B)
        { 0x01D00u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 928 (128B)
        { 0x03D00u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=1952 (128B)
    },
    // Tile 14
    {
        { 0x00E00u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 448 (128B)
        { 0x01E00u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 960 (128B)
        { 0x03E00u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=1984 (128B)
    },
    // Tile 15
    {
        { 0x00F00u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 480 (128B)
        { 0x01F00u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 992 (128B)
        { 0x03F00u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2016 (128B)
    },
    // Tile 16
    {
        { 0x00000u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=   0 (128B)
        { 0x01990u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 818 (128B)
        { 0x01CE8u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v= 925 (128B)
    },
    // Tile 17
    {
        { 0x00100u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=  32 (128B)
        { 0x01A90u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 850 (128B)
        { 0x01DE8u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v= 957 (128B)
    },
    // Tile 18
    {
        { 0x00200u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=  64 (128B)
        { 0x01B90u, 0x01800u, 112u,  16u, 0u }, // Slice 1: v= 882(112B) + v= 768(16B)
        { 0x01EE8u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v= 989 (128B)
    },
    // Tile 19
    {
        { 0x00300u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=  96 (128B)
        { 0x01890u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 786 (128B)
        { 0x01FE8u, 0x01C00u,  24u, 104u, 0u }, // Slice 2: v=1021(24B) + v= 896(104B)
    },
    // Tile 20
    {
        { 0x00400u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 128 (128B)
        { 0x01000u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 512 (128B)
        { 0x01D78u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v= 943 (128B)
    },
    // Tile 21
    {
        { 0x00500u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 160 (128B)
        { 0x01100u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 544 (128B)
        { 0x01E78u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v= 975 (128B)
    },
    // Tile 22
    {
        { 0x00600u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 192 (128B)
        { 0x01200u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 576 (128B)
        { 0x01F78u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=1007 (128B)
    },
    // Tile 23
    {
        { 0x00700u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 224 (128B)
        { 0x01300u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 608 (128B)
        { 0x01C78u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v= 911 (128B)
    },
    // Tile 24
    {
        { 0x00800u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 256 (128B)
        { 0x010F8u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 543 (128B)
        { 0x017D0u, 0x01400u,  48u,  80u, 0u }, // Slice 2: v= 762(48B) + v= 640(80B)
    },
    // Tile 25
    {
        { 0x00900u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 288 (128B)
        { 0x011F8u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 575 (128B)
        { 0x014D0u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v= 666 (128B)
    },
    // Tile 26
    {
        { 0x00A00u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 320 (128B)
        { 0x012F8u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 607 (128B)
        { 0x015D0u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v= 698 (128B)
    },
    // Tile 27
    {
        { 0x00B00u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 352 (128B)
        { 0x013F8u, 0x01000u,   8u, 120u, 0u }, // Slice 1: v= 639(8B) + v= 512(120B)
        { 0x016D0u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v= 730 (128B)
    },
    // Tile 28
    {
        { 0x00C00u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 384 (128B)
        { 0x01600u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 704 (128B)
        { 0x01AE8u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v= 861 (128B)
    },
    // Tile 29
    {
        { 0x00D00u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 416 (128B)
        { 0x01700u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 736 (128B)
        { 0x01BE8u, 0x01800u,  24u, 104u, 0u }, // Slice 2: v= 893(24B) + v= 768(104B)
    },
    // Tile 30
    {
        { 0x00E00u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 448 (128B)
        { 0x01400u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 640 (128B)
        { 0x018E8u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v= 797 (128B)
    },
    // Tile 31
    {
        { 0x00F00u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 480 (128B)
        { 0x01500u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 672 (128B)
        { 0x019E8u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v= 829 (128B)
    },
    // Tile 32
    {
        { 0x02000u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1024 (128B)
        { 0x04000u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2048 (128B)
        { 0x04C80u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2448 (128B)
    },
    // Tile 33
    {
        { 0x02200u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1088 (128B)
        { 0x04200u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2112 (128B)
        { 0x04E80u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2512 (128B)
    },
    // Tile 34
    {
        { 0x02400u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1152 (128B)
        { 0x04000u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2048 (128B)
        { 0x04400u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2176 (128B)
    },
    // Tile 35
    {
        { 0x02600u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1216 (128B)
        { 0x04200u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2112 (128B)
        { 0x04600u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2240 (128B)
    },
    // Tile 36
    {
        { 0x02800u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1280 (128B)
        { 0x04400u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2176 (128B)
        { 0x04800u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2304 (128B)
    },
    // Tile 37
    {
        { 0x02A00u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1344 (128B)
        { 0x04600u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2240 (128B)
        { 0x04A00u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2368 (128B)
    },
    // Tile 38
    {
        { 0x02C00u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1408 (128B)
        { 0x04800u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2304 (128B)
        { 0x04C00u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2432 (128B)
    },
    // Tile 39
    {
        { 0x02E00u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1472 (128B)
        { 0x04A00u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2368 (128B)
        { 0x04E00u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2496 (128B)
    },
};

// Worker 0 Face 2 Slices (Edges 3..5 for Deg 6, Checks 32..47 for Deg 3)
constexpr CirculantSliceDesc TILE_SLICES_W0_F2[40][3] = {
    // Tile  0
    {
        { 0x04338u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=2151 (128B)
        { 0x04748u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2281 (128B)
        { 0x04800u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2304 (128B)
    },
    // Tile  1
    {
        { 0x04038u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=2055 (128B)
        { 0x04448u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2185 (128B)
        { 0x04900u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2336 (128B)
    },
    // Tile  2
    {
        { 0x04138u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=2087 (128B)
        { 0x04548u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2217 (128B)
        { 0x04A00u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2368 (128B)
    },
    // Tile  3
    {
        { 0x04238u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=2119 (128B)
        { 0x04648u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2249 (128B)
        { 0x04B00u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2400 (128B)
    },
    // Tile  4
    {
        { 0x045A8u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=2229 (128B)
        { 0x04A50u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2378 (128B)
        { 0x04D68u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2477 (128B)
    },
    // Tile  5
    {
        { 0x046A8u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=2261 (128B)
        { 0x04B50u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2410 (128B)
        { 0x04E68u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2509 (128B)
    },
    // Tile  6
    {
        { 0x047A8u, 0x04400u,  88u,  40u, 0u }, // Slice 0: v=2293(88B) + v=2176(40B)
        { 0x04850u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2314 (128B)
        { 0x04F68u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2541 (128B)
    },
    // Tile  7
    {
        { 0x044A8u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=2197 (128B)
        { 0x04950u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2346 (128B)
        { 0x04C68u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2445 (128B)
    },
    // Tile  8
    {
        { 0x042C8u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=2137 (128B)
        { 0x04840u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2312 (128B)
        { 0x04FB8u, 0x04C00u,  72u,  56u, 0u }, // Slice 2: v=2551(72B) + v=2432(56B)
    },
    // Tile  9
    {
        { 0x043C8u, 0x04000u,  56u,  72u, 0u }, // Slice 0: v=2169(56B) + v=2048(72B)
        { 0x04940u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2344 (128B)
        { 0x04CB8u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2455 (128B)
    },
    // Tile 10
    {
        { 0x040C8u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=2073 (128B)
        { 0x04A40u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2376 (128B)
        { 0x04DB8u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2487 (128B)
    },
    // Tile 11
    {
        { 0x041C8u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=2105 (128B)
        { 0x04B40u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2408 (128B)
        { 0x04EB8u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2519 (128B)
    },
    // Tile 12
    {
        { 0x04308u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=2145 (128B)
        { 0x04780u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2288 (128B)
        { 0x04D18u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2467 (128B)
    },
    // Tile 13
    {
        { 0x04008u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=2049 (128B)
        { 0x04480u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2192 (128B)
        { 0x04E18u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2499 (128B)
    },
    // Tile 14
    {
        { 0x04108u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=2081 (128B)
        { 0x04580u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2224 (128B)
        { 0x04F18u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2531 (128B)
    },
    // Tile 15
    {
        { 0x04208u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=2113 (128B)
        { 0x04680u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2256 (128B)
        { 0x04C18u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2435 (128B)
    },
    // Tile 16
    {
        { 0x03398u, 0x03000u, 104u,  24u, 0u }, // Slice 0: v=1651(104B) + v=1536(24B)
        { 0x034F0u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=1694 (128B)
        { 0x04000u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2048 (128B)
    },
    // Tile 17
    {
        { 0x03098u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1555 (128B)
        { 0x035F0u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=1726 (128B)
        { 0x04100u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2080 (128B)
    },
    // Tile 18
    {
        { 0x03198u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1587 (128B)
        { 0x036F0u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=1758 (128B)
        { 0x04200u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2112 (128B)
    },
    // Tile 19
    {
        { 0x03298u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1619 (128B)
        { 0x037F0u, 0x03400u,  16u, 112u, 0u }, // Slice 1: v=1790(16B) + v=1664(112B)
        { 0x04300u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2144 (128B)
    },
    // Tile 20
    {
        { 0x035D8u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1723 (128B)
        { 0x03B30u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=1894 (128B)
        { 0x04400u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2176 (128B)
    },
    // Tile 21
    {
        { 0x036D8u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1755 (128B)
        { 0x03830u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=1798 (128B)
        { 0x04500u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2208 (128B)
    },
    // Tile 22
    {
        { 0x037D8u, 0x03400u,  40u,  88u, 0u }, // Slice 0: v=1787(40B) + v=1664(88B)
        { 0x03930u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=1830 (128B)
        { 0x04600u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2240 (128B)
    },
    // Tile 23
    {
        { 0x034D8u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1691 (128B)
        { 0x03A30u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=1862 (128B)
        { 0x04700u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2272 (128B)
    },
    // Tile 24
    {
        { 0x03808u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1793 (128B)
        { 0x03E28u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=1989 (128B)
        { 0x04800u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2304 (128B)
    },
    // Tile 25
    {
        { 0x03908u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1825 (128B)
        { 0x03F28u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2021 (128B)
        { 0x04900u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2336 (128B)
    },
    // Tile 26
    {
        { 0x03A08u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1857 (128B)
        { 0x03C28u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=1925 (128B)
        { 0x04A00u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2368 (128B)
    },
    // Tile 27
    {
        { 0x03B08u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1889 (128B)
        { 0x03D28u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=1957 (128B)
        { 0x04B00u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2400 (128B)
    },
    // Tile 28
    {
        { 0x032F0u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1630 (128B)
        { 0x03F18u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2019 (128B)
        { 0x04C00u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2432 (128B)
    },
    // Tile 29
    {
        { 0x033F0u, 0x03000u,  16u, 112u, 0u }, // Slice 0: v=1662(16B) + v=1536(112B)
        { 0x03C18u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=1923 (128B)
        { 0x04D00u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2464 (128B)
    },
    // Tile 30
    {
        { 0x030F0u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1566 (128B)
        { 0x03D18u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=1955 (128B)
        { 0x04E00u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2496 (128B)
    },
    // Tile 31
    {
        { 0x031F0u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1598 (128B)
        { 0x03E18u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=1987 (128B)
        { 0x04F00u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2528 (128B)
    },
    // Tile 32
    {
        { 0x02100u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1056 (128B)
        { 0x04100u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2080 (128B)
        { 0x04D80u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2480 (128B)
    },
    // Tile 33
    {
        { 0x02300u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1120 (128B)
        { 0x04300u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2144 (128B)
        { 0x04F80u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2544 (128B)
    },
    // Tile 34
    {
        { 0x02500u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1184 (128B)
        { 0x04100u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2080 (128B)
        { 0x04500u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2208 (128B)
    },
    // Tile 35
    {
        { 0x02700u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1248 (128B)
        { 0x04300u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2144 (128B)
        { 0x04700u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2272 (128B)
    },
    // Tile 36
    {
        { 0x02900u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1312 (128B)
        { 0x04500u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2208 (128B)
        { 0x04900u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2336 (128B)
    },
    // Tile 37
    {
        { 0x02B00u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1376 (128B)
        { 0x04700u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2272 (128B)
        { 0x04B00u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2400 (128B)
    },
    // Tile 38
    {
        { 0x02D00u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1440 (128B)
        { 0x04900u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2336 (128B)
        { 0x04D00u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2464 (128B)
    },
    // Tile 39
    {
        { 0x02F00u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1504 (128B)
        { 0x04B00u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2400 (128B)
        { 0x04F00u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2528 (128B)
    },
};

// Worker 1 Face 1 Slices (Edges 0..2 for Deg 6, Checks 16..31 for Deg 3)
constexpr CirculantSliceDesc TILE_SLICES_W1_F1[40][3] = {
    // Tile  0
    {
        { 0x00080u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=  16 (128B)
        { 0x01080u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 528 (128B)
        { 0x03080u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=1552 (128B)
    },
    // Tile  1
    {
        { 0x00180u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=  48 (128B)
        { 0x01180u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 560 (128B)
        { 0x03180u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=1584 (128B)
    },
    // Tile  2
    {
        { 0x00280u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=  80 (128B)
        { 0x01280u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 592 (128B)
        { 0x03280u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=1616 (128B)
    },
    // Tile  3
    {
        { 0x00380u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 112 (128B)
        { 0x01380u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 624 (128B)
        { 0x03380u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=1648 (128B)
    },
    // Tile  4
    {
        { 0x00480u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 144 (128B)
        { 0x01480u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 656 (128B)
        { 0x03480u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=1680 (128B)
    },
    // Tile  5
    {
        { 0x00580u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 176 (128B)
        { 0x01580u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 688 (128B)
        { 0x03580u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=1712 (128B)
    },
    // Tile  6
    {
        { 0x00680u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 208 (128B)
        { 0x01680u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 720 (128B)
        { 0x03680u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=1744 (128B)
    },
    // Tile  7
    {
        { 0x00780u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 240 (128B)
        { 0x01780u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 752 (128B)
        { 0x03780u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=1776 (128B)
    },
    // Tile  8
    {
        { 0x00880u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 272 (128B)
        { 0x01880u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 784 (128B)
        { 0x03880u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=1808 (128B)
    },
    // Tile  9
    {
        { 0x00980u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 304 (128B)
        { 0x01980u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 816 (128B)
        { 0x03980u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=1840 (128B)
    },
    // Tile 10
    {
        { 0x00A80u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 336 (128B)
        { 0x01A80u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 848 (128B)
        { 0x03A80u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=1872 (128B)
    },
    // Tile 11
    {
        { 0x00B80u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 368 (128B)
        { 0x01B80u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 880 (128B)
        { 0x03B80u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=1904 (128B)
    },
    // Tile 12
    {
        { 0x00C80u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 400 (128B)
        { 0x01C80u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 912 (128B)
        { 0x03C80u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=1936 (128B)
    },
    // Tile 13
    {
        { 0x00D80u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 432 (128B)
        { 0x01D80u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 944 (128B)
        { 0x03D80u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=1968 (128B)
    },
    // Tile 14
    {
        { 0x00E80u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 464 (128B)
        { 0x01E80u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 976 (128B)
        { 0x03E80u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2000 (128B)
    },
    // Tile 15
    {
        { 0x00F80u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 496 (128B)
        { 0x01F80u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=1008 (128B)
        { 0x03F80u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2032 (128B)
    },
    // Tile 16
    {
        { 0x00080u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=  16 (128B)
        { 0x01A10u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 834 (128B)
        { 0x01D68u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v= 941 (128B)
    },
    // Tile 17
    {
        { 0x00180u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=  48 (128B)
        { 0x01B10u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 866 (128B)
        { 0x01E68u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v= 973 (128B)
    },
    // Tile 18
    {
        { 0x00280u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=  80 (128B)
        { 0x01810u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 770 (128B)
        { 0x01F68u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=1005 (128B)
    },
    // Tile 19
    {
        { 0x00380u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 112 (128B)
        { 0x01910u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 802 (128B)
        { 0x01C68u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v= 909 (128B)
    },
    // Tile 20
    {
        { 0x00480u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 144 (128B)
        { 0x01080u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 528 (128B)
        { 0x01DF8u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v= 959 (128B)
    },
    // Tile 21
    {
        { 0x00580u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 176 (128B)
        { 0x01180u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 560 (128B)
        { 0x01EF8u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v= 991 (128B)
    },
    // Tile 22
    {
        { 0x00680u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 208 (128B)
        { 0x01280u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 592 (128B)
        { 0x01FF8u, 0x01C00u,   8u, 120u, 0u }, // Slice 2: v=1023(8B) + v= 896(120B)
    },
    // Tile 23
    {
        { 0x00780u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 240 (128B)
        { 0x01380u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 624 (128B)
        { 0x01CF8u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v= 927 (128B)
    },
    // Tile 24
    {
        { 0x00880u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 272 (128B)
        { 0x01178u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 559 (128B)
        { 0x01450u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v= 650 (128B)
    },
    // Tile 25
    {
        { 0x00980u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 304 (128B)
        { 0x01278u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 591 (128B)
        { 0x01550u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v= 682 (128B)
    },
    // Tile 26
    {
        { 0x00A80u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 336 (128B)
        { 0x01378u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 623 (128B)
        { 0x01650u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v= 714 (128B)
    },
    // Tile 27
    {
        { 0x00B80u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 368 (128B)
        { 0x01078u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 527 (128B)
        { 0x01750u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v= 746 (128B)
    },
    // Tile 28
    {
        { 0x00C80u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 400 (128B)
        { 0x01680u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 720 (128B)
        { 0x01B68u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v= 877 (128B)
    },
    // Tile 29
    {
        { 0x00D80u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 432 (128B)
        { 0x01780u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 752 (128B)
        { 0x01868u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v= 781 (128B)
    },
    // Tile 30
    {
        { 0x00E80u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 464 (128B)
        { 0x01480u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 656 (128B)
        { 0x01968u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v= 813 (128B)
    },
    // Tile 31
    {
        { 0x00F80u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v= 496 (128B)
        { 0x01580u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v= 688 (128B)
        { 0x01A68u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v= 845 (128B)
    },
    // Tile 32
    {
        { 0x02080u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1040 (128B)
        { 0x04080u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2064 (128B)
        { 0x04D00u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2464 (128B)
    },
    // Tile 33
    {
        { 0x02280u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1104 (128B)
        { 0x04280u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2128 (128B)
        { 0x04F00u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2528 (128B)
    },
    // Tile 34
    {
        { 0x02480u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1168 (128B)
        { 0x04080u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2064 (128B)
        { 0x04480u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2192 (128B)
    },
    // Tile 35
    {
        { 0x02680u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1232 (128B)
        { 0x04280u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2128 (128B)
        { 0x04680u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2256 (128B)
    },
    // Tile 36
    {
        { 0x02880u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1296 (128B)
        { 0x04480u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2192 (128B)
        { 0x04880u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2320 (128B)
    },
    // Tile 37
    {
        { 0x02A80u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1360 (128B)
        { 0x04680u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2256 (128B)
        { 0x04A80u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2384 (128B)
    },
    // Tile 38
    {
        { 0x02C80u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1424 (128B)
        { 0x04880u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2320 (128B)
        { 0x04C80u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2448 (128B)
    },
    // Tile 39
    {
        { 0x02E80u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1488 (128B)
        { 0x04A80u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2384 (128B)
        { 0x04E80u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2512 (128B)
    },
};

// Worker 1 Face 3 Slices (Edges 3..5 for Deg 6, Checks 48..63 for Deg 3)
constexpr CirculantSliceDesc TILE_SLICES_W1_F3[40][3] = {
    // Tile  0
    {
        { 0x043B8u, 0x04000u,  72u,  56u, 0u }, // Slice 0: v=2167(72B) + v=2048(56B)
        { 0x047C8u, 0x04400u,  56u,  72u, 0u }, // Slice 1: v=2297(56B) + v=2176(72B)
        { 0x04880u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2320 (128B)
    },
    // Tile  1
    {
        { 0x040B8u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=2071 (128B)
        { 0x044C8u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2201 (128B)
        { 0x04980u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2352 (128B)
    },
    // Tile  2
    {
        { 0x041B8u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=2103 (128B)
        { 0x045C8u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2233 (128B)
        { 0x04A80u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2384 (128B)
    },
    // Tile  3
    {
        { 0x042B8u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=2135 (128B)
        { 0x046C8u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2265 (128B)
        { 0x04B80u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2416 (128B)
    },
    // Tile  4
    {
        { 0x04628u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=2245 (128B)
        { 0x04AD0u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2394 (128B)
        { 0x04DE8u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2493 (128B)
    },
    // Tile  5
    {
        { 0x04728u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=2277 (128B)
        { 0x04BD0u, 0x04800u,  48u,  80u, 0u }, // Slice 1: v=2426(48B) + v=2304(80B)
        { 0x04EE8u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2525 (128B)
    },
    // Tile  6
    {
        { 0x04428u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=2181 (128B)
        { 0x048D0u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2330 (128B)
        { 0x04FE8u, 0x04C00u,  24u, 104u, 0u }, // Slice 2: v=2557(24B) + v=2432(104B)
    },
    // Tile  7
    {
        { 0x04528u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=2213 (128B)
        { 0x049D0u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2362 (128B)
        { 0x04CE8u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2461 (128B)
    },
    // Tile  8
    {
        { 0x04348u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=2153 (128B)
        { 0x048C0u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2328 (128B)
        { 0x04C38u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2439 (128B)
    },
    // Tile  9
    {
        { 0x04048u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=2057 (128B)
        { 0x049C0u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2360 (128B)
        { 0x04D38u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2471 (128B)
    },
    // Tile 10
    {
        { 0x04148u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=2089 (128B)
        { 0x04AC0u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2392 (128B)
        { 0x04E38u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2503 (128B)
    },
    // Tile 11
    {
        { 0x04248u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=2121 (128B)
        { 0x04BC0u, 0x04800u,  64u,  64u, 0u }, // Slice 1: v=2424(64B) + v=2304(64B)
        { 0x04F38u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2535 (128B)
    },
    // Tile 12
    {
        { 0x04388u, 0x04000u, 120u,   8u, 0u }, // Slice 0: v=2161(120B) + v=2048(8B)
        { 0x04400u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2176 (128B)
        { 0x04D98u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2483 (128B)
    },
    // Tile 13
    {
        { 0x04088u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=2065 (128B)
        { 0x04500u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2208 (128B)
        { 0x04E98u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2515 (128B)
    },
    // Tile 14
    {
        { 0x04188u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=2097 (128B)
        { 0x04600u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2240 (128B)
        { 0x04F98u, 0x04C00u, 104u,  24u, 0u }, // Slice 2: v=2547(104B) + v=2432(24B)
    },
    // Tile 15
    {
        { 0x04288u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=2129 (128B)
        { 0x04700u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2272 (128B)
        { 0x04C98u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2451 (128B)
    },
    // Tile 16
    {
        { 0x03018u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1539 (128B)
        { 0x03570u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=1710 (128B)
        { 0x04080u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2064 (128B)
    },
    // Tile 17
    {
        { 0x03118u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1571 (128B)
        { 0x03670u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=1742 (128B)
        { 0x04180u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2096 (128B)
    },
    // Tile 18
    {
        { 0x03218u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1603 (128B)
        { 0x03770u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=1774 (128B)
        { 0x04280u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2128 (128B)
    },
    // Tile 19
    {
        { 0x03318u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1635 (128B)
        { 0x03470u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=1678 (128B)
        { 0x04380u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2160 (128B)
    },
    // Tile 20
    {
        { 0x03658u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1739 (128B)
        { 0x03BB0u, 0x03800u,  80u,  48u, 0u }, // Slice 1: v=1910(80B) + v=1792(48B)
        { 0x04480u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2192 (128B)
    },
    // Tile 21
    {
        { 0x03758u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1771 (128B)
        { 0x038B0u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=1814 (128B)
        { 0x04580u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2224 (128B)
    },
    // Tile 22
    {
        { 0x03458u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1675 (128B)
        { 0x039B0u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=1846 (128B)
        { 0x04680u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2256 (128B)
    },
    // Tile 23
    {
        { 0x03558u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1707 (128B)
        { 0x03AB0u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=1878 (128B)
        { 0x04780u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2288 (128B)
    },
    // Tile 24
    {
        { 0x03888u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1809 (128B)
        { 0x03EA8u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2005 (128B)
        { 0x04880u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2320 (128B)
    },
    // Tile 25
    {
        { 0x03988u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1841 (128B)
        { 0x03FA8u, 0x03C00u,  88u,  40u, 0u }, // Slice 1: v=2037(88B) + v=1920(40B)
        { 0x04980u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2352 (128B)
    },
    // Tile 26
    {
        { 0x03A88u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1873 (128B)
        { 0x03CA8u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=1941 (128B)
        { 0x04A80u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2384 (128B)
    },
    // Tile 27
    {
        { 0x03B88u, 0x03800u, 120u,   8u, 0u }, // Slice 0: v=1905(120B) + v=1792(8B)
        { 0x03DA8u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=1973 (128B)
        { 0x04B80u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2416 (128B)
    },
    // Tile 28
    {
        { 0x03370u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1646 (128B)
        { 0x03F98u, 0x03C00u, 104u,  24u, 0u }, // Slice 1: v=2035(104B) + v=1920(24B)
        { 0x04C80u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2448 (128B)
    },
    // Tile 29
    {
        { 0x03070u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1550 (128B)
        { 0x03C98u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=1939 (128B)
        { 0x04D80u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2480 (128B)
    },
    // Tile 30
    {
        { 0x03170u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1582 (128B)
        { 0x03D98u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=1971 (128B)
        { 0x04E80u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2512 (128B)
    },
    // Tile 31
    {
        { 0x03270u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1614 (128B)
        { 0x03E98u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2003 (128B)
        { 0x04F80u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2544 (128B)
    },
    // Tile 32
    {
        { 0x02180u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1072 (128B)
        { 0x04180u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2096 (128B)
        { 0x04E00u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2496 (128B)
    },
    // Tile 33
    {
        { 0x02380u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1136 (128B)
        { 0x04380u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2160 (128B)
        { 0x04C00u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2432 (128B)
    },
    // Tile 34
    {
        { 0x02580u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1200 (128B)
        { 0x04180u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2096 (128B)
        { 0x04580u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2224 (128B)
    },
    // Tile 35
    {
        { 0x02780u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1264 (128B)
        { 0x04380u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2160 (128B)
        { 0x04780u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2288 (128B)
    },
    // Tile 36
    {
        { 0x02980u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1328 (128B)
        { 0x04580u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2224 (128B)
        { 0x04980u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2352 (128B)
    },
    // Tile 37
    {
        { 0x02B80u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1392 (128B)
        { 0x04780u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2288 (128B)
        { 0x04B80u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2416 (128B)
    },
    // Tile 38
    {
        { 0x02D80u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1456 (128B)
        { 0x04980u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2352 (128B)
        { 0x04D80u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2480 (128B)
    },
    // Tile 39
    {
        { 0x02F80u, 0x00000u, 128u,   0u, 0u }, // Slice 0: v=1520 (128B)
        { 0x04B80u, 0x00000u, 128u,   0u, 0u }, // Slice 1: v=2416 (128B)
        { 0x04F80u, 0x00000u, 128u,   0u, 0u }, // Slice 2: v=2544 (128B)
    },
};

// Check to Variable Map for Full Parity Syndrome Verification (H * c^T == 0)
constexpr uint16_t CHECK_TO_VAR[1536][6] = {
    { 1024, 2048, 2448, 65535, 65535, 65535 },
    { 1025, 2049, 2449, 65535, 65535, 65535 },
    { 1026, 2050, 2450, 65535, 65535, 65535 },
    { 1027, 2051, 2451, 65535, 65535, 65535 },
    { 1028, 2052, 2452, 65535, 65535, 65535 },
    { 1029, 2053, 2453, 65535, 65535, 65535 },
    { 1030, 2054, 2454, 65535, 65535, 65535 },
    { 1031, 2055, 2455, 65535, 65535, 65535 },
    { 1032, 2056, 2456, 65535, 65535, 65535 },
    { 1033, 2057, 2457, 65535, 65535, 65535 },
    { 1034, 2058, 2458, 65535, 65535, 65535 },
    { 1035, 2059, 2459, 65535, 65535, 65535 },
    { 1036, 2060, 2460, 65535, 65535, 65535 },
    { 1037, 2061, 2461, 65535, 65535, 65535 },
    { 1038, 2062, 2462, 65535, 65535, 65535 },
    { 1039, 2063, 2463, 65535, 65535, 65535 },
    { 1040, 2064, 2464, 65535, 65535, 65535 },
    { 1041, 2065, 2465, 65535, 65535, 65535 },
    { 1042, 2066, 2466, 65535, 65535, 65535 },
    { 1043, 2067, 2467, 65535, 65535, 65535 },
    { 1044, 2068, 2468, 65535, 65535, 65535 },
    { 1045, 2069, 2469, 65535, 65535, 65535 },
    { 1046, 2070, 2470, 65535, 65535, 65535 },
    { 1047, 2071, 2471, 65535, 65535, 65535 },
    { 1048, 2072, 2472, 65535, 65535, 65535 },
    { 1049, 2073, 2473, 65535, 65535, 65535 },
    { 1050, 2074, 2474, 65535, 65535, 65535 },
    { 1051, 2075, 2475, 65535, 65535, 65535 },
    { 1052, 2076, 2476, 65535, 65535, 65535 },
    { 1053, 2077, 2477, 65535, 65535, 65535 },
    { 1054, 2078, 2478, 65535, 65535, 65535 },
    { 1055, 2079, 2479, 65535, 65535, 65535 },
    { 1056, 2080, 2480, 65535, 65535, 65535 },
    { 1057, 2081, 2481, 65535, 65535, 65535 },
    { 1058, 2082, 2482, 65535, 65535, 65535 },
    { 1059, 2083, 2483, 65535, 65535, 65535 },
    { 1060, 2084, 2484, 65535, 65535, 65535 },
    { 1061, 2085, 2485, 65535, 65535, 65535 },
    { 1062, 2086, 2486, 65535, 65535, 65535 },
    { 1063, 2087, 2487, 65535, 65535, 65535 },
    { 1064, 2088, 2488, 65535, 65535, 65535 },
    { 1065, 2089, 2489, 65535, 65535, 65535 },
    { 1066, 2090, 2490, 65535, 65535, 65535 },
    { 1067, 2091, 2491, 65535, 65535, 65535 },
    { 1068, 2092, 2492, 65535, 65535, 65535 },
    { 1069, 2093, 2493, 65535, 65535, 65535 },
    { 1070, 2094, 2494, 65535, 65535, 65535 },
    { 1071, 2095, 2495, 65535, 65535, 65535 },
    { 1072, 2096, 2496, 65535, 65535, 65535 },
    { 1073, 2097, 2497, 65535, 65535, 65535 },
    { 1074, 2098, 2498, 65535, 65535, 65535 },
    { 1075, 2099, 2499, 65535, 65535, 65535 },
    { 1076, 2100, 2500, 65535, 65535, 65535 },
    { 1077, 2101, 2501, 65535, 65535, 65535 },
    { 1078, 2102, 2502, 65535, 65535, 65535 },
    { 1079, 2103, 2503, 65535, 65535, 65535 },
    { 1080, 2104, 2504, 65535, 65535, 65535 },
    { 1081, 2105, 2505, 65535, 65535, 65535 },
    { 1082, 2106, 2506, 65535, 65535, 65535 },
    { 1083, 2107, 2507, 65535, 65535, 65535 },
    { 1084, 2108, 2508, 65535, 65535, 65535 },
    { 1085, 2109, 2509, 65535, 65535, 65535 },
    { 1086, 2110, 2510, 65535, 65535, 65535 },
    { 1087, 2111, 2511, 65535, 65535, 65535 },
    { 1088, 2112, 2512, 65535, 65535, 65535 },
    { 1089, 2113, 2513, 65535, 65535, 65535 },
    { 1090, 2114, 2514, 65535, 65535, 65535 },
    { 1091, 2115, 2515, 65535, 65535, 65535 },
    { 1092, 2116, 2516, 65535, 65535, 65535 },
    { 1093, 2117, 2517, 65535, 65535, 65535 },
    { 1094, 2118, 2518, 65535, 65535, 65535 },
    { 1095, 2119, 2519, 65535, 65535, 65535 },
    { 1096, 2120, 2520, 65535, 65535, 65535 },
    { 1097, 2121, 2521, 65535, 65535, 65535 },
    { 1098, 2122, 2522, 65535, 65535, 65535 },
    { 1099, 2123, 2523, 65535, 65535, 65535 },
    { 1100, 2124, 2524, 65535, 65535, 65535 },
    { 1101, 2125, 2525, 65535, 65535, 65535 },
    { 1102, 2126, 2526, 65535, 65535, 65535 },
    { 1103, 2127, 2527, 65535, 65535, 65535 },
    { 1104, 2128, 2528, 65535, 65535, 65535 },
    { 1105, 2129, 2529, 65535, 65535, 65535 },
    { 1106, 2130, 2530, 65535, 65535, 65535 },
    { 1107, 2131, 2531, 65535, 65535, 65535 },
    { 1108, 2132, 2532, 65535, 65535, 65535 },
    { 1109, 2133, 2533, 65535, 65535, 65535 },
    { 1110, 2134, 2534, 65535, 65535, 65535 },
    { 1111, 2135, 2535, 65535, 65535, 65535 },
    { 1112, 2136, 2536, 65535, 65535, 65535 },
    { 1113, 2137, 2537, 65535, 65535, 65535 },
    { 1114, 2138, 2538, 65535, 65535, 65535 },
    { 1115, 2139, 2539, 65535, 65535, 65535 },
    { 1116, 2140, 2540, 65535, 65535, 65535 },
    { 1117, 2141, 2541, 65535, 65535, 65535 },
    { 1118, 2142, 2542, 65535, 65535, 65535 },
    { 1119, 2143, 2543, 65535, 65535, 65535 },
    { 1120, 2144, 2544, 65535, 65535, 65535 },
    { 1121, 2145, 2545, 65535, 65535, 65535 },
    { 1122, 2146, 2546, 65535, 65535, 65535 },
    { 1123, 2147, 2547, 65535, 65535, 65535 },
    { 1124, 2148, 2548, 65535, 65535, 65535 },
    { 1125, 2149, 2549, 65535, 65535, 65535 },
    { 1126, 2150, 2550, 65535, 65535, 65535 },
    { 1127, 2151, 2551, 65535, 65535, 65535 },
    { 1128, 2152, 2552, 65535, 65535, 65535 },
    { 1129, 2153, 2553, 65535, 65535, 65535 },
    { 1130, 2154, 2554, 65535, 65535, 65535 },
    { 1131, 2155, 2555, 65535, 65535, 65535 },
    { 1132, 2156, 2556, 65535, 65535, 65535 },
    { 1133, 2157, 2557, 65535, 65535, 65535 },
    { 1134, 2158, 2558, 65535, 65535, 65535 },
    { 1135, 2159, 2559, 65535, 65535, 65535 },
    { 1136, 2160, 2432, 65535, 65535, 65535 },
    { 1137, 2161, 2433, 65535, 65535, 65535 },
    { 1138, 2162, 2434, 65535, 65535, 65535 },
    { 1139, 2163, 2435, 65535, 65535, 65535 },
    { 1140, 2164, 2436, 65535, 65535, 65535 },
    { 1141, 2165, 2437, 65535, 65535, 65535 },
    { 1142, 2166, 2438, 65535, 65535, 65535 },
    { 1143, 2167, 2439, 65535, 65535, 65535 },
    { 1144, 2168, 2440, 65535, 65535, 65535 },
    { 1145, 2169, 2441, 65535, 65535, 65535 },
    { 1146, 2170, 2442, 65535, 65535, 65535 },
    { 1147, 2171, 2443, 65535, 65535, 65535 },
    { 1148, 2172, 2444, 65535, 65535, 65535 },
    { 1149, 2173, 2445, 65535, 65535, 65535 },
    { 1150, 2174, 2446, 65535, 65535, 65535 },
    { 1151, 2175, 2447, 65535, 65535, 65535 },
    { 1152, 2048, 2176, 65535, 65535, 65535 },
    { 1153, 2049, 2177, 65535, 65535, 65535 },
    { 1154, 2050, 2178, 65535, 65535, 65535 },
    { 1155, 2051, 2179, 65535, 65535, 65535 },
    { 1156, 2052, 2180, 65535, 65535, 65535 },
    { 1157, 2053, 2181, 65535, 65535, 65535 },
    { 1158, 2054, 2182, 65535, 65535, 65535 },
    { 1159, 2055, 2183, 65535, 65535, 65535 },
    { 1160, 2056, 2184, 65535, 65535, 65535 },
    { 1161, 2057, 2185, 65535, 65535, 65535 },
    { 1162, 2058, 2186, 65535, 65535, 65535 },
    { 1163, 2059, 2187, 65535, 65535, 65535 },
    { 1164, 2060, 2188, 65535, 65535, 65535 },
    { 1165, 2061, 2189, 65535, 65535, 65535 },
    { 1166, 2062, 2190, 65535, 65535, 65535 },
    { 1167, 2063, 2191, 65535, 65535, 65535 },
    { 1168, 2064, 2192, 65535, 65535, 65535 },
    { 1169, 2065, 2193, 65535, 65535, 65535 },
    { 1170, 2066, 2194, 65535, 65535, 65535 },
    { 1171, 2067, 2195, 65535, 65535, 65535 },
    { 1172, 2068, 2196, 65535, 65535, 65535 },
    { 1173, 2069, 2197, 65535, 65535, 65535 },
    { 1174, 2070, 2198, 65535, 65535, 65535 },
    { 1175, 2071, 2199, 65535, 65535, 65535 },
    { 1176, 2072, 2200, 65535, 65535, 65535 },
    { 1177, 2073, 2201, 65535, 65535, 65535 },
    { 1178, 2074, 2202, 65535, 65535, 65535 },
    { 1179, 2075, 2203, 65535, 65535, 65535 },
    { 1180, 2076, 2204, 65535, 65535, 65535 },
    { 1181, 2077, 2205, 65535, 65535, 65535 },
    { 1182, 2078, 2206, 65535, 65535, 65535 },
    { 1183, 2079, 2207, 65535, 65535, 65535 },
    { 1184, 2080, 2208, 65535, 65535, 65535 },
    { 1185, 2081, 2209, 65535, 65535, 65535 },
    { 1186, 2082, 2210, 65535, 65535, 65535 },
    { 1187, 2083, 2211, 65535, 65535, 65535 },
    { 1188, 2084, 2212, 65535, 65535, 65535 },
    { 1189, 2085, 2213, 65535, 65535, 65535 },
    { 1190, 2086, 2214, 65535, 65535, 65535 },
    { 1191, 2087, 2215, 65535, 65535, 65535 },
    { 1192, 2088, 2216, 65535, 65535, 65535 },
    { 1193, 2089, 2217, 65535, 65535, 65535 },
    { 1194, 2090, 2218, 65535, 65535, 65535 },
    { 1195, 2091, 2219, 65535, 65535, 65535 },
    { 1196, 2092, 2220, 65535, 65535, 65535 },
    { 1197, 2093, 2221, 65535, 65535, 65535 },
    { 1198, 2094, 2222, 65535, 65535, 65535 },
    { 1199, 2095, 2223, 65535, 65535, 65535 },
    { 1200, 2096, 2224, 65535, 65535, 65535 },
    { 1201, 2097, 2225, 65535, 65535, 65535 },
    { 1202, 2098, 2226, 65535, 65535, 65535 },
    { 1203, 2099, 2227, 65535, 65535, 65535 },
    { 1204, 2100, 2228, 65535, 65535, 65535 },
    { 1205, 2101, 2229, 65535, 65535, 65535 },
    { 1206, 2102, 2230, 65535, 65535, 65535 },
    { 1207, 2103, 2231, 65535, 65535, 65535 },
    { 1208, 2104, 2232, 65535, 65535, 65535 },
    { 1209, 2105, 2233, 65535, 65535, 65535 },
    { 1210, 2106, 2234, 65535, 65535, 65535 },
    { 1211, 2107, 2235, 65535, 65535, 65535 },
    { 1212, 2108, 2236, 65535, 65535, 65535 },
    { 1213, 2109, 2237, 65535, 65535, 65535 },
    { 1214, 2110, 2238, 65535, 65535, 65535 },
    { 1215, 2111, 2239, 65535, 65535, 65535 },
    { 1216, 2112, 2240, 65535, 65535, 65535 },
    { 1217, 2113, 2241, 65535, 65535, 65535 },
    { 1218, 2114, 2242, 65535, 65535, 65535 },
    { 1219, 2115, 2243, 65535, 65535, 65535 },
    { 1220, 2116, 2244, 65535, 65535, 65535 },
    { 1221, 2117, 2245, 65535, 65535, 65535 },
    { 1222, 2118, 2246, 65535, 65535, 65535 },
    { 1223, 2119, 2247, 65535, 65535, 65535 },
    { 1224, 2120, 2248, 65535, 65535, 65535 },
    { 1225, 2121, 2249, 65535, 65535, 65535 },
    { 1226, 2122, 2250, 65535, 65535, 65535 },
    { 1227, 2123, 2251, 65535, 65535, 65535 },
    { 1228, 2124, 2252, 65535, 65535, 65535 },
    { 1229, 2125, 2253, 65535, 65535, 65535 },
    { 1230, 2126, 2254, 65535, 65535, 65535 },
    { 1231, 2127, 2255, 65535, 65535, 65535 },
    { 1232, 2128, 2256, 65535, 65535, 65535 },
    { 1233, 2129, 2257, 65535, 65535, 65535 },
    { 1234, 2130, 2258, 65535, 65535, 65535 },
    { 1235, 2131, 2259, 65535, 65535, 65535 },
    { 1236, 2132, 2260, 65535, 65535, 65535 },
    { 1237, 2133, 2261, 65535, 65535, 65535 },
    { 1238, 2134, 2262, 65535, 65535, 65535 },
    { 1239, 2135, 2263, 65535, 65535, 65535 },
    { 1240, 2136, 2264, 65535, 65535, 65535 },
    { 1241, 2137, 2265, 65535, 65535, 65535 },
    { 1242, 2138, 2266, 65535, 65535, 65535 },
    { 1243, 2139, 2267, 65535, 65535, 65535 },
    { 1244, 2140, 2268, 65535, 65535, 65535 },
    { 1245, 2141, 2269, 65535, 65535, 65535 },
    { 1246, 2142, 2270, 65535, 65535, 65535 },
    { 1247, 2143, 2271, 65535, 65535, 65535 },
    { 1248, 2144, 2272, 65535, 65535, 65535 },
    { 1249, 2145, 2273, 65535, 65535, 65535 },
    { 1250, 2146, 2274, 65535, 65535, 65535 },
    { 1251, 2147, 2275, 65535, 65535, 65535 },
    { 1252, 2148, 2276, 65535, 65535, 65535 },
    { 1253, 2149, 2277, 65535, 65535, 65535 },
    { 1254, 2150, 2278, 65535, 65535, 65535 },
    { 1255, 2151, 2279, 65535, 65535, 65535 },
    { 1256, 2152, 2280, 65535, 65535, 65535 },
    { 1257, 2153, 2281, 65535, 65535, 65535 },
    { 1258, 2154, 2282, 65535, 65535, 65535 },
    { 1259, 2155, 2283, 65535, 65535, 65535 },
    { 1260, 2156, 2284, 65535, 65535, 65535 },
    { 1261, 2157, 2285, 65535, 65535, 65535 },
    { 1262, 2158, 2286, 65535, 65535, 65535 },
    { 1263, 2159, 2287, 65535, 65535, 65535 },
    { 1264, 2160, 2288, 65535, 65535, 65535 },
    { 1265, 2161, 2289, 65535, 65535, 65535 },
    { 1266, 2162, 2290, 65535, 65535, 65535 },
    { 1267, 2163, 2291, 65535, 65535, 65535 },
    { 1268, 2164, 2292, 65535, 65535, 65535 },
    { 1269, 2165, 2293, 65535, 65535, 65535 },
    { 1270, 2166, 2294, 65535, 65535, 65535 },
    { 1271, 2167, 2295, 65535, 65535, 65535 },
    { 1272, 2168, 2296, 65535, 65535, 65535 },
    { 1273, 2169, 2297, 65535, 65535, 65535 },
    { 1274, 2170, 2298, 65535, 65535, 65535 },
    { 1275, 2171, 2299, 65535, 65535, 65535 },
    { 1276, 2172, 2300, 65535, 65535, 65535 },
    { 1277, 2173, 2301, 65535, 65535, 65535 },
    { 1278, 2174, 2302, 65535, 65535, 65535 },
    { 1279, 2175, 2303, 65535, 65535, 65535 },
    { 1280, 2176, 2304, 65535, 65535, 65535 },
    { 1281, 2177, 2305, 65535, 65535, 65535 },
    { 1282, 2178, 2306, 65535, 65535, 65535 },
    { 1283, 2179, 2307, 65535, 65535, 65535 },
    { 1284, 2180, 2308, 65535, 65535, 65535 },
    { 1285, 2181, 2309, 65535, 65535, 65535 },
    { 1286, 2182, 2310, 65535, 65535, 65535 },
    { 1287, 2183, 2311, 65535, 65535, 65535 },
    { 1288, 2184, 2312, 65535, 65535, 65535 },
    { 1289, 2185, 2313, 65535, 65535, 65535 },
    { 1290, 2186, 2314, 65535, 65535, 65535 },
    { 1291, 2187, 2315, 65535, 65535, 65535 },
    { 1292, 2188, 2316, 65535, 65535, 65535 },
    { 1293, 2189, 2317, 65535, 65535, 65535 },
    { 1294, 2190, 2318, 65535, 65535, 65535 },
    { 1295, 2191, 2319, 65535, 65535, 65535 },
    { 1296, 2192, 2320, 65535, 65535, 65535 },
    { 1297, 2193, 2321, 65535, 65535, 65535 },
    { 1298, 2194, 2322, 65535, 65535, 65535 },
    { 1299, 2195, 2323, 65535, 65535, 65535 },
    { 1300, 2196, 2324, 65535, 65535, 65535 },
    { 1301, 2197, 2325, 65535, 65535, 65535 },
    { 1302, 2198, 2326, 65535, 65535, 65535 },
    { 1303, 2199, 2327, 65535, 65535, 65535 },
    { 1304, 2200, 2328, 65535, 65535, 65535 },
    { 1305, 2201, 2329, 65535, 65535, 65535 },
    { 1306, 2202, 2330, 65535, 65535, 65535 },
    { 1307, 2203, 2331, 65535, 65535, 65535 },
    { 1308, 2204, 2332, 65535, 65535, 65535 },
    { 1309, 2205, 2333, 65535, 65535, 65535 },
    { 1310, 2206, 2334, 65535, 65535, 65535 },
    { 1311, 2207, 2335, 65535, 65535, 65535 },
    { 1312, 2208, 2336, 65535, 65535, 65535 },
    { 1313, 2209, 2337, 65535, 65535, 65535 },
    { 1314, 2210, 2338, 65535, 65535, 65535 },
    { 1315, 2211, 2339, 65535, 65535, 65535 },
    { 1316, 2212, 2340, 65535, 65535, 65535 },
    { 1317, 2213, 2341, 65535, 65535, 65535 },
    { 1318, 2214, 2342, 65535, 65535, 65535 },
    { 1319, 2215, 2343, 65535, 65535, 65535 },
    { 1320, 2216, 2344, 65535, 65535, 65535 },
    { 1321, 2217, 2345, 65535, 65535, 65535 },
    { 1322, 2218, 2346, 65535, 65535, 65535 },
    { 1323, 2219, 2347, 65535, 65535, 65535 },
    { 1324, 2220, 2348, 65535, 65535, 65535 },
    { 1325, 2221, 2349, 65535, 65535, 65535 },
    { 1326, 2222, 2350, 65535, 65535, 65535 },
    { 1327, 2223, 2351, 65535, 65535, 65535 },
    { 1328, 2224, 2352, 65535, 65535, 65535 },
    { 1329, 2225, 2353, 65535, 65535, 65535 },
    { 1330, 2226, 2354, 65535, 65535, 65535 },
    { 1331, 2227, 2355, 65535, 65535, 65535 },
    { 1332, 2228, 2356, 65535, 65535, 65535 },
    { 1333, 2229, 2357, 65535, 65535, 65535 },
    { 1334, 2230, 2358, 65535, 65535, 65535 },
    { 1335, 2231, 2359, 65535, 65535, 65535 },
    { 1336, 2232, 2360, 65535, 65535, 65535 },
    { 1337, 2233, 2361, 65535, 65535, 65535 },
    { 1338, 2234, 2362, 65535, 65535, 65535 },
    { 1339, 2235, 2363, 65535, 65535, 65535 },
    { 1340, 2236, 2364, 65535, 65535, 65535 },
    { 1341, 2237, 2365, 65535, 65535, 65535 },
    { 1342, 2238, 2366, 65535, 65535, 65535 },
    { 1343, 2239, 2367, 65535, 65535, 65535 },
    { 1344, 2240, 2368, 65535, 65535, 65535 },
    { 1345, 2241, 2369, 65535, 65535, 65535 },
    { 1346, 2242, 2370, 65535, 65535, 65535 },
    { 1347, 2243, 2371, 65535, 65535, 65535 },
    { 1348, 2244, 2372, 65535, 65535, 65535 },
    { 1349, 2245, 2373, 65535, 65535, 65535 },
    { 1350, 2246, 2374, 65535, 65535, 65535 },
    { 1351, 2247, 2375, 65535, 65535, 65535 },
    { 1352, 2248, 2376, 65535, 65535, 65535 },
    { 1353, 2249, 2377, 65535, 65535, 65535 },
    { 1354, 2250, 2378, 65535, 65535, 65535 },
    { 1355, 2251, 2379, 65535, 65535, 65535 },
    { 1356, 2252, 2380, 65535, 65535, 65535 },
    { 1357, 2253, 2381, 65535, 65535, 65535 },
    { 1358, 2254, 2382, 65535, 65535, 65535 },
    { 1359, 2255, 2383, 65535, 65535, 65535 },
    { 1360, 2256, 2384, 65535, 65535, 65535 },
    { 1361, 2257, 2385, 65535, 65535, 65535 },
    { 1362, 2258, 2386, 65535, 65535, 65535 },
    { 1363, 2259, 2387, 65535, 65535, 65535 },
    { 1364, 2260, 2388, 65535, 65535, 65535 },
    { 1365, 2261, 2389, 65535, 65535, 65535 },
    { 1366, 2262, 2390, 65535, 65535, 65535 },
    { 1367, 2263, 2391, 65535, 65535, 65535 },
    { 1368, 2264, 2392, 65535, 65535, 65535 },
    { 1369, 2265, 2393, 65535, 65535, 65535 },
    { 1370, 2266, 2394, 65535, 65535, 65535 },
    { 1371, 2267, 2395, 65535, 65535, 65535 },
    { 1372, 2268, 2396, 65535, 65535, 65535 },
    { 1373, 2269, 2397, 65535, 65535, 65535 },
    { 1374, 2270, 2398, 65535, 65535, 65535 },
    { 1375, 2271, 2399, 65535, 65535, 65535 },
    { 1376, 2272, 2400, 65535, 65535, 65535 },
    { 1377, 2273, 2401, 65535, 65535, 65535 },
    { 1378, 2274, 2402, 65535, 65535, 65535 },
    { 1379, 2275, 2403, 65535, 65535, 65535 },
    { 1380, 2276, 2404, 65535, 65535, 65535 },
    { 1381, 2277, 2405, 65535, 65535, 65535 },
    { 1382, 2278, 2406, 65535, 65535, 65535 },
    { 1383, 2279, 2407, 65535, 65535, 65535 },
    { 1384, 2280, 2408, 65535, 65535, 65535 },
    { 1385, 2281, 2409, 65535, 65535, 65535 },
    { 1386, 2282, 2410, 65535, 65535, 65535 },
    { 1387, 2283, 2411, 65535, 65535, 65535 },
    { 1388, 2284, 2412, 65535, 65535, 65535 },
    { 1389, 2285, 2413, 65535, 65535, 65535 },
    { 1390, 2286, 2414, 65535, 65535, 65535 },
    { 1391, 2287, 2415, 65535, 65535, 65535 },
    { 1392, 2288, 2416, 65535, 65535, 65535 },
    { 1393, 2289, 2417, 65535, 65535, 65535 },
    { 1394, 2290, 2418, 65535, 65535, 65535 },
    { 1395, 2291, 2419, 65535, 65535, 65535 },
    { 1396, 2292, 2420, 65535, 65535, 65535 },
    { 1397, 2293, 2421, 65535, 65535, 65535 },
    { 1398, 2294, 2422, 65535, 65535, 65535 },
    { 1399, 2295, 2423, 65535, 65535, 65535 },
    { 1400, 2296, 2424, 65535, 65535, 65535 },
    { 1401, 2297, 2425, 65535, 65535, 65535 },
    { 1402, 2298, 2426, 65535, 65535, 65535 },
    { 1403, 2299, 2427, 65535, 65535, 65535 },
    { 1404, 2300, 2428, 65535, 65535, 65535 },
    { 1405, 2301, 2429, 65535, 65535, 65535 },
    { 1406, 2302, 2430, 65535, 65535, 65535 },
    { 1407, 2303, 2431, 65535, 65535, 65535 },
    { 1408, 2304, 2432, 65535, 65535, 65535 },
    { 1409, 2305, 2433, 65535, 65535, 65535 },
    { 1410, 2306, 2434, 65535, 65535, 65535 },
    { 1411, 2307, 2435, 65535, 65535, 65535 },
    { 1412, 2308, 2436, 65535, 65535, 65535 },
    { 1413, 2309, 2437, 65535, 65535, 65535 },
    { 1414, 2310, 2438, 65535, 65535, 65535 },
    { 1415, 2311, 2439, 65535, 65535, 65535 },
    { 1416, 2312, 2440, 65535, 65535, 65535 },
    { 1417, 2313, 2441, 65535, 65535, 65535 },
    { 1418, 2314, 2442, 65535, 65535, 65535 },
    { 1419, 2315, 2443, 65535, 65535, 65535 },
    { 1420, 2316, 2444, 65535, 65535, 65535 },
    { 1421, 2317, 2445, 65535, 65535, 65535 },
    { 1422, 2318, 2446, 65535, 65535, 65535 },
    { 1423, 2319, 2447, 65535, 65535, 65535 },
    { 1424, 2320, 2448, 65535, 65535, 65535 },
    { 1425, 2321, 2449, 65535, 65535, 65535 },
    { 1426, 2322, 2450, 65535, 65535, 65535 },
    { 1427, 2323, 2451, 65535, 65535, 65535 },
    { 1428, 2324, 2452, 65535, 65535, 65535 },
    { 1429, 2325, 2453, 65535, 65535, 65535 },
    { 1430, 2326, 2454, 65535, 65535, 65535 },
    { 1431, 2327, 2455, 65535, 65535, 65535 },
    { 1432, 2328, 2456, 65535, 65535, 65535 },
    { 1433, 2329, 2457, 65535, 65535, 65535 },
    { 1434, 2330, 2458, 65535, 65535, 65535 },
    { 1435, 2331, 2459, 65535, 65535, 65535 },
    { 1436, 2332, 2460, 65535, 65535, 65535 },
    { 1437, 2333, 2461, 65535, 65535, 65535 },
    { 1438, 2334, 2462, 65535, 65535, 65535 },
    { 1439, 2335, 2463, 65535, 65535, 65535 },
    { 1440, 2336, 2464, 65535, 65535, 65535 },
    { 1441, 2337, 2465, 65535, 65535, 65535 },
    { 1442, 2338, 2466, 65535, 65535, 65535 },
    { 1443, 2339, 2467, 65535, 65535, 65535 },
    { 1444, 2340, 2468, 65535, 65535, 65535 },
    { 1445, 2341, 2469, 65535, 65535, 65535 },
    { 1446, 2342, 2470, 65535, 65535, 65535 },
    { 1447, 2343, 2471, 65535, 65535, 65535 },
    { 1448, 2344, 2472, 65535, 65535, 65535 },
    { 1449, 2345, 2473, 65535, 65535, 65535 },
    { 1450, 2346, 2474, 65535, 65535, 65535 },
    { 1451, 2347, 2475, 65535, 65535, 65535 },
    { 1452, 2348, 2476, 65535, 65535, 65535 },
    { 1453, 2349, 2477, 65535, 65535, 65535 },
    { 1454, 2350, 2478, 65535, 65535, 65535 },
    { 1455, 2351, 2479, 65535, 65535, 65535 },
    { 1456, 2352, 2480, 65535, 65535, 65535 },
    { 1457, 2353, 2481, 65535, 65535, 65535 },
    { 1458, 2354, 2482, 65535, 65535, 65535 },
    { 1459, 2355, 2483, 65535, 65535, 65535 },
    { 1460, 2356, 2484, 65535, 65535, 65535 },
    { 1461, 2357, 2485, 65535, 65535, 65535 },
    { 1462, 2358, 2486, 65535, 65535, 65535 },
    { 1463, 2359, 2487, 65535, 65535, 65535 },
    { 1464, 2360, 2488, 65535, 65535, 65535 },
    { 1465, 2361, 2489, 65535, 65535, 65535 },
    { 1466, 2362, 2490, 65535, 65535, 65535 },
    { 1467, 2363, 2491, 65535, 65535, 65535 },
    { 1468, 2364, 2492, 65535, 65535, 65535 },
    { 1469, 2365, 2493, 65535, 65535, 65535 },
    { 1470, 2366, 2494, 65535, 65535, 65535 },
    { 1471, 2367, 2495, 65535, 65535, 65535 },
    { 1472, 2368, 2496, 65535, 65535, 65535 },
    { 1473, 2369, 2497, 65535, 65535, 65535 },
    { 1474, 2370, 2498, 65535, 65535, 65535 },
    { 1475, 2371, 2499, 65535, 65535, 65535 },
    { 1476, 2372, 2500, 65535, 65535, 65535 },
    { 1477, 2373, 2501, 65535, 65535, 65535 },
    { 1478, 2374, 2502, 65535, 65535, 65535 },
    { 1479, 2375, 2503, 65535, 65535, 65535 },
    { 1480, 2376, 2504, 65535, 65535, 65535 },
    { 1481, 2377, 2505, 65535, 65535, 65535 },
    { 1482, 2378, 2506, 65535, 65535, 65535 },
    { 1483, 2379, 2507, 65535, 65535, 65535 },
    { 1484, 2380, 2508, 65535, 65535, 65535 },
    { 1485, 2381, 2509, 65535, 65535, 65535 },
    { 1486, 2382, 2510, 65535, 65535, 65535 },
    { 1487, 2383, 2511, 65535, 65535, 65535 },
    { 1488, 2384, 2512, 65535, 65535, 65535 },
    { 1489, 2385, 2513, 65535, 65535, 65535 },
    { 1490, 2386, 2514, 65535, 65535, 65535 },
    { 1491, 2387, 2515, 65535, 65535, 65535 },
    { 1492, 2388, 2516, 65535, 65535, 65535 },
    { 1493, 2389, 2517, 65535, 65535, 65535 },
    { 1494, 2390, 2518, 65535, 65535, 65535 },
    { 1495, 2391, 2519, 65535, 65535, 65535 },
    { 1496, 2392, 2520, 65535, 65535, 65535 },
    { 1497, 2393, 2521, 65535, 65535, 65535 },
    { 1498, 2394, 2522, 65535, 65535, 65535 },
    { 1499, 2395, 2523, 65535, 65535, 65535 },
    { 1500, 2396, 2524, 65535, 65535, 65535 },
    { 1501, 2397, 2525, 65535, 65535, 65535 },
    { 1502, 2398, 2526, 65535, 65535, 65535 },
    { 1503, 2399, 2527, 65535, 65535, 65535 },
    { 1504, 2400, 2528, 65535, 65535, 65535 },
    { 1505, 2401, 2529, 65535, 65535, 65535 },
    { 1506, 2402, 2530, 65535, 65535, 65535 },
    { 1507, 2403, 2531, 65535, 65535, 65535 },
    { 1508, 2404, 2532, 65535, 65535, 65535 },
    { 1509, 2405, 2533, 65535, 65535, 65535 },
    { 1510, 2406, 2534, 65535, 65535, 65535 },
    { 1511, 2407, 2535, 65535, 65535, 65535 },
    { 1512, 2408, 2536, 65535, 65535, 65535 },
    { 1513, 2409, 2537, 65535, 65535, 65535 },
    { 1514, 2410, 2538, 65535, 65535, 65535 },
    { 1515, 2411, 2539, 65535, 65535, 65535 },
    { 1516, 2412, 2540, 65535, 65535, 65535 },
    { 1517, 2413, 2541, 65535, 65535, 65535 },
    { 1518, 2414, 2542, 65535, 65535, 65535 },
    { 1519, 2415, 2543, 65535, 65535, 65535 },
    { 1520, 2416, 2544, 65535, 65535, 65535 },
    { 1521, 2417, 2545, 65535, 65535, 65535 },
    { 1522, 2418, 2546, 65535, 65535, 65535 },
    { 1523, 2419, 2547, 65535, 65535, 65535 },
    { 1524, 2420, 2548, 65535, 65535, 65535 },
    { 1525, 2421, 2549, 65535, 65535, 65535 },
    { 1526, 2422, 2550, 65535, 65535, 65535 },
    { 1527, 2423, 2551, 65535, 65535, 65535 },
    { 1528, 2424, 2552, 65535, 65535, 65535 },
    { 1529, 2425, 2553, 65535, 65535, 65535 },
    { 1530, 2426, 2554, 65535, 65535, 65535 },
    { 1531, 2427, 2555, 65535, 65535, 65535 },
    { 1532, 2428, 2556, 65535, 65535, 65535 },
    { 1533, 2429, 2557, 65535, 65535, 65535 },
    { 1534, 2430, 2558, 65535, 65535, 65535 },
    { 1535, 2431, 2559, 65535, 65535, 65535 },
    {    0,  512, 1536, 2151, 2281, 2304 },
    {    1,  513, 1537, 2152, 2282, 2305 },
    {    2,  514, 1538, 2153, 2283, 2306 },
    {    3,  515, 1539, 2154, 2284, 2307 },
    {    4,  516, 1540, 2155, 2285, 2308 },
    {    5,  517, 1541, 2156, 2286, 2309 },
    {    6,  518, 1542, 2157, 2287, 2310 },
    {    7,  519, 1543, 2158, 2288, 2311 },
    {    8,  520, 1544, 2159, 2289, 2312 },
    {    9,  521, 1545, 2160, 2290, 2313 },
    {   10,  522, 1546, 2161, 2291, 2314 },
    {   11,  523, 1547, 2162, 2292, 2315 },
    {   12,  524, 1548, 2163, 2293, 2316 },
    {   13,  525, 1549, 2164, 2294, 2317 },
    {   14,  526, 1550, 2165, 2295, 2318 },
    {   15,  527, 1551, 2166, 2296, 2319 },
    {   16,  528, 1552, 2167, 2297, 2320 },
    {   17,  529, 1553, 2168, 2298, 2321 },
    {   18,  530, 1554, 2169, 2299, 2322 },
    {   19,  531, 1555, 2170, 2300, 2323 },
    {   20,  532, 1556, 2171, 2301, 2324 },
    {   21,  533, 1557, 2172, 2302, 2325 },
    {   22,  534, 1558, 2173, 2303, 2326 },
    {   23,  535, 1559, 2174, 2176, 2327 },
    {   24,  536, 1560, 2175, 2177, 2328 },
    {   25,  537, 1561, 2048, 2178, 2329 },
    {   26,  538, 1562, 2049, 2179, 2330 },
    {   27,  539, 1563, 2050, 2180, 2331 },
    {   28,  540, 1564, 2051, 2181, 2332 },
    {   29,  541, 1565, 2052, 2182, 2333 },
    {   30,  542, 1566, 2053, 2183, 2334 },
    {   31,  543, 1567, 2054, 2184, 2335 },
    {   32,  544, 1568, 2055, 2185, 2336 },
    {   33,  545, 1569, 2056, 2186, 2337 },
    {   34,  546, 1570, 2057, 2187, 2338 },
    {   35,  547, 1571, 2058, 2188, 2339 },
    {   36,  548, 1572, 2059, 2189, 2340 },
    {   37,  549, 1573, 2060, 2190, 2341 },
    {   38,  550, 1574, 2061, 2191, 2342 },
    {   39,  551, 1575, 2062, 2192, 2343 },
    {   40,  552, 1576, 2063, 2193, 2344 },
    {   41,  553, 1577, 2064, 2194, 2345 },
    {   42,  554, 1578, 2065, 2195, 2346 },
    {   43,  555, 1579, 2066, 2196, 2347 },
    {   44,  556, 1580, 2067, 2197, 2348 },
    {   45,  557, 1581, 2068, 2198, 2349 },
    {   46,  558, 1582, 2069, 2199, 2350 },
    {   47,  559, 1583, 2070, 2200, 2351 },
    {   48,  560, 1584, 2071, 2201, 2352 },
    {   49,  561, 1585, 2072, 2202, 2353 },
    {   50,  562, 1586, 2073, 2203, 2354 },
    {   51,  563, 1587, 2074, 2204, 2355 },
    {   52,  564, 1588, 2075, 2205, 2356 },
    {   53,  565, 1589, 2076, 2206, 2357 },
    {   54,  566, 1590, 2077, 2207, 2358 },
    {   55,  567, 1591, 2078, 2208, 2359 },
    {   56,  568, 1592, 2079, 2209, 2360 },
    {   57,  569, 1593, 2080, 2210, 2361 },
    {   58,  570, 1594, 2081, 2211, 2362 },
    {   59,  571, 1595, 2082, 2212, 2363 },
    {   60,  572, 1596, 2083, 2213, 2364 },
    {   61,  573, 1597, 2084, 2214, 2365 },
    {   62,  574, 1598, 2085, 2215, 2366 },
    {   63,  575, 1599, 2086, 2216, 2367 },
    {   64,  576, 1600, 2087, 2217, 2368 },
    {   65,  577, 1601, 2088, 2218, 2369 },
    {   66,  578, 1602, 2089, 2219, 2370 },
    {   67,  579, 1603, 2090, 2220, 2371 },
    {   68,  580, 1604, 2091, 2221, 2372 },
    {   69,  581, 1605, 2092, 2222, 2373 },
    {   70,  582, 1606, 2093, 2223, 2374 },
    {   71,  583, 1607, 2094, 2224, 2375 },
    {   72,  584, 1608, 2095, 2225, 2376 },
    {   73,  585, 1609, 2096, 2226, 2377 },
    {   74,  586, 1610, 2097, 2227, 2378 },
    {   75,  587, 1611, 2098, 2228, 2379 },
    {   76,  588, 1612, 2099, 2229, 2380 },
    {   77,  589, 1613, 2100, 2230, 2381 },
    {   78,  590, 1614, 2101, 2231, 2382 },
    {   79,  591, 1615, 2102, 2232, 2383 },
    {   80,  592, 1616, 2103, 2233, 2384 },
    {   81,  593, 1617, 2104, 2234, 2385 },
    {   82,  594, 1618, 2105, 2235, 2386 },
    {   83,  595, 1619, 2106, 2236, 2387 },
    {   84,  596, 1620, 2107, 2237, 2388 },
    {   85,  597, 1621, 2108, 2238, 2389 },
    {   86,  598, 1622, 2109, 2239, 2390 },
    {   87,  599, 1623, 2110, 2240, 2391 },
    {   88,  600, 1624, 2111, 2241, 2392 },
    {   89,  601, 1625, 2112, 2242, 2393 },
    {   90,  602, 1626, 2113, 2243, 2394 },
    {   91,  603, 1627, 2114, 2244, 2395 },
    {   92,  604, 1628, 2115, 2245, 2396 },
    {   93,  605, 1629, 2116, 2246, 2397 },
    {   94,  606, 1630, 2117, 2247, 2398 },
    {   95,  607, 1631, 2118, 2248, 2399 },
    {   96,  608, 1632, 2119, 2249, 2400 },
    {   97,  609, 1633, 2120, 2250, 2401 },
    {   98,  610, 1634, 2121, 2251, 2402 },
    {   99,  611, 1635, 2122, 2252, 2403 },
    {  100,  612, 1636, 2123, 2253, 2404 },
    {  101,  613, 1637, 2124, 2254, 2405 },
    {  102,  614, 1638, 2125, 2255, 2406 },
    {  103,  615, 1639, 2126, 2256, 2407 },
    {  104,  616, 1640, 2127, 2257, 2408 },
    {  105,  617, 1641, 2128, 2258, 2409 },
    {  106,  618, 1642, 2129, 2259, 2410 },
    {  107,  619, 1643, 2130, 2260, 2411 },
    {  108,  620, 1644, 2131, 2261, 2412 },
    {  109,  621, 1645, 2132, 2262, 2413 },
    {  110,  622, 1646, 2133, 2263, 2414 },
    {  111,  623, 1647, 2134, 2264, 2415 },
    {  112,  624, 1648, 2135, 2265, 2416 },
    {  113,  625, 1649, 2136, 2266, 2417 },
    {  114,  626, 1650, 2137, 2267, 2418 },
    {  115,  627, 1651, 2138, 2268, 2419 },
    {  116,  628, 1652, 2139, 2269, 2420 },
    {  117,  629, 1653, 2140, 2270, 2421 },
    {  118,  630, 1654, 2141, 2271, 2422 },
    {  119,  631, 1655, 2142, 2272, 2423 },
    {  120,  632, 1656, 2143, 2273, 2424 },
    {  121,  633, 1657, 2144, 2274, 2425 },
    {  122,  634, 1658, 2145, 2275, 2426 },
    {  123,  635, 1659, 2146, 2276, 2427 },
    {  124,  636, 1660, 2147, 2277, 2428 },
    {  125,  637, 1661, 2148, 2278, 2429 },
    {  126,  638, 1662, 2149, 2279, 2430 },
    {  127,  639, 1663, 2150, 2280, 2431 },
    {  128,  640, 1664, 2229, 2378, 2477 },
    {  129,  641, 1665, 2230, 2379, 2478 },
    {  130,  642, 1666, 2231, 2380, 2479 },
    {  131,  643, 1667, 2232, 2381, 2480 },
    {  132,  644, 1668, 2233, 2382, 2481 },
    {  133,  645, 1669, 2234, 2383, 2482 },
    {  134,  646, 1670, 2235, 2384, 2483 },
    {  135,  647, 1671, 2236, 2385, 2484 },
    {  136,  648, 1672, 2237, 2386, 2485 },
    {  137,  649, 1673, 2238, 2387, 2486 },
    {  138,  650, 1674, 2239, 2388, 2487 },
    {  139,  651, 1675, 2240, 2389, 2488 },
    {  140,  652, 1676, 2241, 2390, 2489 },
    {  141,  653, 1677, 2242, 2391, 2490 },
    {  142,  654, 1678, 2243, 2392, 2491 },
    {  143,  655, 1679, 2244, 2393, 2492 },
    {  144,  656, 1680, 2245, 2394, 2493 },
    {  145,  657, 1681, 2246, 2395, 2494 },
    {  146,  658, 1682, 2247, 2396, 2495 },
    {  147,  659, 1683, 2248, 2397, 2496 },
    {  148,  660, 1684, 2249, 2398, 2497 },
    {  149,  661, 1685, 2250, 2399, 2498 },
    {  150,  662, 1686, 2251, 2400, 2499 },
    {  151,  663, 1687, 2252, 2401, 2500 },
    {  152,  664, 1688, 2253, 2402, 2501 },
    {  153,  665, 1689, 2254, 2403, 2502 },
    {  154,  666, 1690, 2255, 2404, 2503 },
    {  155,  667, 1691, 2256, 2405, 2504 },
    {  156,  668, 1692, 2257, 2406, 2505 },
    {  157,  669, 1693, 2258, 2407, 2506 },
    {  158,  670, 1694, 2259, 2408, 2507 },
    {  159,  671, 1695, 2260, 2409, 2508 },
    {  160,  672, 1696, 2261, 2410, 2509 },
    {  161,  673, 1697, 2262, 2411, 2510 },
    {  162,  674, 1698, 2263, 2412, 2511 },
    {  163,  675, 1699, 2264, 2413, 2512 },
    {  164,  676, 1700, 2265, 2414, 2513 },
    {  165,  677, 1701, 2266, 2415, 2514 },
    {  166,  678, 1702, 2267, 2416, 2515 },
    {  167,  679, 1703, 2268, 2417, 2516 },
    {  168,  680, 1704, 2269, 2418, 2517 },
    {  169,  681, 1705, 2270, 2419, 2518 },
    {  170,  682, 1706, 2271, 2420, 2519 },
    {  171,  683, 1707, 2272, 2421, 2520 },
    {  172,  684, 1708, 2273, 2422, 2521 },
    {  173,  685, 1709, 2274, 2423, 2522 },
    {  174,  686, 1710, 2275, 2424, 2523 },
    {  175,  687, 1711, 2276, 2425, 2524 },
    {  176,  688, 1712, 2277, 2426, 2525 },
    {  177,  689, 1713, 2278, 2427, 2526 },
    {  178,  690, 1714, 2279, 2428, 2527 },
    {  179,  691, 1715, 2280, 2429, 2528 },
    {  180,  692, 1716, 2281, 2430, 2529 },
    {  181,  693, 1717, 2282, 2431, 2530 },
    {  182,  694, 1718, 2283, 2304, 2531 },
    {  183,  695, 1719, 2284, 2305, 2532 },
    {  184,  696, 1720, 2285, 2306, 2533 },
    {  185,  697, 1721, 2286, 2307, 2534 },
    {  186,  698, 1722, 2287, 2308, 2535 },
    {  187,  699, 1723, 2288, 2309, 2536 },
    {  188,  700, 1724, 2289, 2310, 2537 },
    {  189,  701, 1725, 2290, 2311, 2538 },
    {  190,  702, 1726, 2291, 2312, 2539 },
    {  191,  703, 1727, 2292, 2313, 2540 },
    {  192,  704, 1728, 2293, 2314, 2541 },
    {  193,  705, 1729, 2294, 2315, 2542 },
    {  194,  706, 1730, 2295, 2316, 2543 },
    {  195,  707, 1731, 2296, 2317, 2544 },
    {  196,  708, 1732, 2297, 2318, 2545 },
    {  197,  709, 1733, 2298, 2319, 2546 },
    {  198,  710, 1734, 2299, 2320, 2547 },
    {  199,  711, 1735, 2300, 2321, 2548 },
    {  200,  712, 1736, 2301, 2322, 2549 },
    {  201,  713, 1737, 2302, 2323, 2550 },
    {  202,  714, 1738, 2303, 2324, 2551 },
    {  203,  715, 1739, 2176, 2325, 2552 },
    {  204,  716, 1740, 2177, 2326, 2553 },
    {  205,  717, 1741, 2178, 2327, 2554 },
    {  206,  718, 1742, 2179, 2328, 2555 },
    {  207,  719, 1743, 2180, 2329, 2556 },
    {  208,  720, 1744, 2181, 2330, 2557 },
    {  209,  721, 1745, 2182, 2331, 2558 },
    {  210,  722, 1746, 2183, 2332, 2559 },
    {  211,  723, 1747, 2184, 2333, 2432 },
    {  212,  724, 1748, 2185, 2334, 2433 },
    {  213,  725, 1749, 2186, 2335, 2434 },
    {  214,  726, 1750, 2187, 2336, 2435 },
    {  215,  727, 1751, 2188, 2337, 2436 },
    {  216,  728, 1752, 2189, 2338, 2437 },
    {  217,  729, 1753, 2190, 2339, 2438 },
    {  218,  730, 1754, 2191, 2340, 2439 },
    {  219,  731, 1755, 2192, 2341, 2440 },
    {  220,  732, 1756, 2193, 2342, 2441 },
    {  221,  733, 1757, 2194, 2343, 2442 },
    {  222,  734, 1758, 2195, 2344, 2443 },
    {  223,  735, 1759, 2196, 2345, 2444 },
    {  224,  736, 1760, 2197, 2346, 2445 },
    {  225,  737, 1761, 2198, 2347, 2446 },
    {  226,  738, 1762, 2199, 2348, 2447 },
    {  227,  739, 1763, 2200, 2349, 2448 },
    {  228,  740, 1764, 2201, 2350, 2449 },
    {  229,  741, 1765, 2202, 2351, 2450 },
    {  230,  742, 1766, 2203, 2352, 2451 },
    {  231,  743, 1767, 2204, 2353, 2452 },
    {  232,  744, 1768, 2205, 2354, 2453 },
    {  233,  745, 1769, 2206, 2355, 2454 },
    {  234,  746, 1770, 2207, 2356, 2455 },
    {  235,  747, 1771, 2208, 2357, 2456 },
    {  236,  748, 1772, 2209, 2358, 2457 },
    {  237,  749, 1773, 2210, 2359, 2458 },
    {  238,  750, 1774, 2211, 2360, 2459 },
    {  239,  751, 1775, 2212, 2361, 2460 },
    {  240,  752, 1776, 2213, 2362, 2461 },
    {  241,  753, 1777, 2214, 2363, 2462 },
    {  242,  754, 1778, 2215, 2364, 2463 },
    {  243,  755, 1779, 2216, 2365, 2464 },
    {  244,  756, 1780, 2217, 2366, 2465 },
    {  245,  757, 1781, 2218, 2367, 2466 },
    {  246,  758, 1782, 2219, 2368, 2467 },
    {  247,  759, 1783, 2220, 2369, 2468 },
    {  248,  760, 1784, 2221, 2370, 2469 },
    {  249,  761, 1785, 2222, 2371, 2470 },
    {  250,  762, 1786, 2223, 2372, 2471 },
    {  251,  763, 1787, 2224, 2373, 2472 },
    {  252,  764, 1788, 2225, 2374, 2473 },
    {  253,  765, 1789, 2226, 2375, 2474 },
    {  254,  766, 1790, 2227, 2376, 2475 },
    {  255,  767, 1791, 2228, 2377, 2476 },
    {  256,  768, 1792, 2137, 2312, 2551 },
    {  257,  769, 1793, 2138, 2313, 2552 },
    {  258,  770, 1794, 2139, 2314, 2553 },
    {  259,  771, 1795, 2140, 2315, 2554 },
    {  260,  772, 1796, 2141, 2316, 2555 },
    {  261,  773, 1797, 2142, 2317, 2556 },
    {  262,  774, 1798, 2143, 2318, 2557 },
    {  263,  775, 1799, 2144, 2319, 2558 },
    {  264,  776, 1800, 2145, 2320, 2559 },
    {  265,  777, 1801, 2146, 2321, 2432 },
    {  266,  778, 1802, 2147, 2322, 2433 },
    {  267,  779, 1803, 2148, 2323, 2434 },
    {  268,  780, 1804, 2149, 2324, 2435 },
    {  269,  781, 1805, 2150, 2325, 2436 },
    {  270,  782, 1806, 2151, 2326, 2437 },
    {  271,  783, 1807, 2152, 2327, 2438 },
    {  272,  784, 1808, 2153, 2328, 2439 },
    {  273,  785, 1809, 2154, 2329, 2440 },
    {  274,  786, 1810, 2155, 2330, 2441 },
    {  275,  787, 1811, 2156, 2331, 2442 },
    {  276,  788, 1812, 2157, 2332, 2443 },
    {  277,  789, 1813, 2158, 2333, 2444 },
    {  278,  790, 1814, 2159, 2334, 2445 },
    {  279,  791, 1815, 2160, 2335, 2446 },
    {  280,  792, 1816, 2161, 2336, 2447 },
    {  281,  793, 1817, 2162, 2337, 2448 },
    {  282,  794, 1818, 2163, 2338, 2449 },
    {  283,  795, 1819, 2164, 2339, 2450 },
    {  284,  796, 1820, 2165, 2340, 2451 },
    {  285,  797, 1821, 2166, 2341, 2452 },
    {  286,  798, 1822, 2167, 2342, 2453 },
    {  287,  799, 1823, 2168, 2343, 2454 },
    {  288,  800, 1824, 2169, 2344, 2455 },
    {  289,  801, 1825, 2170, 2345, 2456 },
    {  290,  802, 1826, 2171, 2346, 2457 },
    {  291,  803, 1827, 2172, 2347, 2458 },
    {  292,  804, 1828, 2173, 2348, 2459 },
    {  293,  805, 1829, 2174, 2349, 2460 },
    {  294,  806, 1830, 2175, 2350, 2461 },
    {  295,  807, 1831, 2048, 2351, 2462 },
    {  296,  808, 1832, 2049, 2352, 2463 },
    {  297,  809, 1833, 2050, 2353, 2464 },
    {  298,  810, 1834, 2051, 2354, 2465 },
    {  299,  811, 1835, 2052, 2355, 2466 },
    {  300,  812, 1836, 2053, 2356, 2467 },
    {  301,  813, 1837, 2054, 2357, 2468 },
    {  302,  814, 1838, 2055, 2358, 2469 },
    {  303,  815, 1839, 2056, 2359, 2470 },
    {  304,  816, 1840, 2057, 2360, 2471 },
    {  305,  817, 1841, 2058, 2361, 2472 },
    {  306,  818, 1842, 2059, 2362, 2473 },
    {  307,  819, 1843, 2060, 2363, 2474 },
    {  308,  820, 1844, 2061, 2364, 2475 },
    {  309,  821, 1845, 2062, 2365, 2476 },
    {  310,  822, 1846, 2063, 2366, 2477 },
    {  311,  823, 1847, 2064, 2367, 2478 },
    {  312,  824, 1848, 2065, 2368, 2479 },
    {  313,  825, 1849, 2066, 2369, 2480 },
    {  314,  826, 1850, 2067, 2370, 2481 },
    {  315,  827, 1851, 2068, 2371, 2482 },
    {  316,  828, 1852, 2069, 2372, 2483 },
    {  317,  829, 1853, 2070, 2373, 2484 },
    {  318,  830, 1854, 2071, 2374, 2485 },
    {  319,  831, 1855, 2072, 2375, 2486 },
    {  320,  832, 1856, 2073, 2376, 2487 },
    {  321,  833, 1857, 2074, 2377, 2488 },
    {  322,  834, 1858, 2075, 2378, 2489 },
    {  323,  835, 1859, 2076, 2379, 2490 },
    {  324,  836, 1860, 2077, 2380, 2491 },
    {  325,  837, 1861, 2078, 2381, 2492 },
    {  326,  838, 1862, 2079, 2382, 2493 },
    {  327,  839, 1863, 2080, 2383, 2494 },
    {  328,  840, 1864, 2081, 2384, 2495 },
    {  329,  841, 1865, 2082, 2385, 2496 },
    {  330,  842, 1866, 2083, 2386, 2497 },
    {  331,  843, 1867, 2084, 2387, 2498 },
    {  332,  844, 1868, 2085, 2388, 2499 },
    {  333,  845, 1869, 2086, 2389, 2500 },
    {  334,  846, 1870, 2087, 2390, 2501 },
    {  335,  847, 1871, 2088, 2391, 2502 },
    {  336,  848, 1872, 2089, 2392, 2503 },
    {  337,  849, 1873, 2090, 2393, 2504 },
    {  338,  850, 1874, 2091, 2394, 2505 },
    {  339,  851, 1875, 2092, 2395, 2506 },
    {  340,  852, 1876, 2093, 2396, 2507 },
    {  341,  853, 1877, 2094, 2397, 2508 },
    {  342,  854, 1878, 2095, 2398, 2509 },
    {  343,  855, 1879, 2096, 2399, 2510 },
    {  344,  856, 1880, 2097, 2400, 2511 },
    {  345,  857, 1881, 2098, 2401, 2512 },
    {  346,  858, 1882, 2099, 2402, 2513 },
    {  347,  859, 1883, 2100, 2403, 2514 },
    {  348,  860, 1884, 2101, 2404, 2515 },
    {  349,  861, 1885, 2102, 2405, 2516 },
    {  350,  862, 1886, 2103, 2406, 2517 },
    {  351,  863, 1887, 2104, 2407, 2518 },
    {  352,  864, 1888, 2105, 2408, 2519 },
    {  353,  865, 1889, 2106, 2409, 2520 },
    {  354,  866, 1890, 2107, 2410, 2521 },
    {  355,  867, 1891, 2108, 2411, 2522 },
    {  356,  868, 1892, 2109, 2412, 2523 },
    {  357,  869, 1893, 2110, 2413, 2524 },
    {  358,  870, 1894, 2111, 2414, 2525 },
    {  359,  871, 1895, 2112, 2415, 2526 },
    {  360,  872, 1896, 2113, 2416, 2527 },
    {  361,  873, 1897, 2114, 2417, 2528 },
    {  362,  874, 1898, 2115, 2418, 2529 },
    {  363,  875, 1899, 2116, 2419, 2530 },
    {  364,  876, 1900, 2117, 2420, 2531 },
    {  365,  877, 1901, 2118, 2421, 2532 },
    {  366,  878, 1902, 2119, 2422, 2533 },
    {  367,  879, 1903, 2120, 2423, 2534 },
    {  368,  880, 1904, 2121, 2424, 2535 },
    {  369,  881, 1905, 2122, 2425, 2536 },
    {  370,  882, 1906, 2123, 2426, 2537 },
    {  371,  883, 1907, 2124, 2427, 2538 },
    {  372,  884, 1908, 2125, 2428, 2539 },
    {  373,  885, 1909, 2126, 2429, 2540 },
    {  374,  886, 1910, 2127, 2430, 2541 },
    {  375,  887, 1911, 2128, 2431, 2542 },
    {  376,  888, 1912, 2129, 2304, 2543 },
    {  377,  889, 1913, 2130, 2305, 2544 },
    {  378,  890, 1914, 2131, 2306, 2545 },
    {  379,  891, 1915, 2132, 2307, 2546 },
    {  380,  892, 1916, 2133, 2308, 2547 },
    {  381,  893, 1917, 2134, 2309, 2548 },
    {  382,  894, 1918, 2135, 2310, 2549 },
    {  383,  895, 1919, 2136, 2311, 2550 },
    {  384,  896, 1920, 2145, 2288, 2467 },
    {  385,  897, 1921, 2146, 2289, 2468 },
    {  386,  898, 1922, 2147, 2290, 2469 },
    {  387,  899, 1923, 2148, 2291, 2470 },
    {  388,  900, 1924, 2149, 2292, 2471 },
    {  389,  901, 1925, 2150, 2293, 2472 },
    {  390,  902, 1926, 2151, 2294, 2473 },
    {  391,  903, 1927, 2152, 2295, 2474 },
    {  392,  904, 1928, 2153, 2296, 2475 },
    {  393,  905, 1929, 2154, 2297, 2476 },
    {  394,  906, 1930, 2155, 2298, 2477 },
    {  395,  907, 1931, 2156, 2299, 2478 },
    {  396,  908, 1932, 2157, 2300, 2479 },
    {  397,  909, 1933, 2158, 2301, 2480 },
    {  398,  910, 1934, 2159, 2302, 2481 },
    {  399,  911, 1935, 2160, 2303, 2482 },
    {  400,  912, 1936, 2161, 2176, 2483 },
    {  401,  913, 1937, 2162, 2177, 2484 },
    {  402,  914, 1938, 2163, 2178, 2485 },
    {  403,  915, 1939, 2164, 2179, 2486 },
    {  404,  916, 1940, 2165, 2180, 2487 },
    {  405,  917, 1941, 2166, 2181, 2488 },
    {  406,  918, 1942, 2167, 2182, 2489 },
    {  407,  919, 1943, 2168, 2183, 2490 },
    {  408,  920, 1944, 2169, 2184, 2491 },
    {  409,  921, 1945, 2170, 2185, 2492 },
    {  410,  922, 1946, 2171, 2186, 2493 },
    {  411,  923, 1947, 2172, 2187, 2494 },
    {  412,  924, 1948, 2173, 2188, 2495 },
    {  413,  925, 1949, 2174, 2189, 2496 },
    {  414,  926, 1950, 2175, 2190, 2497 },
    {  415,  927, 1951, 2048, 2191, 2498 },
    {  416,  928, 1952, 2049, 2192, 2499 },
    {  417,  929, 1953, 2050, 2193, 2500 },
    {  418,  930, 1954, 2051, 2194, 2501 },
    {  419,  931, 1955, 2052, 2195, 2502 },
    {  420,  932, 1956, 2053, 2196, 2503 },
    {  421,  933, 1957, 2054, 2197, 2504 },
    {  422,  934, 1958, 2055, 2198, 2505 },
    {  423,  935, 1959, 2056, 2199, 2506 },
    {  424,  936, 1960, 2057, 2200, 2507 },
    {  425,  937, 1961, 2058, 2201, 2508 },
    {  426,  938, 1962, 2059, 2202, 2509 },
    {  427,  939, 1963, 2060, 2203, 2510 },
    {  428,  940, 1964, 2061, 2204, 2511 },
    {  429,  941, 1965, 2062, 2205, 2512 },
    {  430,  942, 1966, 2063, 2206, 2513 },
    {  431,  943, 1967, 2064, 2207, 2514 },
    {  432,  944, 1968, 2065, 2208, 2515 },
    {  433,  945, 1969, 2066, 2209, 2516 },
    {  434,  946, 1970, 2067, 2210, 2517 },
    {  435,  947, 1971, 2068, 2211, 2518 },
    {  436,  948, 1972, 2069, 2212, 2519 },
    {  437,  949, 1973, 2070, 2213, 2520 },
    {  438,  950, 1974, 2071, 2214, 2521 },
    {  439,  951, 1975, 2072, 2215, 2522 },
    {  440,  952, 1976, 2073, 2216, 2523 },
    {  441,  953, 1977, 2074, 2217, 2524 },
    {  442,  954, 1978, 2075, 2218, 2525 },
    {  443,  955, 1979, 2076, 2219, 2526 },
    {  444,  956, 1980, 2077, 2220, 2527 },
    {  445,  957, 1981, 2078, 2221, 2528 },
    {  446,  958, 1982, 2079, 2222, 2529 },
    {  447,  959, 1983, 2080, 2223, 2530 },
    {  448,  960, 1984, 2081, 2224, 2531 },
    {  449,  961, 1985, 2082, 2225, 2532 },
    {  450,  962, 1986, 2083, 2226, 2533 },
    {  451,  963, 1987, 2084, 2227, 2534 },
    {  452,  964, 1988, 2085, 2228, 2535 },
    {  453,  965, 1989, 2086, 2229, 2536 },
    {  454,  966, 1990, 2087, 2230, 2537 },
    {  455,  967, 1991, 2088, 2231, 2538 },
    {  456,  968, 1992, 2089, 2232, 2539 },
    {  457,  969, 1993, 2090, 2233, 2540 },
    {  458,  970, 1994, 2091, 2234, 2541 },
    {  459,  971, 1995, 2092, 2235, 2542 },
    {  460,  972, 1996, 2093, 2236, 2543 },
    {  461,  973, 1997, 2094, 2237, 2544 },
    {  462,  974, 1998, 2095, 2238, 2545 },
    {  463,  975, 1999, 2096, 2239, 2546 },
    {  464,  976, 2000, 2097, 2240, 2547 },
    {  465,  977, 2001, 2098, 2241, 2548 },
    {  466,  978, 2002, 2099, 2242, 2549 },
    {  467,  979, 2003, 2100, 2243, 2550 },
    {  468,  980, 2004, 2101, 2244, 2551 },
    {  469,  981, 2005, 2102, 2245, 2552 },
    {  470,  982, 2006, 2103, 2246, 2553 },
    {  471,  983, 2007, 2104, 2247, 2554 },
    {  472,  984, 2008, 2105, 2248, 2555 },
    {  473,  985, 2009, 2106, 2249, 2556 },
    {  474,  986, 2010, 2107, 2250, 2557 },
    {  475,  987, 2011, 2108, 2251, 2558 },
    {  476,  988, 2012, 2109, 2252, 2559 },
    {  477,  989, 2013, 2110, 2253, 2432 },
    {  478,  990, 2014, 2111, 2254, 2433 },
    {  479,  991, 2015, 2112, 2255, 2434 },
    {  480,  992, 2016, 2113, 2256, 2435 },
    {  481,  993, 2017, 2114, 2257, 2436 },
    {  482,  994, 2018, 2115, 2258, 2437 },
    {  483,  995, 2019, 2116, 2259, 2438 },
    {  484,  996, 2020, 2117, 2260, 2439 },
    {  485,  997, 2021, 2118, 2261, 2440 },
    {  486,  998, 2022, 2119, 2262, 2441 },
    {  487,  999, 2023, 2120, 2263, 2442 },
    {  488, 1000, 2024, 2121, 2264, 2443 },
    {  489, 1001, 2025, 2122, 2265, 2444 },
    {  490, 1002, 2026, 2123, 2266, 2445 },
    {  491, 1003, 2027, 2124, 2267, 2446 },
    {  492, 1004, 2028, 2125, 2268, 2447 },
    {  493, 1005, 2029, 2126, 2269, 2448 },
    {  494, 1006, 2030, 2127, 2270, 2449 },
    {  495, 1007, 2031, 2128, 2271, 2450 },
    {  496, 1008, 2032, 2129, 2272, 2451 },
    {  497, 1009, 2033, 2130, 2273, 2452 },
    {  498, 1010, 2034, 2131, 2274, 2453 },
    {  499, 1011, 2035, 2132, 2275, 2454 },
    {  500, 1012, 2036, 2133, 2276, 2455 },
    {  501, 1013, 2037, 2134, 2277, 2456 },
    {  502, 1014, 2038, 2135, 2278, 2457 },
    {  503, 1015, 2039, 2136, 2279, 2458 },
    {  504, 1016, 2040, 2137, 2280, 2459 },
    {  505, 1017, 2041, 2138, 2281, 2460 },
    {  506, 1018, 2042, 2139, 2282, 2461 },
    {  507, 1019, 2043, 2140, 2283, 2462 },
    {  508, 1020, 2044, 2141, 2284, 2463 },
    {  509, 1021, 2045, 2142, 2285, 2464 },
    {  510, 1022, 2046, 2143, 2286, 2465 },
    {  511, 1023, 2047, 2144, 2287, 2466 },
    {    0,  818,  925, 1651, 1694, 2048 },
    {    1,  819,  926, 1652, 1695, 2049 },
    {    2,  820,  927, 1653, 1696, 2050 },
    {    3,  821,  928, 1654, 1697, 2051 },
    {    4,  822,  929, 1655, 1698, 2052 },
    {    5,  823,  930, 1656, 1699, 2053 },
    {    6,  824,  931, 1657, 1700, 2054 },
    {    7,  825,  932, 1658, 1701, 2055 },
    {    8,  826,  933, 1659, 1702, 2056 },
    {    9,  827,  934, 1660, 1703, 2057 },
    {   10,  828,  935, 1661, 1704, 2058 },
    {   11,  829,  936, 1662, 1705, 2059 },
    {   12,  830,  937, 1663, 1706, 2060 },
    {   13,  831,  938, 1536, 1707, 2061 },
    {   14,  832,  939, 1537, 1708, 2062 },
    {   15,  833,  940, 1538, 1709, 2063 },
    {   16,  834,  941, 1539, 1710, 2064 },
    {   17,  835,  942, 1540, 1711, 2065 },
    {   18,  836,  943, 1541, 1712, 2066 },
    {   19,  837,  944, 1542, 1713, 2067 },
    {   20,  838,  945, 1543, 1714, 2068 },
    {   21,  839,  946, 1544, 1715, 2069 },
    {   22,  840,  947, 1545, 1716, 2070 },
    {   23,  841,  948, 1546, 1717, 2071 },
    {   24,  842,  949, 1547, 1718, 2072 },
    {   25,  843,  950, 1548, 1719, 2073 },
    {   26,  844,  951, 1549, 1720, 2074 },
    {   27,  845,  952, 1550, 1721, 2075 },
    {   28,  846,  953, 1551, 1722, 2076 },
    {   29,  847,  954, 1552, 1723, 2077 },
    {   30,  848,  955, 1553, 1724, 2078 },
    {   31,  849,  956, 1554, 1725, 2079 },
    {   32,  850,  957, 1555, 1726, 2080 },
    {   33,  851,  958, 1556, 1727, 2081 },
    {   34,  852,  959, 1557, 1728, 2082 },
    {   35,  853,  960, 1558, 1729, 2083 },
    {   36,  854,  961, 1559, 1730, 2084 },
    {   37,  855,  962, 1560, 1731, 2085 },
    {   38,  856,  963, 1561, 1732, 2086 },
    {   39,  857,  964, 1562, 1733, 2087 },
    {   40,  858,  965, 1563, 1734, 2088 },
    {   41,  859,  966, 1564, 1735, 2089 },
    {   42,  860,  967, 1565, 1736, 2090 },
    {   43,  861,  968, 1566, 1737, 2091 },
    {   44,  862,  969, 1567, 1738, 2092 },
    {   45,  863,  970, 1568, 1739, 2093 },
    {   46,  864,  971, 1569, 1740, 2094 },
    {   47,  865,  972, 1570, 1741, 2095 },
    {   48,  866,  973, 1571, 1742, 2096 },
    {   49,  867,  974, 1572, 1743, 2097 },
    {   50,  868,  975, 1573, 1744, 2098 },
    {   51,  869,  976, 1574, 1745, 2099 },
    {   52,  870,  977, 1575, 1746, 2100 },
    {   53,  871,  978, 1576, 1747, 2101 },
    {   54,  872,  979, 1577, 1748, 2102 },
    {   55,  873,  980, 1578, 1749, 2103 },
    {   56,  874,  981, 1579, 1750, 2104 },
    {   57,  875,  982, 1580, 1751, 2105 },
    {   58,  876,  983, 1581, 1752, 2106 },
    {   59,  877,  984, 1582, 1753, 2107 },
    {   60,  878,  985, 1583, 1754, 2108 },
    {   61,  879,  986, 1584, 1755, 2109 },
    {   62,  880,  987, 1585, 1756, 2110 },
    {   63,  881,  988, 1586, 1757, 2111 },
    {   64,  882,  989, 1587, 1758, 2112 },
    {   65,  883,  990, 1588, 1759, 2113 },
    {   66,  884,  991, 1589, 1760, 2114 },
    {   67,  885,  992, 1590, 1761, 2115 },
    {   68,  886,  993, 1591, 1762, 2116 },
    {   69,  887,  994, 1592, 1763, 2117 },
    {   70,  888,  995, 1593, 1764, 2118 },
    {   71,  889,  996, 1594, 1765, 2119 },
    {   72,  890,  997, 1595, 1766, 2120 },
    {   73,  891,  998, 1596, 1767, 2121 },
    {   74,  892,  999, 1597, 1768, 2122 },
    {   75,  893, 1000, 1598, 1769, 2123 },
    {   76,  894, 1001, 1599, 1770, 2124 },
    {   77,  895, 1002, 1600, 1771, 2125 },
    {   78,  768, 1003, 1601, 1772, 2126 },
    {   79,  769, 1004, 1602, 1773, 2127 },
    {   80,  770, 1005, 1603, 1774, 2128 },
    {   81,  771, 1006, 1604, 1775, 2129 },
    {   82,  772, 1007, 1605, 1776, 2130 },
    {   83,  773, 1008, 1606, 1777, 2131 },
    {   84,  774, 1009, 1607, 1778, 2132 },
    {   85,  775, 1010, 1608, 1779, 2133 },
    {   86,  776, 1011, 1609, 1780, 2134 },
    {   87,  777, 1012, 1610, 1781, 2135 },
    {   88,  778, 1013, 1611, 1782, 2136 },
    {   89,  779, 1014, 1612, 1783, 2137 },
    {   90,  780, 1015, 1613, 1784, 2138 },
    {   91,  781, 1016, 1614, 1785, 2139 },
    {   92,  782, 1017, 1615, 1786, 2140 },
    {   93,  783, 1018, 1616, 1787, 2141 },
    {   94,  784, 1019, 1617, 1788, 2142 },
    {   95,  785, 1020, 1618, 1789, 2143 },
    {   96,  786, 1021, 1619, 1790, 2144 },
    {   97,  787, 1022, 1620, 1791, 2145 },
    {   98,  788, 1023, 1621, 1664, 2146 },
    {   99,  789,  896, 1622, 1665, 2147 },
    {  100,  790,  897, 1623, 1666, 2148 },
    {  101,  791,  898, 1624, 1667, 2149 },
    {  102,  792,  899, 1625, 1668, 2150 },
    {  103,  793,  900, 1626, 1669, 2151 },
    {  104,  794,  901, 1627, 1670, 2152 },
    {  105,  795,  902, 1628, 1671, 2153 },
    {  106,  796,  903, 1629, 1672, 2154 },
    {  107,  797,  904, 1630, 1673, 2155 },
    {  108,  798,  905, 1631, 1674, 2156 },
    {  109,  799,  906, 1632, 1675, 2157 },
    {  110,  800,  907, 1633, 1676, 2158 },
    {  111,  801,  908, 1634, 1677, 2159 },
    {  112,  802,  909, 1635, 1678, 2160 },
    {  113,  803,  910, 1636, 1679, 2161 },
    {  114,  804,  911, 1637, 1680, 2162 },
    {  115,  805,  912, 1638, 1681, 2163 },
    {  116,  806,  913, 1639, 1682, 2164 },
    {  117,  807,  914, 1640, 1683, 2165 },
    {  118,  808,  915, 1641, 1684, 2166 },
    {  119,  809,  916, 1642, 1685, 2167 },
    {  120,  810,  917, 1643, 1686, 2168 },
    {  121,  811,  918, 1644, 1687, 2169 },
    {  122,  812,  919, 1645, 1688, 2170 },
    {  123,  813,  920, 1646, 1689, 2171 },
    {  124,  814,  921, 1647, 1690, 2172 },
    {  125,  815,  922, 1648, 1691, 2173 },
    {  126,  816,  923, 1649, 1692, 2174 },
    {  127,  817,  924, 1650, 1693, 2175 },
    {  128,  512,  943, 1723, 1894, 2176 },
    {  129,  513,  944, 1724, 1895, 2177 },
    {  130,  514,  945, 1725, 1896, 2178 },
    {  131,  515,  946, 1726, 1897, 2179 },
    {  132,  516,  947, 1727, 1898, 2180 },
    {  133,  517,  948, 1728, 1899, 2181 },
    {  134,  518,  949, 1729, 1900, 2182 },
    {  135,  519,  950, 1730, 1901, 2183 },
    {  136,  520,  951, 1731, 1902, 2184 },
    {  137,  521,  952, 1732, 1903, 2185 },
    {  138,  522,  953, 1733, 1904, 2186 },
    {  139,  523,  954, 1734, 1905, 2187 },
    {  140,  524,  955, 1735, 1906, 2188 },
    {  141,  525,  956, 1736, 1907, 2189 },
    {  142,  526,  957, 1737, 1908, 2190 },
    {  143,  527,  958, 1738, 1909, 2191 },
    {  144,  528,  959, 1739, 1910, 2192 },
    {  145,  529,  960, 1740, 1911, 2193 },
    {  146,  530,  961, 1741, 1912, 2194 },
    {  147,  531,  962, 1742, 1913, 2195 },
    {  148,  532,  963, 1743, 1914, 2196 },
    {  149,  533,  964, 1744, 1915, 2197 },
    {  150,  534,  965, 1745, 1916, 2198 },
    {  151,  535,  966, 1746, 1917, 2199 },
    {  152,  536,  967, 1747, 1918, 2200 },
    {  153,  537,  968, 1748, 1919, 2201 },
    {  154,  538,  969, 1749, 1792, 2202 },
    {  155,  539,  970, 1750, 1793, 2203 },
    {  156,  540,  971, 1751, 1794, 2204 },
    {  157,  541,  972, 1752, 1795, 2205 },
    {  158,  542,  973, 1753, 1796, 2206 },
    {  159,  543,  974, 1754, 1797, 2207 },
    {  160,  544,  975, 1755, 1798, 2208 },
    {  161,  545,  976, 1756, 1799, 2209 },
    {  162,  546,  977, 1757, 1800, 2210 },
    {  163,  547,  978, 1758, 1801, 2211 },
    {  164,  548,  979, 1759, 1802, 2212 },
    {  165,  549,  980, 1760, 1803, 2213 },
    {  166,  550,  981, 1761, 1804, 2214 },
    {  167,  551,  982, 1762, 1805, 2215 },
    {  168,  552,  983, 1763, 1806, 2216 },
    {  169,  553,  984, 1764, 1807, 2217 },
    {  170,  554,  985, 1765, 1808, 2218 },
    {  171,  555,  986, 1766, 1809, 2219 },
    {  172,  556,  987, 1767, 1810, 2220 },
    {  173,  557,  988, 1768, 1811, 2221 },
    {  174,  558,  989, 1769, 1812, 2222 },
    {  175,  559,  990, 1770, 1813, 2223 },
    {  176,  560,  991, 1771, 1814, 2224 },
    {  177,  561,  992, 1772, 1815, 2225 },
    {  178,  562,  993, 1773, 1816, 2226 },
    {  179,  563,  994, 1774, 1817, 2227 },
    {  180,  564,  995, 1775, 1818, 2228 },
    {  181,  565,  996, 1776, 1819, 2229 },
    {  182,  566,  997, 1777, 1820, 2230 },
    {  183,  567,  998, 1778, 1821, 2231 },
    {  184,  568,  999, 1779, 1822, 2232 },
    {  185,  569, 1000, 1780, 1823, 2233 },
    {  186,  570, 1001, 1781, 1824, 2234 },
    {  187,  571, 1002, 1782, 1825, 2235 },
    {  188,  572, 1003, 1783, 1826, 2236 },
    {  189,  573, 1004, 1784, 1827, 2237 },
    {  190,  574, 1005, 1785, 1828, 2238 },
    {  191,  575, 1006, 1786, 1829, 2239 },
    {  192,  576, 1007, 1787, 1830, 2240 },
    {  193,  577, 1008, 1788, 1831, 2241 },
    {  194,  578, 1009, 1789, 1832, 2242 },
    {  195,  579, 1010, 1790, 1833, 2243 },
    {  196,  580, 1011, 1791, 1834, 2244 },
    {  197,  581, 1012, 1664, 1835, 2245 },
    {  198,  582, 1013, 1665, 1836, 2246 },
    {  199,  583, 1014, 1666, 1837, 2247 },
    {  200,  584, 1015, 1667, 1838, 2248 },
    {  201,  585, 1016, 1668, 1839, 2249 },
    {  202,  586, 1017, 1669, 1840, 2250 },
    {  203,  587, 1018, 1670, 1841, 2251 },
    {  204,  588, 1019, 1671, 1842, 2252 },
    {  205,  589, 1020, 1672, 1843, 2253 },
    {  206,  590, 1021, 1673, 1844, 2254 },
    {  207,  591, 1022, 1674, 1845, 2255 },
    {  208,  592, 1023, 1675, 1846, 2256 },
    {  209,  593,  896, 1676, 1847, 2257 },
    {  210,  594,  897, 1677, 1848, 2258 },
    {  211,  595,  898, 1678, 1849, 2259 },
    {  212,  596,  899, 1679, 1850, 2260 },
    {  213,  597,  900, 1680, 1851, 2261 },
    {  214,  598,  901, 1681, 1852, 2262 },
    {  215,  599,  902, 1682, 1853, 2263 },
    {  216,  600,  903, 1683, 1854, 2264 },
    {  217,  601,  904, 1684, 1855, 2265 },
    {  218,  602,  905, 1685, 1856, 2266 },
    {  219,  603,  906, 1686, 1857, 2267 },
    {  220,  604,  907, 1687, 1858, 2268 },
    {  221,  605,  908, 1688, 1859, 2269 },
    {  222,  606,  909, 1689, 1860, 2270 },
    {  223,  607,  910, 1690, 1861, 2271 },
    {  224,  608,  911, 1691, 1862, 2272 },
    {  225,  609,  912, 1692, 1863, 2273 },
    {  226,  610,  913, 1693, 1864, 2274 },
    {  227,  611,  914, 1694, 1865, 2275 },
    {  228,  612,  915, 1695, 1866, 2276 },
    {  229,  613,  916, 1696, 1867, 2277 },
    {  230,  614,  917, 1697, 1868, 2278 },
    {  231,  615,  918, 1698, 1869, 2279 },
    {  232,  616,  919, 1699, 1870, 2280 },
    {  233,  617,  920, 1700, 1871, 2281 },
    {  234,  618,  921, 1701, 1872, 2282 },
    {  235,  619,  922, 1702, 1873, 2283 },
    {  236,  620,  923, 1703, 1874, 2284 },
    {  237,  621,  924, 1704, 1875, 2285 },
    {  238,  622,  925, 1705, 1876, 2286 },
    {  239,  623,  926, 1706, 1877, 2287 },
    {  240,  624,  927, 1707, 1878, 2288 },
    {  241,  625,  928, 1708, 1879, 2289 },
    {  242,  626,  929, 1709, 1880, 2290 },
    {  243,  627,  930, 1710, 1881, 2291 },
    {  244,  628,  931, 1711, 1882, 2292 },
    {  245,  629,  932, 1712, 1883, 2293 },
    {  246,  630,  933, 1713, 1884, 2294 },
    {  247,  631,  934, 1714, 1885, 2295 },
    {  248,  632,  935, 1715, 1886, 2296 },
    {  249,  633,  936, 1716, 1887, 2297 },
    {  250,  634,  937, 1717, 1888, 2298 },
    {  251,  635,  938, 1718, 1889, 2299 },
    {  252,  636,  939, 1719, 1890, 2300 },
    {  253,  637,  940, 1720, 1891, 2301 },
    {  254,  638,  941, 1721, 1892, 2302 },
    {  255,  639,  942, 1722, 1893, 2303 },
    {  256,  543,  762, 1793, 1989, 2304 },
    {  257,  544,  763, 1794, 1990, 2305 },
    {  258,  545,  764, 1795, 1991, 2306 },
    {  259,  546,  765, 1796, 1992, 2307 },
    {  260,  547,  766, 1797, 1993, 2308 },
    {  261,  548,  767, 1798, 1994, 2309 },
    {  262,  549,  640, 1799, 1995, 2310 },
    {  263,  550,  641, 1800, 1996, 2311 },
    {  264,  551,  642, 1801, 1997, 2312 },
    {  265,  552,  643, 1802, 1998, 2313 },
    {  266,  553,  644, 1803, 1999, 2314 },
    {  267,  554,  645, 1804, 2000, 2315 },
    {  268,  555,  646, 1805, 2001, 2316 },
    {  269,  556,  647, 1806, 2002, 2317 },
    {  270,  557,  648, 1807, 2003, 2318 },
    {  271,  558,  649, 1808, 2004, 2319 },
    {  272,  559,  650, 1809, 2005, 2320 },
    {  273,  560,  651, 1810, 2006, 2321 },
    {  274,  561,  652, 1811, 2007, 2322 },
    {  275,  562,  653, 1812, 2008, 2323 },
    {  276,  563,  654, 1813, 2009, 2324 },
    {  277,  564,  655, 1814, 2010, 2325 },
    {  278,  565,  656, 1815, 2011, 2326 },
    {  279,  566,  657, 1816, 2012, 2327 },
    {  280,  567,  658, 1817, 2013, 2328 },
    {  281,  568,  659, 1818, 2014, 2329 },
    {  282,  569,  660, 1819, 2015, 2330 },
    {  283,  570,  661, 1820, 2016, 2331 },
    {  284,  571,  662, 1821, 2017, 2332 },
    {  285,  572,  663, 1822, 2018, 2333 },
    {  286,  573,  664, 1823, 2019, 2334 },
    {  287,  574,  665, 1824, 2020, 2335 },
    {  288,  575,  666, 1825, 2021, 2336 },
    {  289,  576,  667, 1826, 2022, 2337 },
    {  290,  577,  668, 1827, 2023, 2338 },
    {  291,  578,  669, 1828, 2024, 2339 },
    {  292,  579,  670, 1829, 2025, 2340 },
    {  293,  580,  671, 1830, 2026, 2341 },
    {  294,  581,  672, 1831, 2027, 2342 },
    {  295,  582,  673, 1832, 2028, 2343 },
    {  296,  583,  674, 1833, 2029, 2344 },
    {  297,  584,  675, 1834, 2030, 2345 },
    {  298,  585,  676, 1835, 2031, 2346 },
    {  299,  586,  677, 1836, 2032, 2347 },
    {  300,  587,  678, 1837, 2033, 2348 },
    {  301,  588,  679, 1838, 2034, 2349 },
    {  302,  589,  680, 1839, 2035, 2350 },
    {  303,  590,  681, 1840, 2036, 2351 },
    {  304,  591,  682, 1841, 2037, 2352 },
    {  305,  592,  683, 1842, 2038, 2353 },
    {  306,  593,  684, 1843, 2039, 2354 },
    {  307,  594,  685, 1844, 2040, 2355 },
    {  308,  595,  686, 1845, 2041, 2356 },
    {  309,  596,  687, 1846, 2042, 2357 },
    {  310,  597,  688, 1847, 2043, 2358 },
    {  311,  598,  689, 1848, 2044, 2359 },
    {  312,  599,  690, 1849, 2045, 2360 },
    {  313,  600,  691, 1850, 2046, 2361 },
    {  314,  601,  692, 1851, 2047, 2362 },
    {  315,  602,  693, 1852, 1920, 2363 },
    {  316,  603,  694, 1853, 1921, 2364 },
    {  317,  604,  695, 1854, 1922, 2365 },
    {  318,  605,  696, 1855, 1923, 2366 },
    {  319,  606,  697, 1856, 1924, 2367 },
    {  320,  607,  698, 1857, 1925, 2368 },
    {  321,  608,  699, 1858, 1926, 2369 },
    {  322,  609,  700, 1859, 1927, 2370 },
    {  323,  610,  701, 1860, 1928, 2371 },
    {  324,  611,  702, 1861, 1929, 2372 },
    {  325,  612,  703, 1862, 1930, 2373 },
    {  326,  613,  704, 1863, 1931, 2374 },
    {  327,  614,  705, 1864, 1932, 2375 },
    {  328,  615,  706, 1865, 1933, 2376 },
    {  329,  616,  707, 1866, 1934, 2377 },
    {  330,  617,  708, 1867, 1935, 2378 },
    {  331,  618,  709, 1868, 1936, 2379 },
    {  332,  619,  710, 1869, 1937, 2380 },
    {  333,  620,  711, 1870, 1938, 2381 },
    {  334,  621,  712, 1871, 1939, 2382 },
    {  335,  622,  713, 1872, 1940, 2383 },
    {  336,  623,  714, 1873, 1941, 2384 },
    {  337,  624,  715, 1874, 1942, 2385 },
    {  338,  625,  716, 1875, 1943, 2386 },
    {  339,  626,  717, 1876, 1944, 2387 },
    {  340,  627,  718, 1877, 1945, 2388 },
    {  341,  628,  719, 1878, 1946, 2389 },
    {  342,  629,  720, 1879, 1947, 2390 },
    {  343,  630,  721, 1880, 1948, 2391 },
    {  344,  631,  722, 1881, 1949, 2392 },
    {  345,  632,  723, 1882, 1950, 2393 },
    {  346,  633,  724, 1883, 1951, 2394 },
    {  347,  634,  725, 1884, 1952, 2395 },
    {  348,  635,  726, 1885, 1953, 2396 },
    {  349,  636,  727, 1886, 1954, 2397 },
    {  350,  637,  728, 1887, 1955, 2398 },
    {  351,  638,  729, 1888, 1956, 2399 },
    {  352,  639,  730, 1889, 1957, 2400 },
    {  353,  512,  731, 1890, 1958, 2401 },
    {  354,  513,  732, 1891, 1959, 2402 },
    {  355,  514,  733, 1892, 1960, 2403 },
    {  356,  515,  734, 1893, 1961, 2404 },
    {  357,  516,  735, 1894, 1962, 2405 },
    {  358,  517,  736, 1895, 1963, 2406 },
    {  359,  518,  737, 1896, 1964, 2407 },
    {  360,  519,  738, 1897, 1965, 2408 },
    {  361,  520,  739, 1898, 1966, 2409 },
    {  362,  521,  740, 1899, 1967, 2410 },
    {  363,  522,  741, 1900, 1968, 2411 },
    {  364,  523,  742, 1901, 1969, 2412 },
    {  365,  524,  743, 1902, 1970, 2413 },
    {  366,  525,  744, 1903, 1971, 2414 },
    {  367,  526,  745, 1904, 1972, 2415 },
    {  368,  527,  746, 1905, 1973, 2416 },
    {  369,  528,  747, 1906, 1974, 2417 },
    {  370,  529,  748, 1907, 1975, 2418 },
    {  371,  530,  749, 1908, 1976, 2419 },
    {  372,  531,  750, 1909, 1977, 2420 },
    {  373,  532,  751, 1910, 1978, 2421 },
    {  374,  533,  752, 1911, 1979, 2422 },
    {  375,  534,  753, 1912, 1980, 2423 },
    {  376,  535,  754, 1913, 1981, 2424 },
    {  377,  536,  755, 1914, 1982, 2425 },
    {  378,  537,  756, 1915, 1983, 2426 },
    {  379,  538,  757, 1916, 1984, 2427 },
    {  380,  539,  758, 1917, 1985, 2428 },
    {  381,  540,  759, 1918, 1986, 2429 },
    {  382,  541,  760, 1919, 1987, 2430 },
    {  383,  542,  761, 1792, 1988, 2431 },
    {  384,  704,  861, 1630, 2019, 2432 },
    {  385,  705,  862, 1631, 2020, 2433 },
    {  386,  706,  863, 1632, 2021, 2434 },
    {  387,  707,  864, 1633, 2022, 2435 },
    {  388,  708,  865, 1634, 2023, 2436 },
    {  389,  709,  866, 1635, 2024, 2437 },
    {  390,  710,  867, 1636, 2025, 2438 },
    {  391,  711,  868, 1637, 2026, 2439 },
    {  392,  712,  869, 1638, 2027, 2440 },
    {  393,  713,  870, 1639, 2028, 2441 },
    {  394,  714,  871, 1640, 2029, 2442 },
    {  395,  715,  872, 1641, 2030, 2443 },
    {  396,  716,  873, 1642, 2031, 2444 },
    {  397,  717,  874, 1643, 2032, 2445 },
    {  398,  718,  875, 1644, 2033, 2446 },
    {  399,  719,  876, 1645, 2034, 2447 },
    {  400,  720,  877, 1646, 2035, 2448 },
    {  401,  721,  878, 1647, 2036, 2449 },
    {  402,  722,  879, 1648, 2037, 2450 },
    {  403,  723,  880, 1649, 2038, 2451 },
    {  404,  724,  881, 1650, 2039, 2452 },
    {  405,  725,  882, 1651, 2040, 2453 },
    {  406,  726,  883, 1652, 2041, 2454 },
    {  407,  727,  884, 1653, 2042, 2455 },
    {  408,  728,  885, 1654, 2043, 2456 },
    {  409,  729,  886, 1655, 2044, 2457 },
    {  410,  730,  887, 1656, 2045, 2458 },
    {  411,  731,  888, 1657, 2046, 2459 },
    {  412,  732,  889, 1658, 2047, 2460 },
    {  413,  733,  890, 1659, 1920, 2461 },
    {  414,  734,  891, 1660, 1921, 2462 },
    {  415,  735,  892, 1661, 1922, 2463 },
    {  416,  736,  893, 1662, 1923, 2464 },
    {  417,  737,  894, 1663, 1924, 2465 },
    {  418,  738,  895, 1536, 1925, 2466 },
    {  419,  739,  768, 1537, 1926, 2467 },
    {  420,  740,  769, 1538, 1927, 2468 },
    {  421,  741,  770, 1539, 1928, 2469 },
    {  422,  742,  771, 1540, 1929, 2470 },
    {  423,  743,  772, 1541, 1930, 2471 },
    {  424,  744,  773, 1542, 1931, 2472 },
    {  425,  745,  774, 1543, 1932, 2473 },
    {  426,  746,  775, 1544, 1933, 2474 },
    {  427,  747,  776, 1545, 1934, 2475 },
    {  428,  748,  777, 1546, 1935, 2476 },
    {  429,  749,  778, 1547, 1936, 2477 },
    {  430,  750,  779, 1548, 1937, 2478 },
    {  431,  751,  780, 1549, 1938, 2479 },
    {  432,  752,  781, 1550, 1939, 2480 },
    {  433,  753,  782, 1551, 1940, 2481 },
    {  434,  754,  783, 1552, 1941, 2482 },
    {  435,  755,  784, 1553, 1942, 2483 },
    {  436,  756,  785, 1554, 1943, 2484 },
    {  437,  757,  786, 1555, 1944, 2485 },
    {  438,  758,  787, 1556, 1945, 2486 },
    {  439,  759,  788, 1557, 1946, 2487 },
    {  440,  760,  789, 1558, 1947, 2488 },
    {  441,  761,  790, 1559, 1948, 2489 },
    {  442,  762,  791, 1560, 1949, 2490 },
    {  443,  763,  792, 1561, 1950, 2491 },
    {  444,  764,  793, 1562, 1951, 2492 },
    {  445,  765,  794, 1563, 1952, 2493 },
    {  446,  766,  795, 1564, 1953, 2494 },
    {  447,  767,  796, 1565, 1954, 2495 },
    {  448,  640,  797, 1566, 1955, 2496 },
    {  449,  641,  798, 1567, 1956, 2497 },
    {  450,  642,  799, 1568, 1957, 2498 },
    {  451,  643,  800, 1569, 1958, 2499 },
    {  452,  644,  801, 1570, 1959, 2500 },
    {  453,  645,  802, 1571, 1960, 2501 },
    {  454,  646,  803, 1572, 1961, 2502 },
    {  455,  647,  804, 1573, 1962, 2503 },
    {  456,  648,  805, 1574, 1963, 2504 },
    {  457,  649,  806, 1575, 1964, 2505 },
    {  458,  650,  807, 1576, 1965, 2506 },
    {  459,  651,  808, 1577, 1966, 2507 },
    {  460,  652,  809, 1578, 1967, 2508 },
    {  461,  653,  810, 1579, 1968, 2509 },
    {  462,  654,  811, 1580, 1969, 2510 },
    {  463,  655,  812, 1581, 1970, 2511 },
    {  464,  656,  813, 1582, 1971, 2512 },
    {  465,  657,  814, 1583, 1972, 2513 },
    {  466,  658,  815, 1584, 1973, 2514 },
    {  467,  659,  816, 1585, 1974, 2515 },
    {  468,  660,  817, 1586, 1975, 2516 },
    {  469,  661,  818, 1587, 1976, 2517 },
    {  470,  662,  819, 1588, 1977, 2518 },
    {  471,  663,  820, 1589, 1978, 2519 },
    {  472,  664,  821, 1590, 1979, 2520 },
    {  473,  665,  822, 1591, 1980, 2521 },
    {  474,  666,  823, 1592, 1981, 2522 },
    {  475,  667,  824, 1593, 1982, 2523 },
    {  476,  668,  825, 1594, 1983, 2524 },
    {  477,  669,  826, 1595, 1984, 2525 },
    {  478,  670,  827, 1596, 1985, 2526 },
    {  479,  671,  828, 1597, 1986, 2527 },
    {  480,  672,  829, 1598, 1987, 2528 },
    {  481,  673,  830, 1599, 1988, 2529 },
    {  482,  674,  831, 1600, 1989, 2530 },
    {  483,  675,  832, 1601, 1990, 2531 },
    {  484,  676,  833, 1602, 1991, 2532 },
    {  485,  677,  834, 1603, 1992, 2533 },
    {  486,  678,  835, 1604, 1993, 2534 },
    {  487,  679,  836, 1605, 1994, 2535 },
    {  488,  680,  837, 1606, 1995, 2536 },
    {  489,  681,  838, 1607, 1996, 2537 },
    {  490,  682,  839, 1608, 1997, 2538 },
    {  491,  683,  840, 1609, 1998, 2539 },
    {  492,  684,  841, 1610, 1999, 2540 },
    {  493,  685,  842, 1611, 2000, 2541 },
    {  494,  686,  843, 1612, 2001, 2542 },
    {  495,  687,  844, 1613, 2002, 2543 },
    {  496,  688,  845, 1614, 2003, 2544 },
    {  497,  689,  846, 1615, 2004, 2545 },
    {  498,  690,  847, 1616, 2005, 2546 },
    {  499,  691,  848, 1617, 2006, 2547 },
    {  500,  692,  849, 1618, 2007, 2548 },
    {  501,  693,  850, 1619, 2008, 2549 },
    {  502,  694,  851, 1620, 2009, 2550 },
    {  503,  695,  852, 1621, 2010, 2551 },
    {  504,  696,  853, 1622, 2011, 2552 },
    {  505,  697,  854, 1623, 2012, 2553 },
    {  506,  698,  855, 1624, 2013, 2554 },
    {  507,  699,  856, 1625, 2014, 2555 },
    {  508,  700,  857, 1626, 2015, 2556 },
    {  509,  701,  858, 1627, 2016, 2557 },
    {  510,  702,  859, 1628, 2017, 2558 },
    {  511,  703,  860, 1629, 2018, 2559 },
};

} // namespace QC_LDPC_4CW
