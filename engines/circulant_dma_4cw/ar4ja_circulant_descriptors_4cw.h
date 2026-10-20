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

// Circulant Slice Descriptor (8 Bytes per descriptor)
struct alignas(8) CirculantSliceDesc {
    uint32_t vnode_byte_offset; // Base byte offset in CB_VNODE (v_base * 8)
    uint16_t burst_bytes;       // Total contiguous burst bytes (typically 128)
    uint16_t wrap_offset;       // Byte offset if crossing circulant boundary (0 = no wrap)
};

// ============================================================================
// TILE SLICE DESCRIPTOR TABLE (40 Tiles x 6 Slices)
// Total Memory: 40 * 6 * 8 = 1,920 Bytes (fits entirely in firmware SRAM)
// ============================================================================
constexpr CirculantSliceDesc TILE_SLICES_W0[40][3] = {
    // Tile  0: Checks  512.. 543 (Deg 6)
    {
        { 0x00000u, 128u, 0u }, // Edge 0: v_start =    0
        { 0x01000u, 128u, 0u }, // Edge 1: v_start =  512
        { 0x03000u, 128u, 0u }, // Edge 2: v_start = 1536
    },
    // Tile  1: Checks  544.. 575 (Deg 6)
    {
        { 0x00100u, 128u, 0u }, // Edge 0: v_start =   32
        { 0x01100u, 128u, 0u }, // Edge 1: v_start =  544
        { 0x03100u, 128u, 0u }, // Edge 2: v_start = 1568
    },
    // Tile  2: Checks  576.. 607 (Deg 6)
    {
        { 0x00200u, 128u, 0u }, // Edge 0: v_start =   64
        { 0x01200u, 128u, 0u }, // Edge 1: v_start =  576
        { 0x03200u, 128u, 0u }, // Edge 2: v_start = 1600
    },
    // Tile  3: Checks  608.. 639 (Deg 6)
    {
        { 0x00300u, 128u, 0u }, // Edge 0: v_start =   96
        { 0x01300u, 128u, 0u }, // Edge 1: v_start =  608
        { 0x03300u, 128u, 0u }, // Edge 2: v_start = 1632
    },
    // Tile  4: Checks  640.. 671 (Deg 6)
    {
        { 0x00400u, 128u, 0u }, // Edge 0: v_start =  128
        { 0x01400u, 128u, 0u }, // Edge 1: v_start =  640
        { 0x03400u, 128u, 0u }, // Edge 2: v_start = 1664
    },
    // Tile  5: Checks  672.. 703 (Deg 6)
    {
        { 0x00500u, 128u, 0u }, // Edge 0: v_start =  160
        { 0x01500u, 128u, 0u }, // Edge 1: v_start =  672
        { 0x03500u, 128u, 0u }, // Edge 2: v_start = 1696
    },
    // Tile  6: Checks  704.. 735 (Deg 6)
    {
        { 0x00600u, 128u, 0u }, // Edge 0: v_start =  192
        { 0x01600u, 128u, 0u }, // Edge 1: v_start =  704
        { 0x03600u, 128u, 0u }, // Edge 2: v_start = 1728
    },
    // Tile  7: Checks  736.. 767 (Deg 6)
    {
        { 0x00700u, 128u, 0u }, // Edge 0: v_start =  224
        { 0x01700u, 128u, 0u }, // Edge 1: v_start =  736
        { 0x03700u, 128u, 0u }, // Edge 2: v_start = 1760
    },
    // Tile  8: Checks  768.. 799 (Deg 6)
    {
        { 0x00800u, 128u, 0u }, // Edge 0: v_start =  256
        { 0x01800u, 128u, 0u }, // Edge 1: v_start =  768
        { 0x03800u, 128u, 0u }, // Edge 2: v_start = 1792
    },
    // Tile  9: Checks  800.. 831 (Deg 6)
    {
        { 0x00900u, 128u, 0u }, // Edge 0: v_start =  288
        { 0x01900u, 128u, 0u }, // Edge 1: v_start =  800
        { 0x03900u, 128u, 0u }, // Edge 2: v_start = 1824
    },
    // Tile 10: Checks  832.. 863 (Deg 6)
    {
        { 0x00A00u, 128u, 0u }, // Edge 0: v_start =  320
        { 0x01A00u, 128u, 0u }, // Edge 1: v_start =  832
        { 0x03A00u, 128u, 0u }, // Edge 2: v_start = 1856
    },
    // Tile 11: Checks  864.. 895 (Deg 6)
    {
        { 0x00B00u, 128u, 0u }, // Edge 0: v_start =  352
        { 0x01B00u, 128u, 0u }, // Edge 1: v_start =  864
        { 0x03B00u, 128u, 0u }, // Edge 2: v_start = 1888
    },
    // Tile 12: Checks  896.. 927 (Deg 6)
    {
        { 0x00C00u, 128u, 0u }, // Edge 0: v_start =  384
        { 0x01C00u, 128u, 0u }, // Edge 1: v_start =  896
        { 0x03C00u, 128u, 0u }, // Edge 2: v_start = 1920
    },
    // Tile 13: Checks  928.. 959 (Deg 6)
    {
        { 0x00D00u, 128u, 0u }, // Edge 0: v_start =  416
        { 0x01D00u, 128u, 0u }, // Edge 1: v_start =  928
        { 0x03D00u, 128u, 0u }, // Edge 2: v_start = 1952
    },
    // Tile 14: Checks  960.. 991 (Deg 6)
    {
        { 0x00E00u, 128u, 0u }, // Edge 0: v_start =  448
        { 0x01E00u, 128u, 0u }, // Edge 1: v_start =  960
        { 0x03E00u, 128u, 0u }, // Edge 2: v_start = 1984
    },
    // Tile 15: Checks  992..1023 (Deg 6)
    {
        { 0x00F00u, 128u, 0u }, // Edge 0: v_start =  480
        { 0x01F00u, 128u, 0u }, // Edge 1: v_start =  992
        { 0x03F00u, 128u, 0u }, // Edge 2: v_start = 2016
    },
    // Tile 16: Checks 1024..1055 (Deg 6)
    {
        { 0x00000u, 128u, 0u }, // Edge 0: v_start =    0
        { 0x01990u, 128u, 0u }, // Edge 1: v_start =  818
        { 0x01CE8u, 128u, 0u }, // Edge 2: v_start =  925
    },
    // Tile 17: Checks 1056..1087 (Deg 6)
    {
        { 0x00100u, 128u, 0u }, // Edge 0: v_start =   32
        { 0x01A90u, 128u, 0u }, // Edge 1: v_start =  850
        { 0x01DE8u, 128u, 0u }, // Edge 2: v_start =  957
    },
    // Tile 18: Checks 1088..1119 (Deg 6)
    {
        { 0x00200u, 128u, 0u }, // Edge 0: v_start =   64
        { 0x01B90u, 128u, 0u }, // Edge 1: v_start =  882
        { 0x01EE8u, 128u, 0u }, // Edge 2: v_start =  989
    },
    // Tile 19: Checks 1120..1151 (Deg 6)
    {
        { 0x00300u, 128u, 0u }, // Edge 0: v_start =   96
        { 0x01890u, 128u, 0u }, // Edge 1: v_start =  786
        { 0x01FE8u, 128u, 0u }, // Edge 2: v_start = 1021
    },
    // Tile 20: Checks 1152..1183 (Deg 6)
    {
        { 0x00400u, 128u, 0u }, // Edge 0: v_start =  128
        { 0x01000u, 128u, 0u }, // Edge 1: v_start =  512
        { 0x01D78u, 128u, 0u }, // Edge 2: v_start =  943
    },
    // Tile 21: Checks 1184..1215 (Deg 6)
    {
        { 0x00500u, 128u, 0u }, // Edge 0: v_start =  160
        { 0x01100u, 128u, 0u }, // Edge 1: v_start =  544
        { 0x01E78u, 128u, 0u }, // Edge 2: v_start =  975
    },
    // Tile 22: Checks 1216..1247 (Deg 6)
    {
        { 0x00600u, 128u, 0u }, // Edge 0: v_start =  192
        { 0x01200u, 128u, 0u }, // Edge 1: v_start =  576
        { 0x01F78u, 128u, 0u }, // Edge 2: v_start = 1007
    },
    // Tile 23: Checks 1248..1279 (Deg 6)
    {
        { 0x00700u, 128u, 0u }, // Edge 0: v_start =  224
        { 0x01300u, 128u, 0u }, // Edge 1: v_start =  608
        { 0x01C78u, 128u, 0u }, // Edge 2: v_start =  911
    },
    // Tile 24: Checks 1280..1311 (Deg 6)
    {
        { 0x00800u, 128u, 0u }, // Edge 0: v_start =  256
        { 0x010F8u, 128u, 0u }, // Edge 1: v_start =  543
        { 0x017D0u, 128u, 0u }, // Edge 2: v_start =  762
    },
    // Tile 25: Checks 1312..1343 (Deg 6)
    {
        { 0x00900u, 128u, 0u }, // Edge 0: v_start =  288
        { 0x011F8u, 128u, 0u }, // Edge 1: v_start =  575
        { 0x014D0u, 128u, 0u }, // Edge 2: v_start =  666
    },
    // Tile 26: Checks 1344..1375 (Deg 6)
    {
        { 0x00A00u, 128u, 0u }, // Edge 0: v_start =  320
        { 0x012F8u, 128u, 0u }, // Edge 1: v_start =  607
        { 0x015D0u, 128u, 0u }, // Edge 2: v_start =  698
    },
    // Tile 27: Checks 1376..1407 (Deg 6)
    {
        { 0x00B00u, 128u, 0u }, // Edge 0: v_start =  352
        { 0x013F8u, 128u, 0u }, // Edge 1: v_start =  639
        { 0x016D0u, 128u, 0u }, // Edge 2: v_start =  730
    },
    // Tile 28: Checks 1408..1439 (Deg 6)
    {
        { 0x00C00u, 128u, 0u }, // Edge 0: v_start =  384
        { 0x01600u, 128u, 0u }, // Edge 1: v_start =  704
        { 0x01AE8u, 128u, 0u }, // Edge 2: v_start =  861
    },
    // Tile 29: Checks 1440..1471 (Deg 6)
    {
        { 0x00D00u, 128u, 0u }, // Edge 0: v_start =  416
        { 0x01700u, 128u, 0u }, // Edge 1: v_start =  736
        { 0x01BE8u, 128u, 0u }, // Edge 2: v_start =  893
    },
    // Tile 30: Checks 1472..1503 (Deg 6)
    {
        { 0x00E00u, 128u, 0u }, // Edge 0: v_start =  448
        { 0x01400u, 128u, 0u }, // Edge 1: v_start =  640
        { 0x018E8u, 128u, 0u }, // Edge 2: v_start =  797
    },
    // Tile 31: Checks 1504..1535 (Deg 6)
    {
        { 0x00F00u, 128u, 0u }, // Edge 0: v_start =  480
        { 0x01500u, 128u, 0u }, // Edge 1: v_start =  672
        { 0x019E8u, 128u, 0u }, // Edge 2: v_start =  829
    },
    // Tile 32: Checks    0..  63 (Deg 3)
    {
        { 0x02000u, 128u, 0u }, // Edge 0 (Face 0): v_start = 1024
        { 0x04000u, 128u, 0u }, // Edge 1 (Face 0): v_start = 2048
        { 0x04C80u, 128u, 0u }, // Edge 2 (Face 0): v_start = 2448
    },
    // Tile 33: Checks   64.. 127 (Deg 3)
    {
        { 0x02200u, 128u, 0u }, // Edge 0 (Face 0): v_start = 1088
        { 0x04200u, 128u, 0u }, // Edge 1 (Face 0): v_start = 2112
        { 0x04E80u, 128u, 0u }, // Edge 2 (Face 0): v_start = 2512
    },
    // Tile 34: Checks  128.. 191 (Deg 3)
    {
        { 0x02400u, 128u, 0u }, // Edge 0 (Face 0): v_start = 1152
        { 0x04000u, 128u, 0u }, // Edge 1 (Face 0): v_start = 2048
        { 0x04400u, 128u, 0u }, // Edge 2 (Face 0): v_start = 2176
    },
    // Tile 35: Checks  192.. 255 (Deg 3)
    {
        { 0x02600u, 128u, 0u }, // Edge 0 (Face 0): v_start = 1216
        { 0x04200u, 128u, 0u }, // Edge 1 (Face 0): v_start = 2112
        { 0x04600u, 128u, 0u }, // Edge 2 (Face 0): v_start = 2240
    },
    // Tile 36: Checks  256.. 319 (Deg 3)
    {
        { 0x02800u, 128u, 0u }, // Edge 0 (Face 0): v_start = 1280
        { 0x04400u, 128u, 0u }, // Edge 1 (Face 0): v_start = 2176
        { 0x04800u, 128u, 0u }, // Edge 2 (Face 0): v_start = 2304
    },
    // Tile 37: Checks  320.. 383 (Deg 3)
    {
        { 0x02A00u, 128u, 0u }, // Edge 0 (Face 0): v_start = 1344
        { 0x04600u, 128u, 0u }, // Edge 1 (Face 0): v_start = 2240
        { 0x04A00u, 128u, 0u }, // Edge 2 (Face 0): v_start = 2368
    },
    // Tile 38: Checks  384.. 447 (Deg 3)
    {
        { 0x02C00u, 128u, 0u }, // Edge 0 (Face 0): v_start = 1408
        { 0x04800u, 128u, 0u }, // Edge 1 (Face 0): v_start = 2304
        { 0x04C00u, 128u, 0u }, // Edge 2 (Face 0): v_start = 2432
    },
    // Tile 39: Checks  448.. 511 (Deg 3)
    {
        { 0x02E00u, 128u, 0u }, // Edge 0 (Face 0): v_start = 1472
        { 0x04A00u, 128u, 0u }, // Edge 1 (Face 0): v_start = 2368
        { 0x04E00u, 128u, 0u }, // Edge 2 (Face 0): v_start = 2496
    },
};

constexpr CirculantSliceDesc TILE_SLICES_W1[40][3] = {
    // Tile  0: Checks  512.. 543 (Deg 6)
    {
        { 0x00080u, 128u, 0u }, // Edge 0: v_start =   16
        { 0x01080u, 128u, 0u }, // Edge 1: v_start =  528
        { 0x03080u, 128u, 0u }, // Edge 2: v_start = 1552
    },
    // Tile  1: Checks  544.. 575 (Deg 6)
    {
        { 0x00180u, 128u, 0u }, // Edge 0: v_start =   48
        { 0x01180u, 128u, 0u }, // Edge 1: v_start =  560
        { 0x03180u, 128u, 0u }, // Edge 2: v_start = 1584
    },
    // Tile  2: Checks  576.. 607 (Deg 6)
    {
        { 0x00280u, 128u, 0u }, // Edge 0: v_start =   80
        { 0x01280u, 128u, 0u }, // Edge 1: v_start =  592
        { 0x03280u, 128u, 0u }, // Edge 2: v_start = 1616
    },
    // Tile  3: Checks  608.. 639 (Deg 6)
    {
        { 0x00380u, 128u, 0u }, // Edge 0: v_start =  112
        { 0x01380u, 128u, 0u }, // Edge 1: v_start =  624
        { 0x03380u, 128u, 0u }, // Edge 2: v_start = 1648
    },
    // Tile  4: Checks  640.. 671 (Deg 6)
    {
        { 0x00480u, 128u, 0u }, // Edge 0: v_start =  144
        { 0x01480u, 128u, 0u }, // Edge 1: v_start =  656
        { 0x03480u, 128u, 0u }, // Edge 2: v_start = 1680
    },
    // Tile  5: Checks  672.. 703 (Deg 6)
    {
        { 0x00580u, 128u, 0u }, // Edge 0: v_start =  176
        { 0x01580u, 128u, 0u }, // Edge 1: v_start =  688
        { 0x03580u, 128u, 0u }, // Edge 2: v_start = 1712
    },
    // Tile  6: Checks  704.. 735 (Deg 6)
    {
        { 0x00680u, 128u, 0u }, // Edge 0: v_start =  208
        { 0x01680u, 128u, 0u }, // Edge 1: v_start =  720
        { 0x03680u, 128u, 0u }, // Edge 2: v_start = 1744
    },
    // Tile  7: Checks  736.. 767 (Deg 6)
    {
        { 0x00780u, 128u, 0u }, // Edge 0: v_start =  240
        { 0x01780u, 128u, 0u }, // Edge 1: v_start =  752
        { 0x03780u, 128u, 0u }, // Edge 2: v_start = 1776
    },
    // Tile  8: Checks  768.. 799 (Deg 6)
    {
        { 0x00880u, 128u, 0u }, // Edge 0: v_start =  272
        { 0x01880u, 128u, 0u }, // Edge 1: v_start =  784
        { 0x03880u, 128u, 0u }, // Edge 2: v_start = 1808
    },
    // Tile  9: Checks  800.. 831 (Deg 6)
    {
        { 0x00980u, 128u, 0u }, // Edge 0: v_start =  304
        { 0x01980u, 128u, 0u }, // Edge 1: v_start =  816
        { 0x03980u, 128u, 0u }, // Edge 2: v_start = 1840
    },
    // Tile 10: Checks  832.. 863 (Deg 6)
    {
        { 0x00A80u, 128u, 0u }, // Edge 0: v_start =  336
        { 0x01A80u, 128u, 0u }, // Edge 1: v_start =  848
        { 0x03A80u, 128u, 0u }, // Edge 2: v_start = 1872
    },
    // Tile 11: Checks  864.. 895 (Deg 6)
    {
        { 0x00B80u, 128u, 0u }, // Edge 0: v_start =  368
        { 0x01B80u, 128u, 0u }, // Edge 1: v_start =  880
        { 0x03B80u, 128u, 0u }, // Edge 2: v_start = 1904
    },
    // Tile 12: Checks  896.. 927 (Deg 6)
    {
        { 0x00C80u, 128u, 0u }, // Edge 0: v_start =  400
        { 0x01C80u, 128u, 0u }, // Edge 1: v_start =  912
        { 0x03C80u, 128u, 0u }, // Edge 2: v_start = 1936
    },
    // Tile 13: Checks  928.. 959 (Deg 6)
    {
        { 0x00D80u, 128u, 0u }, // Edge 0: v_start =  432
        { 0x01D80u, 128u, 0u }, // Edge 1: v_start =  944
        { 0x03D80u, 128u, 0u }, // Edge 2: v_start = 1968
    },
    // Tile 14: Checks  960.. 991 (Deg 6)
    {
        { 0x00E80u, 128u, 0u }, // Edge 0: v_start =  464
        { 0x01E80u, 128u, 0u }, // Edge 1: v_start =  976
        { 0x03E80u, 128u, 0u }, // Edge 2: v_start = 2000
    },
    // Tile 15: Checks  992..1023 (Deg 6)
    {
        { 0x00F80u, 128u, 0u }, // Edge 0: v_start =  496
        { 0x01F80u, 128u, 0u }, // Edge 1: v_start = 1008
        { 0x03F80u, 128u, 0u }, // Edge 2: v_start = 2032
    },
    // Tile 16: Checks 1024..1055 (Deg 6)
    {
        { 0x00080u, 128u, 0u }, // Edge 0: v_start =   16
        { 0x01A10u, 128u, 0u }, // Edge 1: v_start =  834
        { 0x01D68u, 128u, 0u }, // Edge 2: v_start =  941
    },
    // Tile 17: Checks 1056..1087 (Deg 6)
    {
        { 0x00180u, 128u, 0u }, // Edge 0: v_start =   48
        { 0x01B10u, 128u, 0u }, // Edge 1: v_start =  866
        { 0x01E68u, 128u, 0u }, // Edge 2: v_start =  973
    },
    // Tile 18: Checks 1088..1119 (Deg 6)
    {
        { 0x00280u, 128u, 0u }, // Edge 0: v_start =   80
        { 0x01810u, 128u, 0u }, // Edge 1: v_start =  770
        { 0x01F68u, 128u, 0u }, // Edge 2: v_start = 1005
    },
    // Tile 19: Checks 1120..1151 (Deg 6)
    {
        { 0x00380u, 128u, 0u }, // Edge 0: v_start =  112
        { 0x01910u, 128u, 0u }, // Edge 1: v_start =  802
        { 0x01C68u, 128u, 0u }, // Edge 2: v_start =  909
    },
    // Tile 20: Checks 1152..1183 (Deg 6)
    {
        { 0x00480u, 128u, 0u }, // Edge 0: v_start =  144
        { 0x01080u, 128u, 0u }, // Edge 1: v_start =  528
        { 0x01DF8u, 128u, 0u }, // Edge 2: v_start =  959
    },
    // Tile 21: Checks 1184..1215 (Deg 6)
    {
        { 0x00580u, 128u, 0u }, // Edge 0: v_start =  176
        { 0x01180u, 128u, 0u }, // Edge 1: v_start =  560
        { 0x01EF8u, 128u, 0u }, // Edge 2: v_start =  991
    },
    // Tile 22: Checks 1216..1247 (Deg 6)
    {
        { 0x00680u, 128u, 0u }, // Edge 0: v_start =  208
        { 0x01280u, 128u, 0u }, // Edge 1: v_start =  592
        { 0x01FF8u, 128u, 0u }, // Edge 2: v_start = 1023
    },
    // Tile 23: Checks 1248..1279 (Deg 6)
    {
        { 0x00780u, 128u, 0u }, // Edge 0: v_start =  240
        { 0x01380u, 128u, 0u }, // Edge 1: v_start =  624
        { 0x01CF8u, 128u, 0u }, // Edge 2: v_start =  927
    },
    // Tile 24: Checks 1280..1311 (Deg 6)
    {
        { 0x00880u, 128u, 0u }, // Edge 0: v_start =  272
        { 0x01178u, 128u, 0u }, // Edge 1: v_start =  559
        { 0x01450u, 128u, 0u }, // Edge 2: v_start =  650
    },
    // Tile 25: Checks 1312..1343 (Deg 6)
    {
        { 0x00980u, 128u, 0u }, // Edge 0: v_start =  304
        { 0x01278u, 128u, 0u }, // Edge 1: v_start =  591
        { 0x01550u, 128u, 0u }, // Edge 2: v_start =  682
    },
    // Tile 26: Checks 1344..1375 (Deg 6)
    {
        { 0x00A80u, 128u, 0u }, // Edge 0: v_start =  336
        { 0x01378u, 128u, 0u }, // Edge 1: v_start =  623
        { 0x01650u, 128u, 0u }, // Edge 2: v_start =  714
    },
    // Tile 27: Checks 1376..1407 (Deg 6)
    {
        { 0x00B80u, 128u, 0u }, // Edge 0: v_start =  368
        { 0x01078u, 128u, 0u }, // Edge 1: v_start =  527
        { 0x01750u, 128u, 0u }, // Edge 2: v_start =  746
    },
    // Tile 28: Checks 1408..1439 (Deg 6)
    {
        { 0x00C80u, 128u, 0u }, // Edge 0: v_start =  400
        { 0x01680u, 128u, 0u }, // Edge 1: v_start =  720
        { 0x01B68u, 128u, 0u }, // Edge 2: v_start =  877
    },
    // Tile 29: Checks 1440..1471 (Deg 6)
    {
        { 0x00D80u, 128u, 0u }, // Edge 0: v_start =  432
        { 0x01780u, 128u, 0u }, // Edge 1: v_start =  752
        { 0x01868u, 128u, 0u }, // Edge 2: v_start =  781
    },
    // Tile 30: Checks 1472..1503 (Deg 6)
    {
        { 0x00E80u, 128u, 0u }, // Edge 0: v_start =  464
        { 0x01480u, 128u, 0u }, // Edge 1: v_start =  656
        { 0x01968u, 128u, 0u }, // Edge 2: v_start =  813
    },
    // Tile 31: Checks 1504..1535 (Deg 6)
    {
        { 0x00F80u, 128u, 0u }, // Edge 0: v_start =  496
        { 0x01580u, 128u, 0u }, // Edge 1: v_start =  688
        { 0x01A68u, 128u, 0u }, // Edge 2: v_start =  845
    },
    // Tile 32: Checks    0..  63 (Deg 3)
    {
        { 0x02080u, 128u, 0u }, // Edge 0 (Face 1): v_start = 1040
        { 0x04080u, 128u, 0u }, // Edge 1 (Face 1): v_start = 2064
        { 0x04D00u, 128u, 0u }, // Edge 2 (Face 1): v_start = 2464
    },
    // Tile 33: Checks   64.. 127 (Deg 3)
    {
        { 0x02280u, 128u, 0u }, // Edge 0 (Face 1): v_start = 1104
        { 0x04280u, 128u, 0u }, // Edge 1 (Face 1): v_start = 2128
        { 0x04F00u, 128u, 0u }, // Edge 2 (Face 1): v_start = 2528
    },
    // Tile 34: Checks  128.. 191 (Deg 3)
    {
        { 0x02480u, 128u, 0u }, // Edge 0 (Face 1): v_start = 1168
        { 0x04080u, 128u, 0u }, // Edge 1 (Face 1): v_start = 2064
        { 0x04480u, 128u, 0u }, // Edge 2 (Face 1): v_start = 2192
    },
    // Tile 35: Checks  192.. 255 (Deg 3)
    {
        { 0x02680u, 128u, 0u }, // Edge 0 (Face 1): v_start = 1232
        { 0x04280u, 128u, 0u }, // Edge 1 (Face 1): v_start = 2128
        { 0x04680u, 128u, 0u }, // Edge 2 (Face 1): v_start = 2256
    },
    // Tile 36: Checks  256.. 319 (Deg 3)
    {
        { 0x02880u, 128u, 0u }, // Edge 0 (Face 1): v_start = 1296
        { 0x04480u, 128u, 0u }, // Edge 1 (Face 1): v_start = 2192
        { 0x04880u, 128u, 0u }, // Edge 2 (Face 1): v_start = 2320
    },
    // Tile 37: Checks  320.. 383 (Deg 3)
    {
        { 0x02A80u, 128u, 0u }, // Edge 0 (Face 1): v_start = 1360
        { 0x04680u, 128u, 0u }, // Edge 1 (Face 1): v_start = 2256
        { 0x04A80u, 128u, 0u }, // Edge 2 (Face 1): v_start = 2384
    },
    // Tile 38: Checks  384.. 447 (Deg 3)
    {
        { 0x02C80u, 128u, 0u }, // Edge 0 (Face 1): v_start = 1424
        { 0x04880u, 128u, 0u }, // Edge 1 (Face 1): v_start = 2320
        { 0x04C80u, 128u, 0u }, // Edge 2 (Face 1): v_start = 2448
    },
    // Tile 39: Checks  448.. 511 (Deg 3)
    {
        { 0x02E80u, 128u, 0u }, // Edge 0 (Face 1): v_start = 1488
        { 0x04A80u, 128u, 0u }, // Edge 1 (Face 1): v_start = 2384
        { 0x04E80u, 128u, 0u }, // Edge 2 (Face 1): v_start = 2512
    },
};

constexpr CirculantSliceDesc TILE_SLICES_FACE2_W0[32][3] = {
    // Tile  0: Checks  512.. 543 (Edges 3..5)
    {
        { 0x04338u, 128u, 0u }, // Edge 3: v_start = 2151
        { 0x04748u, 128u, 0u }, // Edge 4: v_start = 2281
        { 0x04800u, 128u, 0u }, // Edge 5: v_start = 2304
    },
    // Tile  1: Checks  544.. 575 (Edges 3..5)
    {
        { 0x04038u, 128u, 0u }, // Edge 3: v_start = 2055
        { 0x04448u, 128u, 0u }, // Edge 4: v_start = 2185
        { 0x04900u, 128u, 0u }, // Edge 5: v_start = 2336
    },
    // Tile  2: Checks  576.. 607 (Edges 3..5)
    {
        { 0x04138u, 128u, 0u }, // Edge 3: v_start = 2087
        { 0x04548u, 128u, 0u }, // Edge 4: v_start = 2217
        { 0x04A00u, 128u, 0u }, // Edge 5: v_start = 2368
    },
    // Tile  3: Checks  608.. 639 (Edges 3..5)
    {
        { 0x04238u, 128u, 0u }, // Edge 3: v_start = 2119
        { 0x04648u, 128u, 0u }, // Edge 4: v_start = 2249
        { 0x04B00u, 128u, 0u }, // Edge 5: v_start = 2400
    },
    // Tile  4: Checks  640.. 671 (Edges 3..5)
    {
        { 0x045A8u, 128u, 0u }, // Edge 3: v_start = 2229
        { 0x04A50u, 128u, 0u }, // Edge 4: v_start = 2378
        { 0x04D68u, 128u, 0u }, // Edge 5: v_start = 2477
    },
    // Tile  5: Checks  672.. 703 (Edges 3..5)
    {
        { 0x046A8u, 128u, 0u }, // Edge 3: v_start = 2261
        { 0x04B50u, 128u, 0u }, // Edge 4: v_start = 2410
        { 0x04E68u, 128u, 0u }, // Edge 5: v_start = 2509
    },
    // Tile  6: Checks  704.. 735 (Edges 3..5)
    {
        { 0x047A8u, 128u, 0u }, // Edge 3: v_start = 2293
        { 0x04850u, 128u, 0u }, // Edge 4: v_start = 2314
        { 0x04F68u, 128u, 0u }, // Edge 5: v_start = 2541
    },
    // Tile  7: Checks  736.. 767 (Edges 3..5)
    {
        { 0x044A8u, 128u, 0u }, // Edge 3: v_start = 2197
        { 0x04950u, 128u, 0u }, // Edge 4: v_start = 2346
        { 0x04C68u, 128u, 0u }, // Edge 5: v_start = 2445
    },
    // Tile  8: Checks  768.. 799 (Edges 3..5)
    {
        { 0x042C8u, 128u, 0u }, // Edge 3: v_start = 2137
        { 0x04840u, 128u, 0u }, // Edge 4: v_start = 2312
        { 0x04FB8u, 128u, 0u }, // Edge 5: v_start = 2551
    },
    // Tile  9: Checks  800.. 831 (Edges 3..5)
    {
        { 0x043C8u, 128u, 0u }, // Edge 3: v_start = 2169
        { 0x04940u, 128u, 0u }, // Edge 4: v_start = 2344
        { 0x04CB8u, 128u, 0u }, // Edge 5: v_start = 2455
    },
    // Tile 10: Checks  832.. 863 (Edges 3..5)
    {
        { 0x040C8u, 128u, 0u }, // Edge 3: v_start = 2073
        { 0x04A40u, 128u, 0u }, // Edge 4: v_start = 2376
        { 0x04DB8u, 128u, 0u }, // Edge 5: v_start = 2487
    },
    // Tile 11: Checks  864.. 895 (Edges 3..5)
    {
        { 0x041C8u, 128u, 0u }, // Edge 3: v_start = 2105
        { 0x04B40u, 128u, 0u }, // Edge 4: v_start = 2408
        { 0x04EB8u, 128u, 0u }, // Edge 5: v_start = 2519
    },
    // Tile 12: Checks  896.. 927 (Edges 3..5)
    {
        { 0x04308u, 128u, 0u }, // Edge 3: v_start = 2145
        { 0x04780u, 128u, 0u }, // Edge 4: v_start = 2288
        { 0x04D18u, 128u, 0u }, // Edge 5: v_start = 2467
    },
    // Tile 13: Checks  928.. 959 (Edges 3..5)
    {
        { 0x04008u, 128u, 0u }, // Edge 3: v_start = 2049
        { 0x04480u, 128u, 0u }, // Edge 4: v_start = 2192
        { 0x04E18u, 128u, 0u }, // Edge 5: v_start = 2499
    },
    // Tile 14: Checks  960.. 991 (Edges 3..5)
    {
        { 0x04108u, 128u, 0u }, // Edge 3: v_start = 2081
        { 0x04580u, 128u, 0u }, // Edge 4: v_start = 2224
        { 0x04F18u, 128u, 0u }, // Edge 5: v_start = 2531
    },
    // Tile 15: Checks  992..1023 (Edges 3..5)
    {
        { 0x04208u, 128u, 0u }, // Edge 3: v_start = 2113
        { 0x04680u, 128u, 0u }, // Edge 4: v_start = 2256
        { 0x04C18u, 128u, 0u }, // Edge 5: v_start = 2435
    },
    // Tile 16: Checks 1024..1055 (Edges 3..5)
    {
        { 0x03398u, 128u, 0u }, // Edge 3: v_start = 1651
        { 0x034F0u, 128u, 0u }, // Edge 4: v_start = 1694
        { 0x04000u, 128u, 0u }, // Edge 5: v_start = 2048
    },
    // Tile 17: Checks 1056..1087 (Edges 3..5)
    {
        { 0x03098u, 128u, 0u }, // Edge 3: v_start = 1555
        { 0x035F0u, 128u, 0u }, // Edge 4: v_start = 1726
        { 0x04100u, 128u, 0u }, // Edge 5: v_start = 2080
    },
    // Tile 18: Checks 1088..1119 (Edges 3..5)
    {
        { 0x03198u, 128u, 0u }, // Edge 3: v_start = 1587
        { 0x036F0u, 128u, 0u }, // Edge 4: v_start = 1758
        { 0x04200u, 128u, 0u }, // Edge 5: v_start = 2112
    },
    // Tile 19: Checks 1120..1151 (Edges 3..5)
    {
        { 0x03298u, 128u, 0u }, // Edge 3: v_start = 1619
        { 0x037F0u, 128u, 0u }, // Edge 4: v_start = 1790
        { 0x04300u, 128u, 0u }, // Edge 5: v_start = 2144
    },
    // Tile 20: Checks 1152..1183 (Edges 3..5)
    {
        { 0x035D8u, 128u, 0u }, // Edge 3: v_start = 1723
        { 0x03B30u, 128u, 0u }, // Edge 4: v_start = 1894
        { 0x04400u, 128u, 0u }, // Edge 5: v_start = 2176
    },
    // Tile 21: Checks 1184..1215 (Edges 3..5)
    {
        { 0x036D8u, 128u, 0u }, // Edge 3: v_start = 1755
        { 0x03830u, 128u, 0u }, // Edge 4: v_start = 1798
        { 0x04500u, 128u, 0u }, // Edge 5: v_start = 2208
    },
    // Tile 22: Checks 1216..1247 (Edges 3..5)
    {
        { 0x037D8u, 128u, 0u }, // Edge 3: v_start = 1787
        { 0x03930u, 128u, 0u }, // Edge 4: v_start = 1830
        { 0x04600u, 128u, 0u }, // Edge 5: v_start = 2240
    },
    // Tile 23: Checks 1248..1279 (Edges 3..5)
    {
        { 0x034D8u, 128u, 0u }, // Edge 3: v_start = 1691
        { 0x03A30u, 128u, 0u }, // Edge 4: v_start = 1862
        { 0x04700u, 128u, 0u }, // Edge 5: v_start = 2272
    },
    // Tile 24: Checks 1280..1311 (Edges 3..5)
    {
        { 0x03808u, 128u, 0u }, // Edge 3: v_start = 1793
        { 0x03E28u, 128u, 0u }, // Edge 4: v_start = 1989
        { 0x04800u, 128u, 0u }, // Edge 5: v_start = 2304
    },
    // Tile 25: Checks 1312..1343 (Edges 3..5)
    {
        { 0x03908u, 128u, 0u }, // Edge 3: v_start = 1825
        { 0x03F28u, 128u, 0u }, // Edge 4: v_start = 2021
        { 0x04900u, 128u, 0u }, // Edge 5: v_start = 2336
    },
    // Tile 26: Checks 1344..1375 (Edges 3..5)
    {
        { 0x03A08u, 128u, 0u }, // Edge 3: v_start = 1857
        { 0x03C28u, 128u, 0u }, // Edge 4: v_start = 1925
        { 0x04A00u, 128u, 0u }, // Edge 5: v_start = 2368
    },
    // Tile 27: Checks 1376..1407 (Edges 3..5)
    {
        { 0x03B08u, 128u, 0u }, // Edge 3: v_start = 1889
        { 0x03D28u, 128u, 0u }, // Edge 4: v_start = 1957
        { 0x04B00u, 128u, 0u }, // Edge 5: v_start = 2400
    },
    // Tile 28: Checks 1408..1439 (Edges 3..5)
    {
        { 0x032F0u, 128u, 0u }, // Edge 3: v_start = 1630
        { 0x03F18u, 128u, 0u }, // Edge 4: v_start = 2019
        { 0x04C00u, 128u, 0u }, // Edge 5: v_start = 2432
    },
    // Tile 29: Checks 1440..1471 (Edges 3..5)
    {
        { 0x033F0u, 128u, 0u }, // Edge 3: v_start = 1662
        { 0x03C18u, 128u, 0u }, // Edge 4: v_start = 1923
        { 0x04D00u, 128u, 0u }, // Edge 5: v_start = 2464
    },
    // Tile 30: Checks 1472..1503 (Edges 3..5)
    {
        { 0x030F0u, 128u, 0u }, // Edge 3: v_start = 1566
        { 0x03D18u, 128u, 0u }, // Edge 4: v_start = 1955
        { 0x04E00u, 128u, 0u }, // Edge 5: v_start = 2496
    },
    // Tile 31: Checks 1504..1535 (Edges 3..5)
    {
        { 0x031F0u, 128u, 0u }, // Edge 3: v_start = 1598
        { 0x03E18u, 128u, 0u }, // Edge 4: v_start = 1987
        { 0x04F00u, 128u, 0u }, // Edge 5: v_start = 2528
    },
};

constexpr CirculantSliceDesc TILE_SLICES_FACE3_W1[32][3] = {
    // Tile  0: Checks  512.. 543 (Edges 3..5)
    {
        { 0x043B8u, 128u, 0u }, // Edge 3: v_start = 2167
        { 0x047C8u, 128u, 0u }, // Edge 4: v_start = 2297
        { 0x04880u, 128u, 0u }, // Edge 5: v_start = 2320
    },
    // Tile  1: Checks  544.. 575 (Edges 3..5)
    {
        { 0x040B8u, 128u, 0u }, // Edge 3: v_start = 2071
        { 0x044C8u, 128u, 0u }, // Edge 4: v_start = 2201
        { 0x04980u, 128u, 0u }, // Edge 5: v_start = 2352
    },
    // Tile  2: Checks  576.. 607 (Edges 3..5)
    {
        { 0x041B8u, 128u, 0u }, // Edge 3: v_start = 2103
        { 0x045C8u, 128u, 0u }, // Edge 4: v_start = 2233
        { 0x04A80u, 128u, 0u }, // Edge 5: v_start = 2384
    },
    // Tile  3: Checks  608.. 639 (Edges 3..5)
    {
        { 0x042B8u, 128u, 0u }, // Edge 3: v_start = 2135
        { 0x046C8u, 128u, 0u }, // Edge 4: v_start = 2265
        { 0x04B80u, 128u, 0u }, // Edge 5: v_start = 2416
    },
    // Tile  4: Checks  640.. 671 (Edges 3..5)
    {
        { 0x04628u, 128u, 0u }, // Edge 3: v_start = 2245
        { 0x04AD0u, 128u, 0u }, // Edge 4: v_start = 2394
        { 0x04DE8u, 128u, 0u }, // Edge 5: v_start = 2493
    },
    // Tile  5: Checks  672.. 703 (Edges 3..5)
    {
        { 0x04728u, 128u, 0u }, // Edge 3: v_start = 2277
        { 0x04BD0u, 128u, 0u }, // Edge 4: v_start = 2426
        { 0x04EE8u, 128u, 0u }, // Edge 5: v_start = 2525
    },
    // Tile  6: Checks  704.. 735 (Edges 3..5)
    {
        { 0x04428u, 128u, 0u }, // Edge 3: v_start = 2181
        { 0x048D0u, 128u, 0u }, // Edge 4: v_start = 2330
        { 0x04FE8u, 128u, 0u }, // Edge 5: v_start = 2557
    },
    // Tile  7: Checks  736.. 767 (Edges 3..5)
    {
        { 0x04528u, 128u, 0u }, // Edge 3: v_start = 2213
        { 0x049D0u, 128u, 0u }, // Edge 4: v_start = 2362
        { 0x04CE8u, 128u, 0u }, // Edge 5: v_start = 2461
    },
    // Tile  8: Checks  768.. 799 (Edges 3..5)
    {
        { 0x04348u, 128u, 0u }, // Edge 3: v_start = 2153
        { 0x048C0u, 128u, 0u }, // Edge 4: v_start = 2328
        { 0x04C38u, 128u, 0u }, // Edge 5: v_start = 2439
    },
    // Tile  9: Checks  800.. 831 (Edges 3..5)
    {
        { 0x04048u, 128u, 0u }, // Edge 3: v_start = 2057
        { 0x049C0u, 128u, 0u }, // Edge 4: v_start = 2360
        { 0x04D38u, 128u, 0u }, // Edge 5: v_start = 2471
    },
    // Tile 10: Checks  832.. 863 (Edges 3..5)
    {
        { 0x04148u, 128u, 0u }, // Edge 3: v_start = 2089
        { 0x04AC0u, 128u, 0u }, // Edge 4: v_start = 2392
        { 0x04E38u, 128u, 0u }, // Edge 5: v_start = 2503
    },
    // Tile 11: Checks  864.. 895 (Edges 3..5)
    {
        { 0x04248u, 128u, 0u }, // Edge 3: v_start = 2121
        { 0x04BC0u, 128u, 0u }, // Edge 4: v_start = 2424
        { 0x04F38u, 128u, 0u }, // Edge 5: v_start = 2535
    },
    // Tile 12: Checks  896.. 927 (Edges 3..5)
    {
        { 0x04388u, 128u, 0u }, // Edge 3: v_start = 2161
        { 0x04400u, 128u, 0u }, // Edge 4: v_start = 2176
        { 0x04D98u, 128u, 0u }, // Edge 5: v_start = 2483
    },
    // Tile 13: Checks  928.. 959 (Edges 3..5)
    {
        { 0x04088u, 128u, 0u }, // Edge 3: v_start = 2065
        { 0x04500u, 128u, 0u }, // Edge 4: v_start = 2208
        { 0x04E98u, 128u, 0u }, // Edge 5: v_start = 2515
    },
    // Tile 14: Checks  960.. 991 (Edges 3..5)
    {
        { 0x04188u, 128u, 0u }, // Edge 3: v_start = 2097
        { 0x04600u, 128u, 0u }, // Edge 4: v_start = 2240
        { 0x04F98u, 128u, 0u }, // Edge 5: v_start = 2547
    },
    // Tile 15: Checks  992..1023 (Edges 3..5)
    {
        { 0x04288u, 128u, 0u }, // Edge 3: v_start = 2129
        { 0x04700u, 128u, 0u }, // Edge 4: v_start = 2272
        { 0x04C98u, 128u, 0u }, // Edge 5: v_start = 2451
    },
    // Tile 16: Checks 1024..1055 (Edges 3..5)
    {
        { 0x03018u, 128u, 0u }, // Edge 3: v_start = 1539
        { 0x03570u, 128u, 0u }, // Edge 4: v_start = 1710
        { 0x04080u, 128u, 0u }, // Edge 5: v_start = 2064
    },
    // Tile 17: Checks 1056..1087 (Edges 3..5)
    {
        { 0x03118u, 128u, 0u }, // Edge 3: v_start = 1571
        { 0x03670u, 128u, 0u }, // Edge 4: v_start = 1742
        { 0x04180u, 128u, 0u }, // Edge 5: v_start = 2096
    },
    // Tile 18: Checks 1088..1119 (Edges 3..5)
    {
        { 0x03218u, 128u, 0u }, // Edge 3: v_start = 1603
        { 0x03770u, 128u, 0u }, // Edge 4: v_start = 1774
        { 0x04280u, 128u, 0u }, // Edge 5: v_start = 2128
    },
    // Tile 19: Checks 1120..1151 (Edges 3..5)
    {
        { 0x03318u, 128u, 0u }, // Edge 3: v_start = 1635
        { 0x03470u, 128u, 0u }, // Edge 4: v_start = 1678
        { 0x04380u, 128u, 0u }, // Edge 5: v_start = 2160
    },
    // Tile 20: Checks 1152..1183 (Edges 3..5)
    {
        { 0x03658u, 128u, 0u }, // Edge 3: v_start = 1739
        { 0x03BB0u, 128u, 0u }, // Edge 4: v_start = 1910
        { 0x04480u, 128u, 0u }, // Edge 5: v_start = 2192
    },
    // Tile 21: Checks 1184..1215 (Edges 3..5)
    {
        { 0x03758u, 128u, 0u }, // Edge 3: v_start = 1771
        { 0x038B0u, 128u, 0u }, // Edge 4: v_start = 1814
        { 0x04580u, 128u, 0u }, // Edge 5: v_start = 2224
    },
    // Tile 22: Checks 1216..1247 (Edges 3..5)
    {
        { 0x03458u, 128u, 0u }, // Edge 3: v_start = 1675
        { 0x039B0u, 128u, 0u }, // Edge 4: v_start = 1846
        { 0x04680u, 128u, 0u }, // Edge 5: v_start = 2256
    },
    // Tile 23: Checks 1248..1279 (Edges 3..5)
    {
        { 0x03558u, 128u, 0u }, // Edge 3: v_start = 1707
        { 0x03AB0u, 128u, 0u }, // Edge 4: v_start = 1878
        { 0x04780u, 128u, 0u }, // Edge 5: v_start = 2288
    },
    // Tile 24: Checks 1280..1311 (Edges 3..5)
    {
        { 0x03888u, 128u, 0u }, // Edge 3: v_start = 1809
        { 0x03EA8u, 128u, 0u }, // Edge 4: v_start = 2005
        { 0x04880u, 128u, 0u }, // Edge 5: v_start = 2320
    },
    // Tile 25: Checks 1312..1343 (Edges 3..5)
    {
        { 0x03988u, 128u, 0u }, // Edge 3: v_start = 1841
        { 0x03FA8u, 128u, 0u }, // Edge 4: v_start = 2037
        { 0x04980u, 128u, 0u }, // Edge 5: v_start = 2352
    },
    // Tile 26: Checks 1344..1375 (Edges 3..5)
    {
        { 0x03A88u, 128u, 0u }, // Edge 3: v_start = 1873
        { 0x03CA8u, 128u, 0u }, // Edge 4: v_start = 1941
        { 0x04A80u, 128u, 0u }, // Edge 5: v_start = 2384
    },
    // Tile 27: Checks 1376..1407 (Edges 3..5)
    {
        { 0x03B88u, 128u, 0u }, // Edge 3: v_start = 1905
        { 0x03DA8u, 128u, 0u }, // Edge 4: v_start = 1973
        { 0x04B80u, 128u, 0u }, // Edge 5: v_start = 2416
    },
    // Tile 28: Checks 1408..1439 (Edges 3..5)
    {
        { 0x03370u, 128u, 0u }, // Edge 3: v_start = 1646
        { 0x03F98u, 128u, 0u }, // Edge 4: v_start = 2035
        { 0x04C80u, 128u, 0u }, // Edge 5: v_start = 2448
    },
    // Tile 29: Checks 1440..1471 (Edges 3..5)
    {
        { 0x03070u, 128u, 0u }, // Edge 3: v_start = 1550
        { 0x03C98u, 128u, 0u }, // Edge 4: v_start = 1939
        { 0x04D80u, 128u, 0u }, // Edge 5: v_start = 2480
    },
    // Tile 30: Checks 1472..1503 (Edges 3..5)
    {
        { 0x03170u, 128u, 0u }, // Edge 3: v_start = 1582
        { 0x03D98u, 128u, 0u }, // Edge 4: v_start = 1971
        { 0x04E80u, 128u, 0u }, // Edge 5: v_start = 2512
    },
    // Tile 31: Checks 1504..1535 (Edges 3..5)
    {
        { 0x03270u, 128u, 0u }, // Edge 3: v_start = 1614
        { 0x03E98u, 128u, 0u }, // Edge 4: v_start = 2003
        { 0x04F80u, 128u, 0u }, // Edge 5: v_start = 2544
    },
};

constexpr CirculantSliceDesc TILE_SLICES_DEG3_F2_W0[8][3] = {
    // Tile 32: Checks   32..  47 (Face 2)
    {
        { 0x02100u, 128u, 0u }, // Edge 0 (Face 2): v_start = 1056
        { 0x04100u, 128u, 0u }, // Edge 1 (Face 2): v_start = 2080
        { 0x04D80u, 128u, 0u }, // Edge 2 (Face 2): v_start = 2480
    },
    // Tile 33: Checks   96.. 111 (Face 2)
    {
        { 0x02300u, 128u, 0u }, // Edge 0 (Face 2): v_start = 1120
        { 0x04300u, 128u, 0u }, // Edge 1 (Face 2): v_start = 2144
        { 0x04F80u, 128u, 0u }, // Edge 2 (Face 2): v_start = 2544
    },
    // Tile 34: Checks  160.. 175 (Face 2)
    {
        { 0x02500u, 128u, 0u }, // Edge 0 (Face 2): v_start = 1184
        { 0x04100u, 128u, 0u }, // Edge 1 (Face 2): v_start = 2080
        { 0x04500u, 128u, 0u }, // Edge 2 (Face 2): v_start = 2208
    },
    // Tile 35: Checks  224.. 239 (Face 2)
    {
        { 0x02700u, 128u, 0u }, // Edge 0 (Face 2): v_start = 1248
        { 0x04300u, 128u, 0u }, // Edge 1 (Face 2): v_start = 2144
        { 0x04700u, 128u, 0u }, // Edge 2 (Face 2): v_start = 2272
    },
    // Tile 36: Checks  288.. 303 (Face 2)
    {
        { 0x02900u, 128u, 0u }, // Edge 0 (Face 2): v_start = 1312
        { 0x04500u, 128u, 0u }, // Edge 1 (Face 2): v_start = 2208
        { 0x04900u, 128u, 0u }, // Edge 2 (Face 2): v_start = 2336
    },
    // Tile 37: Checks  352.. 367 (Face 2)
    {
        { 0x02B00u, 128u, 0u }, // Edge 0 (Face 2): v_start = 1376
        { 0x04700u, 128u, 0u }, // Edge 1 (Face 2): v_start = 2272
        { 0x04B00u, 128u, 0u }, // Edge 2 (Face 2): v_start = 2400
    },
    // Tile 38: Checks  416.. 431 (Face 2)
    {
        { 0x02D00u, 128u, 0u }, // Edge 0 (Face 2): v_start = 1440
        { 0x04900u, 128u, 0u }, // Edge 1 (Face 2): v_start = 2336
        { 0x04D00u, 128u, 0u }, // Edge 2 (Face 2): v_start = 2464
    },
    // Tile 39: Checks  480.. 495 (Face 2)
    {
        { 0x02F00u, 128u, 0u }, // Edge 0 (Face 2): v_start = 1504
        { 0x04B00u, 128u, 0u }, // Edge 1 (Face 2): v_start = 2400
        { 0x04F00u, 128u, 0u }, // Edge 2 (Face 2): v_start = 2528
    },
};

constexpr CirculantSliceDesc TILE_SLICES_DEG3_F3_W1[8][3] = {
    // Tile 32: Checks   48..  63 (Face 3)
    {
        { 0x02180u, 128u, 0u }, // Edge 0 (Face 3): v_start = 1072
        { 0x04180u, 128u, 0u }, // Edge 1 (Face 3): v_start = 2096
        { 0x04E00u, 128u, 0u }, // Edge 2 (Face 3): v_start = 2496
    },
    // Tile 33: Checks  112.. 127 (Face 3)
    {
        { 0x02380u, 128u, 0u }, // Edge 0 (Face 3): v_start = 1136
        { 0x04380u, 128u, 0u }, // Edge 1 (Face 3): v_start = 2160
        { 0x04C00u, 128u, 0u }, // Edge 2 (Face 3): v_start = 2432
    },
    // Tile 34: Checks  176.. 191 (Face 3)
    {
        { 0x02580u, 128u, 0u }, // Edge 0 (Face 3): v_start = 1200
        { 0x04180u, 128u, 0u }, // Edge 1 (Face 3): v_start = 2096
        { 0x04580u, 128u, 0u }, // Edge 2 (Face 3): v_start = 2224
    },
    // Tile 35: Checks  240.. 255 (Face 3)
    {
        { 0x02780u, 128u, 0u }, // Edge 0 (Face 3): v_start = 1264
        { 0x04380u, 128u, 0u }, // Edge 1 (Face 3): v_start = 2160
        { 0x04780u, 128u, 0u }, // Edge 2 (Face 3): v_start = 2288
    },
    // Tile 36: Checks  304.. 319 (Face 3)
    {
        { 0x02980u, 128u, 0u }, // Edge 0 (Face 3): v_start = 1328
        { 0x04580u, 128u, 0u }, // Edge 1 (Face 3): v_start = 2224
        { 0x04980u, 128u, 0u }, // Edge 2 (Face 3): v_start = 2352
    },
    // Tile 37: Checks  368.. 383 (Face 3)
    {
        { 0x02B80u, 128u, 0u }, // Edge 0 (Face 3): v_start = 1392
        { 0x04780u, 128u, 0u }, // Edge 1 (Face 3): v_start = 2288
        { 0x04B80u, 128u, 0u }, // Edge 2 (Face 3): v_start = 2416
    },
    // Tile 38: Checks  432.. 447 (Face 3)
    {
        { 0x02D80u, 128u, 0u }, // Edge 0 (Face 3): v_start = 1456
        { 0x04980u, 128u, 0u }, // Edge 1 (Face 3): v_start = 2352
        { 0x04D80u, 128u, 0u }, // Edge 2 (Face 3): v_start = 2480
    },
    // Tile 39: Checks  496.. 511 (Face 3)
    {
        { 0x02F80u, 128u, 0u }, // Edge 0 (Face 3): v_start = 1520
        { 0x04B80u, 128u, 0u }, // Edge 1 (Face 3): v_start = 2416
        { 0x04F80u, 128u, 0u }, // Edge 2 (Face 3): v_start = 2544
    },
};

} // namespace QC_LDPC_4CW
