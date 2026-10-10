#!/bin/bash
echo "=== Searching for unpacker documentation and headers ==="
find /home/ttuser/tt-metal/ -name "dprint_tensix_unpack.h" -o -name "llk_unpack_A_api.h"
echo "=== DPrint Tensix Unpack Header ==="
cat /home/ttuser/tt-metal/build_Release/libexec/tt-metalium/tt_metal/hw/inc/api/debug/dprint_tensix_unpack.h
