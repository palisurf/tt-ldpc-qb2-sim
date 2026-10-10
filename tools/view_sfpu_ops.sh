#!/bin/bash
find /home/ttuser/tt-metal/tt_metal/tt-llk/tt_llk_blackhole/ -name "*sfpu*.h" | xargs grep -in "dst_reg" | head -n 30
