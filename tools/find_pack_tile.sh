#!/bin/bash
find /home/ttuser/tt-metal/ -name "*pack*.h" | grep blackhole | xargs grep -rn "pack_tile" | head -n 30
