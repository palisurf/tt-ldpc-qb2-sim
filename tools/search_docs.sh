#!/bin/bash
echo "=== Searching for documentation on tile layout / SFPU ==="
find /home/ttuser/tt-metal/ -name "*.md" -o -name "*.rst" | xargs grep -in "32x32" | head -n 30
echo "=== Searching for face layout in docs ==="
find /home/ttuser/tt-metal/ -name "*.md" -o -name "*.rst" | xargs grep -in "16x16" | head -n 30
echo "=== Searching for dst_reg layout ==="
find /home/ttuser/tt-metal/ -name "*.md" -o -name "*.rst" | xargs grep -in "vector register" | head -n 30
