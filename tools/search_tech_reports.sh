#!/bin/bash
echo "=== Tech reports search ==="
find /home/ttuser/tt-metal/tech_reports/ -type f | xargs grep -in "unpack" | head -n 30
echo "=== Docs search ==="
find /home/ttuser/tt-metal/docs/ -type f | xargs grep -in "unpacker" | head -n 30
