#!/bin/bash
grep -rn "ALL_INTF_ACTIVE" /home/ttuser/tt-metal/tt_metal/ /home/ttuser/tt-metal/docs/ 2>/dev/null | head -n 30
