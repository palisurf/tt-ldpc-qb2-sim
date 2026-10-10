#!/bin/bash
find /home/ttuser/tt-metal/ -name "*isa*" -o -name "*architecture*" -o -name "*tensix*" | grep -E "\.(md|rst|pdf|txt)$" | head -n 30
