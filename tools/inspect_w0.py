import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inspect_tile0_h import w0_pairs
for i in range(10):
    val = w0_pairs[i]
    low = val & 0xffff
    high = (val >> 16) & 0xffff
    print("w0_pairs[%d] = 0x%08x (low=0x%04x, high=0x%04x)" % (i, val, low, high))
