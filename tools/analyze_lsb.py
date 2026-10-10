import struct
import numpy as np

golden_bytes = open("dump/golden_final_vnodes.bin", "rb").read()[:20480]
tensix_bytes = open("dump/tensix_final_vnodes.bin", "rb").read()[:20480]

g_words = np.frombuffer(golden_bytes, dtype=np.uint16)
t_words = np.frombuffer(tensix_bytes, dtype=np.uint16)

diffs = g_words != t_words
mismatches = np.where(diffs)[0]

print(f"Total words: {len(g_words)}")
print(f"Total matching: {np.sum(~diffs)} ({np.sum(~diffs)/len(g_words)*100:.2f}%)")
print(f"Total differing: {len(mismatches)} ({len(mismatches)/len(g_words)*100:.2f}%)")

lsb_diffs = []
for idx in mismatches:
    g = int(g_words[idx])
    t = int(t_words[idx])
    lsb_diffs.append(g - t)

from collections import Counter
counts = Counter(lsb_diffs)
print("\nDistribution of (Golden_uint16 - Tensix_uint16):")
for d, c in sorted(counts.items()):
    print(f"  diff {d:+d}: {c} occurrences ({c/len(mismatches)*100:.2f}%)")
