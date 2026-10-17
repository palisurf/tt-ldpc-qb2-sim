def ams(a, b):
    diff = abs(a - b)
    line1 = 0.69315 - 0.4 * diff
    line2 = 0.65715 - 0.328 * diff
    corr = max(max(line1, line2), 0.0)
    return max(min(a, b) - corr, 0.0)

reg2_abs = [2.0, 2.0, 2.0, 2.0, 3.5, 3.5, 2.0, 2.0, 3.25, 3.25, 2.0, 2.0, 2.0, 2.0, 3.25, 3.25,
            2.0, 2.0, 3.0, 3.0, 2.0, 2.0, 3.0, 3.0, 2.25, 2.25, 3.0, 3.0, 3.5, 3.5, 3.25, 3.25]

print("Searching for pairs with ams ~ 2.797:")
for i in range(32):
    for j in range(32):
        val = ams(reg2_abs[i], reg2_abs[j])
        if abs(val - 2.797) < 0.01:
            print(f"ams(i={i}:{reg2_abs[i]}, j={j}:{reg2_abs[j]}) = {val:.4f}")

print("\nSearching for pairs with ams ~ 2.547:")
for i in range(32):
    for j in range(32):
        val = ams(reg2_abs[i], reg2_abs[j])
        if abs(val - 2.547) < 0.01:
            print(f"ams(i={i}:{reg2_abs[i]}, j={j}:{reg2_abs[j]}) = {val:.4f}")
