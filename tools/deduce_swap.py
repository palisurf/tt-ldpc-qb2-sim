def ams(a, b):
    diff = abs(a - b)
    line1 = 0.69315 - 0.4 * diff
    line2 = 0.65715 - 0.328 * diff
    corr = max(max(line1, line2), 0.0)
    return max(min(a, b) - corr, 0.0)

m1 = [2.0, 2.0, 2.0, 2.0, 3.5, 3.5, 2.0, 2.0, 3.25, 3.25, 2.0, 2.0, 2.0, 2.0, 3.25, 3.25,
      2.0, 2.0, 3.0, 3.0, 2.0, 2.0, 3.0, 3.0, 2.25, 2.25, 3.0, 3.0, 3.5, 3.5, 3.25, 3.25]

tensix_out = [
    1.305, 1.305, 1.664, 1.664, 1.828, 1.828, 1.664, 1.664, 1.914, 1.914, 1.664, 1.664, 1.828, 1.828, 2.547, 2.547,
    1.305, 1.305, 1.305, 1.305, 2.797, 2.797, 1.305, 1.305, 2.547, 2.547, 1.305, 1.305, 1.305, 1.305, 2.547, 2.547
]

print("Deducing what swap_subvec16(m1)[lane] was for lanes 16..31:")
for lane in range(16, 32):
    my_val = m1[lane]
    t_val = tensix_out[lane]
    # search through all lanes j to find which m1[j] gives t_val when paired with my_val
    matches = []
    for j in range(32):
        if abs(ams(my_val, m1[j]) - t_val) < 0.02:
            matches.append(j)
    print(f"Lane {lane} (m1={my_val}): Tensix={t_val} => matched m1 of lanes: {matches}")
