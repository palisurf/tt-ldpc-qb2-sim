def amin(a, b):
    diff = abs(a - b)
    line1 = 0.69315 - 0.4000 * diff
    line2 = 0.65715 - 0.3280 * diff
    corr = max(line1, line2, 0.0)
    return max(min(a, b) - corr, 0.0)

# Lanes 0..31 of reg 0, reg 2, reg 4 for Col 0:
# Lane 0: Row 0 = 3.9844
# Lane 8: Row 1 = 4.1250
# Lane 16: Row 2 = 2.3438
# Lane 24: Row 3 = 4.9375

# In reg 0:
q0 = [3.9844]*16 + [2.3438]*16 # lane 0 and lane 16
q1 = [2.8438]*16 + [0.0]*16
q2 = [0.0]*32

# If swap16 swaps [0..15] with [16..31]:
# Step 1:
m0 = [abs(x) for x in q0]
m0_swap = m0[16:] + m0[:16]
pair_m0 = [amin(m0[i], m0_swap[i]) for i in range(32)]
print("pair_m0[0]:", pair_m0[0], "pair_m0[16]:", pair_m0[16])

# Step 2:
m1 = [abs(x) for x in q1]
m1_swap = m1[16:] + m1[:16]
pair_m1 = [amin(m1[i], m1_swap[i]) for i in range(32)]
print("pair_m1[0]:", pair_m1[0], "pair_m1[16]:", pair_m1[16])

# Step 3:
pair_m2 = [0.0]*32

# Step 4:
m_other_0 = [amin(pair_m1[i], pair_m2[i]) for i in range(32)]
print("m_other_0[0]:", m_other_0[0], "m_other_0[16]:", m_other_0[16])

# Step 5:
r_mag0 = [amin(m_other_0[i], m0_swap[i]) for i in range(32)]
print("r_mag0[0]:", r_mag0[0], "r_mag0[16]:", r_mag0[16])
