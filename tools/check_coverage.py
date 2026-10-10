# Check if all checks are executed in schedule
# Let's inspect the code in ldpc_sim_tile_matrix.cpp
# How many tiles are in deg6_tiles and deg3_tiles?
# deg6_tiles has 26 tiles?
# CHECKS_PER_TILE_DEG6 = 40 checks per tile?
# 26 tiles * 40 checks = 1040 checks. 1024 are real, 16 are padding.
# CHECKS_PER_TILE_DEG3 = 80 checks per tile?
# 7 tiles * 80 checks = 560 checks. 512 are real, 48 are padding.
# Total tiles = 33 tiles.
print("Checking tile schedule...")
