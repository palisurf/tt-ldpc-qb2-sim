with open("/home/ttuser/tt-metal/runtime/sfpi/include/sfpi_funcs.h") as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if "sfpaddi" in line or "sfpmuli" in line:
        for j in range(max(0, i-3), min(len(lines), i+8)):
            print(f"{j+1}: {lines[j].rstrip()}")
