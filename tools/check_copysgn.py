def check():
    with open("/home/ttuser/tt-metal/runtime/sfpi/include/sfpi_lib.h") as fp:
        lines = fp.readlines()
    for i, line in enumerate(lines):
        if "copysgn" in line:
            for j in range(max(0, i-2), min(len(lines), i+15)):
                print(f"{j+1}: {lines[j].rstrip()}")

if __name__ == "__main__":
    check()
