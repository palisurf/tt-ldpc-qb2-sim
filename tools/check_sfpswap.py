import glob

def check():
    files = glob.glob("/home/ttuser/tt-metal/runtime/sfpi/include/*.h")
    for f in files:
        with open(f) as fp:
            lines = fp.readlines()
        for i, line in enumerate(lines):
            if "sfpswap" in line or "SFPSWAP" in line:
                for j in range(max(0, i-2), min(len(lines), i+8)):
                    print(f"{f}:{j+1}: {lines[j].rstrip()}")

if __name__ == "__main__":
    check()
