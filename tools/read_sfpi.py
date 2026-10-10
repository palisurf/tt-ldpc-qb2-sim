import glob

def search():
    files = glob.glob("/home/ttuser/tt-metal/runtime/sfpi/include/*.h")
    for f in sorted(files):
        with open(f, 'r') as fp:
            lines = fp.readlines()
        for i, line in enumerate(lines):
            if "flt_add" in line or "flt_mul" in line or "min_max" in line or "sFloat16b" in line:
                print(f"{f}:{i+1}: {line.strip()}")
                for j in range(max(0, i-2), min(len(lines), i+8)):
                    print(f"   {j+1}: {lines[j].rstrip()}")

if __name__ == "__main__":
    search()
