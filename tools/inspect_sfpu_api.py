import os

def examine_file(path, keywords):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    print(f"=== {os.path.basename(path)} ===")
    with open(path, "r") as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        if any(k in line for k in keywords):
            start = max(0, idx - 2)
            end = min(len(lines), idx + 10)
            for j in range(start, end):
                print(f"{j+1:4d}: {lines[j].rstrip()}")
            print("-" * 50)

def main():
    base = "/home/ttuser/tt-metal/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/"
    examine_file(base + "ckernel_sfpu_abs.h", ["_abs", "sfpu::", "calculate_abs"])
    examine_file(base + "ckernel_sfpu_binary_max_min.h", ["_min", "_max", "calculate_min"])
    examine_file(base + "ckernel_sfpu_binary.h", ["_sub", "_add", "_mul"])

if __name__ == "__main__":
    main()
