import glob

def check():
    files = glob.glob("/home/ttuser/tt-metal/runtime/sfpi/include/*.h")
    for f in files:
        with open(f) as fp:
            text = fp.read()
        if "RoundMode" in text or "round" in text.lower():
            for line in text.splitlines():
                if "round" in line.lower() and ("mode" in line.lower() or "sfp" in line.lower()):
                    print(f"{f}: {line}")

if __name__ == "__main__":
    check()
