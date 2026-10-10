import subprocess

def find_tools():
    res = subprocess.run("which objdump; find / -name '*riscv*objdump*' 2>/dev/null", shell=True, capture_output=True, text=True)
    print("Found objdump:")
    print(res.stdout)

if __name__ == "__main__":
    find_tools()
