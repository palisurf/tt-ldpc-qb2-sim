import subprocess

def check():
    f = "/home/ttuser/.cache/tt-metal-cache/4307071483051000317/kernels/compute_trisc/12172814604445604201/trisc1/trisc1.elf"
    objdump = "/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-objdump"
    res = subprocess.run(f"{objdump} -d -C {f}", shell=True, capture_output=True, text=True)
    lines = res.stdout.splitlines()
    
    start = False
    count = 0
    for line in lines:
        if "7820:" in line or "7824:" in line:
            start = True
        if start:
            print(line)
            count += 1
            if count > 100:
                break

if __name__ == "__main__":
    check()
