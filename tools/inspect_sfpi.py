import subprocess

def main():
    cmd = "grep -rnI 'TT_SFPLOAD' /home/ttuser/tt-metal/ 2>/dev/null | head -n 20"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("=== TT_SFPLOAD in tt-metal ===")
    print(res.stdout)

    cmd = "grep -rnI 'TT_SFPSTORE' /home/ttuser/tt-metal/ 2>/dev/null | head -n 20"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("=== TT_SFPSTORE in tt-metal ===")
    print(res.stdout)

if __name__ == "__main__":
    main()
