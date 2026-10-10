import subprocess

def main():
    cmd = "grep -rnI 'set_clock_state\\|ClockState::BUSY\\|AICLK_GO_BUSY' /home/ttuser/tt-metal/ 2>/dev/null | head -n 30"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("=== Matches for set_clock_state ===")
    print(res.stdout)

if __name__ == "__main__":
    main()
