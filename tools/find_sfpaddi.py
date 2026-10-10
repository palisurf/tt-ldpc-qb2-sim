import subprocess

def search():
    res = subprocess.run("grep -rn 'sfpaddi' /home/ttuser/tt-metal/ 2>/dev/null | head -n 30", shell=True, capture_output=True, text=True)
    print("sfpaddi occurrences:")
    print(res.stdout)

if __name__ == "__main__":
    search()
