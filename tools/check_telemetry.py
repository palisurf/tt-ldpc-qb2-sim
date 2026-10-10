import subprocess
import json

def main():
    try:
        res = subprocess.run(["tt-smi", "-s"], capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
    except Exception as e:
        print(f"Error invoking tt-smi: {e}")
        return

    print("=== Tenstorrent Blackhole Telemetry Snapshot ===")
    for dev in data.get("device_info", []):
        dev_id = dev.get("device_id")
        board = dev.get("board_info", {})
        bus = board.get("bus_id", "N/A")
        board_type = board.get("board_type", "N/A")
        tel = dev.get("telemetry", {})
        aiclk = tel.get("aiclk", "N/A")
        power = tel.get("power", "N/A")
        bpower = tel.get("board_power", "N/A")
        temp = tel.get("asic_temperature", "N/A")
        cur = tel.get("current", "N/A")
        volt = tel.get("voltage", "N/A")
        print(f"Device {dev_id} ({board_type} @ {bus}):")
        print(f"  AICLK: {aiclk} MHz | Core Power: {power} W | Board Power: {bpower} W")
        print(f"  Temp: {temp} C | Current: {cur} A | Voltage: {volt} V")

    print("\n=== Processes Reported by tt-smi ===")
    procs = data.get("processes", [])
    if not procs:
        print("  No active device processes.")
    else:
        for p in procs:
            print(f"  PID {p.get('pid')} on Device {p.get('device')}: {p.get('cmdline')}")

if __name__ == "__main__":
    main()
