#!/usr/bin/env python3
"""
Set or release forced AICLK frequency on all Tenstorrent Blackhole processors.

Usage:
  python3 tools/set_blackhole_clk.py 1400   # Force all Blackhole ASICs to 1400 MHz (firmware max)
  python3 tools/set_blackhole_clk.py 0      # Release override and return to normal DVFS (800 MHz idle / 1350 MHz busy)
  python3 tools/set_blackhole_clk.py --status # Print current AICLK and voltage telemetry without modifying state
"""

import sys
import time
import tt_umd

FORCE_AICLK_MSG = 0x33  # tt::umd::blackhole::ArcMessageType::FORCE_AICLK


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip())
        sys.exit(0)

    status_only = sys.argv[1] in ("-s", "--status", "status")
    target_mhz = 0 if status_only else int(sys.argv[1])

    if not status_only and target_mhz != 0 and (target_mhz < 200 or target_mhz > 1400):
        print(f"[ERROR] Requested frequency {target_mhz} MHz is outside Blackhole ARC firmware bounds [200..1400 MHz] (or 0 to reset).")
        sys.exit(1)

    devices = tt_umd.PCIDevice.enumerate_devices()
    if not devices:
        print("[ERROR] No Tenstorrent PCIe devices found.")
        sys.exit(1)

    for dev_id in sorted(devices):
        dev = tt_umd.TTDevice.create(dev_id)
        dev.init_tt_device()
        fip = dev.get_firmware_info_provider()

        if status_only:
            aiclk = fip.get_aiclk()
            print(f"[CLK] Chip {dev_id}: AICLK = {aiclk} MHz")
            continue

        rc, _, _ = dev.arc_msg(FORCE_AICLK_MSG, True, [target_mhz], 1000)
        time.sleep(0.15)
        aiclk = fip.get_aiclk()
        mode_str = f"FORCED {target_mhz} MHz" if target_mhz > 0 else "AUTO DVFS (Released)"
        if rc == 0:
            print(f"[CLK] Chip {dev_id}: {mode_str} -> OK (Observed AICLK = {aiclk} MHz)")
        else:
            print(f"[CLK] Chip {dev_id}: {mode_str} -> FAILED (ARC exit_code = {rc}, Observed AICLK = {aiclk} MHz)")


if __name__ == "__main__":
    main()
