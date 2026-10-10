import re
import sys

log_prev = '/home/ttuser/LDPC_QB2_GA/Results/simulation_full_1.0_2.5dB_iter200.log'
log_curr = '/home/ttuser/LDPC_QB2_GA/Results/simulation_clip_L6.5_2.50dB_iter200.log'

def parse_log(filepath):
    pts = []
    with open(filepath, 'r') as f:
        for line in f:
            if '[2.50 dB]' in line:
                m = re.search(r'Blocks:\s+([0-9,]+)/.*?FE:\s+(\d+)/.*?FER:\s+([0-9.e+-]+).*?BER:\s+([0-9.e+-]+).*?\|\s+([0-9.]+)s', line)
                if m:
                    blks = int(m.group(1).replace(',', ''))
                    fe = int(m.group(2))
                    fer = float(m.group(3))
                    ber = float(m.group(4))
                    elap = float(m.group(5))
                    pts.append({
                        'blocks': blks,
                        'fe': fe,
                        'fer': fer,
                        'ber': ber,
                        'elapsed': elap,
                        'line': line.strip()
                    })
    return pts

prev_pts = parse_log(log_prev)
curr_pts = parse_log(log_curr)

print(f"Parsed {len(prev_pts)} points from previous unclipped log at 2.50 dB.")
print(f"Parsed {len(curr_pts)} points from current clipped L_max=6.5 log at 2.50 dB.")

if not curr_pts or not prev_pts:
    print("Error parsing logs.")
    sys.exit(1)

# Find first FE in current run
first_fe_curr = None
for p in curr_pts:
    if p['fe'] >= 1:
        first_fe_curr = p
        break

latest_curr = curr_pts[-1]

print("\n" + "="*80)
print("CURRENT SIMULATION (L_max = 6.5, Clipped)")
print("="*80)
if first_fe_curr:
    print(f"First Frame Error (FE=1) observed at:")
    print(f"  Blocks simulated : {first_fe_curr['blocks']:,}")
    print(f"  Elapsed time     : {first_fe_curr['elapsed']:.1f}s ({first_fe_curr['elapsed']/60:.1f} min)")
    print(f"  Current FER      : {first_fe_curr['fer']:.2e}")
    print(f"  Current BER      : {first_fe_curr['ber']:.2e}")
# Progression of FE in current clipped run
curr_fe_changes = []
c_fe_curr = -1
for p in curr_pts:
    if p['fe'] != c_fe_curr:
        c_fe_curr = p['fe']
        curr_fe_changes.append(p)

print("Frame Error Progression in current clipped run (L_max = 6.5) at 2.50 dB:")
prev_be_c = 0
for p in curr_fe_changes:
    total_be_c = int(round(p['ber'] * p['blocks'] * 2048))
    delta_be_c = total_be_c - prev_be_c
    prev_be_c = total_be_c
    print(f"  FE = {p['fe']:2d} | Blocks: {p['blocks']:12,d} | Elapsed: {p['elapsed']:7.1f}s ({p['elapsed']/60:5.1f} min) | Delta BE: {delta_be_c:4d} | FER: {p['fer']:.2e}")

print("\n" + "="*80)
print("PREVIOUS SIMULATION (Unclipped Baseline at 2.50 dB)")
print("="*80)

# Progression of FE in previous run
prev_fe_changes = []
c_fe = -1
for p in prev_pts:
    if p['fe'] != c_fe:
        c_fe = p['fe']
        prev_fe_changes.append(p)

print("Frame Error Progression in previous unclipped run at 2.50 dB:")
for p in prev_fe_changes:
    print(f"  FE = {p['fe']:2d} | Blocks: {p['blocks']:12,d} | Elapsed: {p['elapsed']:7.1f}s ({p['elapsed']/60:5.1f} min) | FER: {p['fer']:.2e} | BER: {p['ber']:.2e}")

# Exact comparisons
print("\n" + "="*80)
print("HEAD-TO-HEAD COMPARISON")
print("="*80)

# 1. At equivalent time of First Error
if first_fe_curr:
    t_target = first_fe_curr['elapsed']
    b_target = first_fe_curr['blocks']
    
    # By blocks
    closest_prev_by_blk = min(prev_pts, key=lambda x: abs(x['blocks'] - b_target))
    # By time
    closest_prev_by_time = min(prev_pts, key=lambda x: abs(x['elapsed'] - t_target))
    
    print(f"\n1. At First FE Event in Clipped Run (Blocks = {b_target:,}, Elapsed = {t_target:.1f}s / {t_target/60:.1f} min):")
    print(f"   - Current Clipped run: FE = {first_fe_curr['fe']}")
    print(f"   - Previous Unclipped (matched blocks {closest_prev_by_blk['blocks']:,}): FE = {closest_prev_by_blk['fe']} (FER = {closest_prev_by_blk['fer']:.2e})")
    print(f"   - Previous Unclipped (matched time {closest_prev_by_time['elapsed']:.1f}s):   FE = {closest_prev_by_time['fe']} (FER = {closest_prev_by_time['fer']:.2e})")

# 2. At Current/Latest time
t_now = latest_curr['elapsed']
b_now = latest_curr['blocks']
closest_prev_by_blk_now = min(prev_pts, key=lambda x: abs(x['blocks'] - b_now))
closest_prev_by_time_now = min(prev_pts, key=lambda x: abs(x['elapsed'] - t_now))

print(f"\n2. At Current/Latest Point in Clipped Run (Blocks = {b_now:,}, Elapsed = {t_now:.1f}s / {t_now/60:.1f} min):")
print(f"   - Current Clipped run: FE = {latest_curr['fe']} (FER = {latest_curr['fer']:.2e}, BER = {latest_curr['ber']:.2e})")
print(f"   - Previous Unclipped (matched blocks {closest_prev_by_blk_now['blocks']:,}): FE = {closest_prev_by_blk_now['fe']} (FER = {closest_prev_by_blk_now['fer']:.2e})")
print(f"   - Previous Unclipped (matched time {closest_prev_by_time_now['elapsed']:.1f}s):   FE = {closest_prev_by_time_now['fe']} (FER = {closest_prev_by_time_now['fer']:.2e})")

print("\n" + "="*80)
print("BIT ERRORS PER FRAME ERROR BREAKDOWN")
print("="*80)

# Bit errors for current run
curr_be = int(round(latest_curr['ber'] * latest_curr['blocks'] * 2048))
print(f"Current Clipped Run: {latest_curr['fe']} Frame Error -> Total Bit Errors = {curr_be} bits")

# Bit errors for previous run
print("\nPrevious Unclipped Run Bit Errors per Frame Error:")
prev_total_be = 0
for i, p in enumerate(prev_fe_changes[1:], 1):
    total_be = int(round(p['ber'] * p['blocks'] * 2048))
    delta_be = total_be - prev_total_be
    prev_total_be = total_be
    print(f"  Event FE #{p['fe']:2d} (at block {p['blocks']:12,d}): Delta Bit Errors = {delta_be:4d} | Cumulative BE = {total_be:5d}")


