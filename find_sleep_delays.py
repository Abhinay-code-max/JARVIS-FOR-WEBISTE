"""
Finds all time.sleep() calls across action files to identify
unnecessary delays we can reduce.
"""
import os
import re

ACTIONS_DIR = r"C:\Users\Abhinay Kandrika\Desktop\JARVIS-XL\actions"

total_sleep_time = 0
results = []

for fname in os.listdir(ACTIONS_DIR):
    if not fname.endswith(".py"):
        continue
    fpath = os.path.join(ACTIONS_DIR, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        m = re.search(r'time\.sleep\(([\d.]+)\)', line)
        if m:
            delay = float(m.group(1))
            total_sleep_time += delay
            results.append((fname, i + 1, delay, line.strip()))

results.sort(key=lambda x: -x[2])

print(f"Found {len(results)} time.sleep() calls across action files.")
print(f"Total cumulative sleep time if all hit: {total_sleep_time:.1f} seconds\n")
print("Top 20 longest delays:")
print("-" * 80)
for fname, line_no, delay, code in results[:20]:
    print(f"{fname:25s} line {line_no:4d}  {delay:5.1f}s   {code}")
