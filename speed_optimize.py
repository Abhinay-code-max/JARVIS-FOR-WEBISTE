"""
Auto-patcher: reduces unnecessary time.sleep() delays across action files.
Cuts most delays by ~40-50% while keeping enough buffer for apps that
genuinely need loading time (WhatsApp, browsers).
"""
import re

ACTIONS_DIR = r"C:\Users\Abhinay Kandrika\Desktop\JARVIS-XL\actions"

# Map of (filename, old_value) -> new_value
# Only reducing the worst offenders, keeping critical ones intact
REDUCTIONS = {
    "open_app.py": {
        2.5: 1.2,    # generic app open wait
        1.5: 0.8,
        0.6: 0.3,
        0.5: 0.3,
    },
    "send_message.py": {
        4.0: 2.5,    # whatsapp load
        3.0: 1.5,
        2.5: 1.2,
        2.0: 1.0,
        1.5: 0.8,
        1.0: 0.6,
        0.8: 0.4,
        0.5: 0.3,
        0.3: 0.2,
    },
    "game_updater.py": {
        5.0: 3.0,
        4.0: 2.5,
        2.0: 1.2,
        1.5: 0.8,
    },
    "flight_finder.py": {
        5.0: 3.0,
    },
}

total_saved = 0.0

for fname, value_map in REDUCTIONS.items():
    fpath = f"{ACTIONS_DIR}\\{fname}"
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"SKIP: {fname} not found.")
        continue

    original = content
    file_saved = 0.0

    # Sort by descending old value so we don't accidentally double-replace
    for old_val, new_val in sorted(value_map.items(), key=lambda x: -x[0]):
        # Match time.sleep(X) or time.sleep(X.0) variations
        patterns = [
            f"time.sleep({old_val})",
            f"time.sleep({old_val:.1f})",
        ]
        for pat in patterns:
            count = content.count(pat)
            if count > 0:
                replacement = f"time.sleep({new_val})"
                content = content.replace(pat, replacement)
                saved = (old_val - new_val) * count
                file_saved += saved
                print(f"  {fname}: {pat} -> {replacement}  (x{count}, saves {saved:.1f}s)")

    if content != original:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
        total_saved += file_saved
        print(f"✅ {fname} patched — saved ~{file_saved:.1f}s per typical call\n")
    else:
        print(f"⚠️ {fname} — no matching patterns found\n")

print(f"\n{'='*50}")
print(f"TOTAL estimated time saved per typical interaction: ~{total_saved:.1f} seconds")
print(f"{'='*50}")
