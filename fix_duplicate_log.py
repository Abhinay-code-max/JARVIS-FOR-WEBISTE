"""
Auto-patcher: removes the duplicate "You:" log line in _listen_deepgram's
on_transcript callback, since _process_message already logs it.
"""

MAIN_PY = r"C:\Users\Abhinay Kandrika\Desktop\JARVIS-XL\main.py"

with open(MAIN_PY, "r", encoding="utf-8") as f:
    lines = f.readlines()

target_line_content = 'self.ui.write_log(f"You: {text}")'
removed = False

new_lines = []
for line in lines:
    if target_line_content in line and not removed:
        # Skip this line entirely (remove the duplicate log call)
        print(f"Removing duplicate log line: {line.strip()}")
        removed = True
        continue
    new_lines.append(line)

if removed:
    with open(MAIN_PY, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print("\n✅ Duplicate log line removed successfully!")
else:
    print("\n⚠️ Target line not found — no changes made.")
