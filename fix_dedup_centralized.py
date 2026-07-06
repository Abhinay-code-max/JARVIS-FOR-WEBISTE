"""
Auto-patcher: moves duplicate detection to the central point where
ALL commands are consumed (_text_command_loop), regardless of which
of the 4 entry points pushed them into the queue.
"""

MAIN_PY = r"C:\Users\Abhinay Kandrika\Desktop\JARVIS-XL\main.py"

with open(MAIN_PY, "r", encoding="utf-8") as f:
    content = f.read()

changed = False

old_loop = '''    def _text_command_loop(self) -> None:
        while True:
            try:
                text = self._text_queue.get(timeout=0.5)
                if text.strip():
                    self._process_message(text)
            except queue.Empty:
                pass'''

new_loop = '''    def _text_command_loop(self) -> None:
        while True:
            try:
                text = self._text_queue.get(timeout=0.5)
                if not text.strip():
                    continue

                import time as _time
                now = _time.time()
                normalized = text.strip().lower()

                if (normalized == self._last_command_text
                        and (now - self._last_command_time) < self._dedup_window_sec):
                    print(f"[Dedup] Skipped duplicate at consumer: {text!r}")
                    continue

                self._last_command_text = normalized
                self._last_command_time = now
                self._process_message(text)
            except queue.Empty:
                pass'''

if "Skipped duplicate at consumer" not in content:
    if old_loop in content:
        content = content.replace(old_loop, new_loop, 1)
        changed = True
        print("✅ Centralized dedup check added to _text_command_loop.")
    else:
        print("❌ WARNING: Could not find _text_command_loop — exact match failed.")
else:
    print("Already present — skipping.")

if changed:
    with open(MAIN_PY, "w", encoding="utf-8") as f:
        f.write(content)
    print("\n✅ main.py patched successfully!")
else:
    print("\n⚠️ No changes made.")
