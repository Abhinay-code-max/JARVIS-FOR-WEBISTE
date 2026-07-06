"""
Auto-patcher: prevents duplicate/echo commands from being processed twice
within a short time window (handles Deepgram occasionally splitting one
utterance into two final transcripts, or picking up TTS echo).
"""

MAIN_PY = r"C:\Users\Abhinay Kandrika\Desktop\JARVIS-XL\main.py"

with open(MAIN_PY, "r", encoding="utf-8") as f:
    content = f.read()

changed = False

# ── 1. Add dedup state to __init__ ───────────────────────────────────────
old_init = '''        # Continuous conversation mode
        self._conversation_active_until: float = 0.0
        self._conversation_window_sec:   float = 8.0'''

new_init = '''        # Continuous conversation mode
        self._conversation_active_until: float = 0.0
        self._conversation_window_sec:   float = 8.0

        # Duplicate-command guard
        self._last_command_text: str = ""
        self._last_command_time: float = 0.0
        self._dedup_window_sec:  float = 2.5'''

if "_dedup_window_sec" not in content:
    if old_init in content:
        content = content.replace(old_init, new_init, 1)
        changed = True
        print("[1/2] Dedup state added to __init__.")
    else:
        print("[1/2] WARNING: Could not find conversation-mode init block.")
else:
    print("[1/2] Already present — skipping.")

# ── 2. Add dedup check to _on_text_command ───────────────────────────────
old_handler = '''    def _on_text_command(self, text: str) -> None:
        self._text_queue.put(text)'''

new_handler = '''    def _on_text_command(self, text: str) -> None:
        import time as _time
        now = _time.time()
        normalized = text.strip().lower()

        # Skip if this is a near-duplicate of the last command within the dedup window
        if normalized and normalized == self._last_command_text:
            if now - self._last_command_time < self._dedup_window_sec:
                print(f"[Dedup] Ignored duplicate command: {text!r}")
                return

        self._last_command_text = normalized
        self._last_command_time = now
        self._text_queue.put(text)'''

if "[Dedup] Ignored duplicate" not in content:
    if old_handler in content:
        content = content.replace(old_handler, new_handler, 1)
        changed = True
        print("[2/2] Dedup check added to _on_text_command.")
    else:
        print("[2/2] WARNING: Could not find _on_text_command — trying alternate location.")
else:
    print("[2/2] Already present — skipping.")

# Also need to apply the same dedup to the Deepgram callback directly,
# since that's the actual entry point for voice (not _on_text_command,
# which is for typed commands)
old_transcript_active = '''            if self._is_conversation_active():
                self._extend_conversation_window()
                self._text_queue.put(text.strip())
                return'''

new_transcript_active = '''            if self._is_conversation_active():
                import time as _time
                now = _time.time()
                normalized = text.strip().lower()
                if normalized == self._last_command_text and (now - self._last_command_time) < self._dedup_window_sec:
                    print(f"[Dedup] Ignored duplicate voice command: {text!r}")
                    return
                self._last_command_text = normalized
                self._last_command_time = now
                self._extend_conversation_window()
                self._text_queue.put(text.strip())
                return'''

if "Ignored duplicate voice command" not in content:
    if old_transcript_active in content:
        content = content.replace(old_transcript_active, new_transcript_active, 1)
        changed = True
        print("[3/3] Dedup check added to Deepgram active-conversation path.")
    else:
        print("[3/3] WARNING: Could not find conversation-active transcript block.")
else:
    print("[3/3] Already present — skipping.")

if changed:
    with open(MAIN_PY, "w", encoding="utf-8") as f:
        f.write(content)
    print("\n✅ Duplicate command guard patched successfully!")
else:
    print("\n⚠️ No changes made.")
