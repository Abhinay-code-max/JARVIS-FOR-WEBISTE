"""
Auto-patcher: adds continuous conversation mode to JARVIS-XL.

How it works:
  - After JARVIS finishes speaking a response, a 8-second "active window" opens
  - During that window, wake word is NOT required
  - If the user speaks again within the window, it resets the timer
  - After 8 seconds of silence, wake word is required again
"""

MAIN_PY = r"C:\Users\Abhinay Kandrika\Desktop\JARVIS-XL\main.py"

with open(MAIN_PY, "r", encoding="utf-8") as f:
    content = f.read()

changed = False

# ── 1. Add conversation-window state to __init__ ────────────────────────
old_init = '''        self._current_user:    str | None = None
        self._user_confidence: float      = 0.0
        self._last_voice_buf:  np.ndarray | None = None'''

new_init = '''        self._current_user:    str | None = None
        self._user_confidence: float      = 0.0
        self._last_voice_buf:  np.ndarray | None = None

        # Continuous conversation mode
        self._conversation_active_until: float = 0.0
        self._conversation_window_sec:   float = 8.0'''

if "_conversation_active_until" not in content:
    if old_init in content:
        content = content.replace(old_init, new_init, 1)
        changed = True
        print("[1/4] Conversation state added to __init__.")
    else:
        print("[1/4] WARNING: Could not find __init__ block.")
else:
    print("[1/4] Already present — skipping.")

# ── 2. Open the conversation window after JARVIS speaks ─────────────────
old_speak = '''    def speak(self, text: str) -> None:
        if not text or not self._tts:
            return
        with self._speaking_lock:
            self._speaking = True
        self._tts_queue.put(text)'''

new_speak = '''    def speak(self, text: str) -> None:
        if not text or not self._tts:
            return
        with self._speaking_lock:
            self._speaking = True
        self._tts_queue.put(text)
        # Open continuous-conversation window once this finishes speaking
        import time as _time
        self._conversation_active_until = _time.time() + self._conversation_window_sec + (len(text) * 0.05)'''

if "_conversation_active_until = _time.time()" not in content:
    if old_speak in content:
        content = content.replace(old_speak, new_speak, 1)
        changed = True
        print("[2/4] Conversation window opener added to speak().")
    else:
        print("[2/4] WARNING: Could not find speak() method.")
else:
    print("[2/4] Already present — skipping.")

# ── 3. Add a helper method to check if conversation is active ───────────
old_marker = '''    def _on_text_command(self, text: str) -> None:
        self._text_queue.put(text)'''

new_marker = '''    def _on_text_command(self, text: str) -> None:
        self._text_queue.put(text)

    def _is_conversation_active(self) -> bool:
        """True if we're within the post-response window (no wake word needed)."""
        import time as _time
        return _time.time() < self._conversation_active_until

    def _extend_conversation_window(self) -> None:
        """Call after processing a follow-up to keep the window open a bit longer."""
        import time as _time
        self._conversation_active_until = _time.time() + self._conversation_window_sec'''

if "_is_conversation_active" not in content:
    if old_marker in content:
        content = content.replace(old_marker, new_marker, 1)
        changed = True
        print("[3/4] Helper methods added.")
    else:
        print("[3/4] WARNING: Could not find _on_text_command method.")
else:
    print("[3/4] Already present — skipping.")

# ── 4. Wire it into the Deepgram transcript callback ─────────────────────
old_callback = '''        def on_transcript(text: str):
            if not text or not text.strip():
                return
            if self.ui.muted:
                return
            print(f"[Deepgram] 🎤 {text}")
            self.ui.write_log(f"You: {text}")
            self._text_queue.put(text.strip())'''

new_callback = '''        def on_transcript(text: str):
            if not text or not text.strip():
                return
            if self.ui.muted:
                return
            print(f"[Deepgram] 🎤 {text}")

            # Continuous conversation mode: if we're in the active window,
            # skip wake-word requirement entirely.
            if self._is_conversation_active():
                self._extend_conversation_window()
                self._text_queue.put(text.strip())
                return

            # Otherwise require the wake word
            if self._wake.check(text):
                cleaned = self._wake.strip_wake_word(text)
                if cleaned:
                    self._extend_conversation_window()
                    self._text_queue.put(cleaned)
                else:
                    # Just the wake word alone — open the window and wait
                    self._extend_conversation_window()
                    self.ui.write_log("REC: Listening...")
            # else: ignore — not addressed to JARVIS'''

if "_is_conversation_active()" in content and "on_transcript" in content:
    if old_callback in content:
        content = content.replace(old_callback, new_callback, 1)
        changed = True
        print("[4/4] Wake-word bypass wired into Deepgram callback.")
    else:
        print("[4/4] WARNING: Could not find on_transcript callback — exact match failed.")
else:
    print("[4/4] Skipping — prerequisite changes not found.")

# ── Save ──────────────────────────────────────────────────────────────────
if changed:
    with open(MAIN_PY, "w", encoding="utf-8") as f:
        f.write(content)
    print("\n✅ Continuous conversation mode patched successfully!")
else:
    print("\n⚠️ No changes were made. Check warnings above.")
