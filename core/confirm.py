# core/confirm.py
"""
Shared confirmation gate for the "run whatever the LLM wrote" code-
execution paths (desktop.py's generated-code exec, code_helper.py's
run/build, dev_agent.py's run/install). None of these asked the user
anything before executing — this makes that a real, blocking yes/no.

Voice/text input all funnels through main.py's JarvisXL._enqueue_command
(the single chokepoint for all 4 producers — typed text, whisper, vosk,
deepgram). When a confirmation is pending, that same chokepoint routes
the next utterance here instead of treating it as a new command.
"""
import threading

_YES = {"yes", "y", "yeah", "yep", "yup", "confirm", "run it", "do it", "go ahead", "sure", "ok", "okay"}
_NO  = {"no", "n", "nope", "nah", "cancel", "stop", "don't", "dont", "abort"}


class _ConfirmationGate:
    def __init__(self):
        self._lock    = threading.Lock()
        self._pending = False
        self._event   = threading.Event()
        self._answer  = False

    def is_pending(self) -> bool:
        with self._lock:
            return self._pending

    def answer(self, text: str) -> bool:
        """Called from the input chokepoint when a confirmation is pending.
        Returns True once consumed as an answer (yes -> proceed, no/anything
        unclear -> treated as a decline, so an unrelated reply can't leave
        the gate hanging)."""
        normalized = text.strip().lower()
        with self._lock:
            if not self._pending:
                return False
            self._answer = normalized in _YES
            self._pending = False
        self._event.set()
        return True

    def request(self, player, prompt: str, speak=None, timeout: float = 45.0) -> bool:
        """Blocking. Shows `prompt`, waits for a yes/no answer (or the
        timeout), returns True only on an explicit yes. No live player to
        ask -> denied by default, never silently allowed."""
        if player is None:
            return False

        with self._lock:
            self._pending = True
            self._event.clear()
            self._answer = False

        try:
            player.write_log(f"CONFIRM: {prompt} (say yes to proceed, no to cancel)")
            if speak:
                speak(f"{prompt} Should I proceed?")

            got_answer = self._event.wait(timeout=timeout)
            with self._lock:
                answer, still_pending = self._answer, self._pending

            if not got_answer or still_pending:
                with self._lock:
                    self._pending = False
                player.write_log("CONFIRM: no response — treating as cancelled.")
                return False

            player.write_log(f"CONFIRM: {'proceeding' if answer else 'cancelled'}.")
            return answer
        finally:
            with self._lock:
                self._pending = False


CONFIRM = _ConfirmationGate()
