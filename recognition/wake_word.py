"""
recognition/wake_word.py
========================
Wake-word detection for JARVIS-XL.

Default: keyword matching in Whisper transcriptions (zero-latency, no extra model).
Upgrade: set USE_OPENWAKEWORD=1 in environment to use the openwakeword library
         (always-on, sub-100ms latency, runs alongside VAD).

Supported wake phrases (case-insensitive):
  "hey jarvis", "ok jarvis", "jarvis", "hey j.a.r.v.i.s"
"""

from __future__ import annotations
import os
import re
import threading
from typing import Callable, Optional

_WAKE_PATTERNS = [
    r"\bhey\s+jarvis\b",
    r"\bok\s+jarvis\b",
    r"\bjarvis\b",
    r"\bhey\s+j\.?a\.?r\.?v\.?i\.?s\.?\b",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _WAKE_PATTERNS]


class WakeWordDetector:
    """
    Passive wake-word detector.

    In 'transcription' mode (default): call `check(text)` on each STT output.
    In 'streaming' mode: wraps openwakeword to run on raw audio frames.
    """

    def __init__(self, on_wake: Optional[Callable] = None):
        self._on_wake  = on_wake
        self._enabled  = True
        self._mode     = "transcription"
        self._oww      = None

        if os.environ.get("USE_OPENWAKEWORD", "0") == "1":
            self._init_openwakeword()

    def _init_openwakeword(self) -> None:
        try:
            import openwakeword
            from openwakeword.model import Model
            self._oww  = Model(wakeword_models=["hey_jarvis"])
            self._mode = "streaming"
            print("[WakeWord] openwakeword loaded (streaming mode).")
        except ImportError:
            print("[WakeWord] openwakeword not available — using transcription mode.")
        except Exception as e:
            print(f"[WakeWord] openwakeword error: {e}")

    # ── Transcription mode ────────────────────────────────────────────────────
    def check(self, text: str) -> bool:
        """
        Returns True if the text contains a wake phrase.
        Call this on every Whisper/Vosk transcription.
        """
        if not self._enabled:
            return True   # when disabled, treat everything as post-wake
        for pat in _COMPILED:
            if pat.search(text):
                return True
        return False

    def strip_wake_word(self, text: str) -> str:
        """Remove the wake phrase from the beginning of the transcription."""
        for pat in _COMPILED:
            text = pat.sub("", text).strip(" ,.")
        return text

    # ── Streaming mode (openwakeword) ─────────────────────────────────────────
    def process_audio_chunk(self, chunk) -> bool:
        """Feed raw int16 audio chunk; returns True if wake word detected."""
        if self._oww is None:
            return False
        try:
            import numpy as np
            prediction = self._oww.predict(np.array(chunk))
            for model_name, scores in prediction.items():
                if scores[-1] > 0.5:
                    if self._on_wake:
                        self._on_wake()
                    return True
        except Exception:
            pass
        return False

    # ── Control ───────────────────────────────────────────────────────────────
    def enable(self)  -> None: self._enabled = True
    def disable(self) -> None: self._enabled = False

    @property
    def mode(self) -> str:
        return self._mode
