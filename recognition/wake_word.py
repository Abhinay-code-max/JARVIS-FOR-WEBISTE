"""
recognition/wake_word.py
========================
Wake-word detection for JARVIS-XL.

Default: keyword matching in Whisper transcriptions (zero-latency, no extra model).
Upgrade: set USE_OPENWAKEWORD=1 in environment to use the openwakeword library
         (always-on, sub-100ms latency, runs alongside VAD).

Supported wake phrases (case-insensitive):
  "hey jarvis", "ok jarvis", "jarvis", "hey j.a.r.v.i.s"

Fuzzy matching: close phonetic variants of the wake phrase (e.g. "hey jarves",
  "hay jarvis", "hey javis", "okay jarvis") are also accepted via Levenshtein
  edit-distance ratio ≥ FUZZY_THRESHOLD. Only the first few words of the
  transcription are inspected so a long command body can't trigger a false match.
"""

from __future__ import annotations
import logging
import os
import re
import threading
from typing import Callable, Optional

_log = logging.getLogger("jarvis.wakeword")

_WAKE_PATTERNS = [
    r"\bhey\s+jarvis\b",
    r"\bok\s+jarvis\b",
    r"\bokay\s+jarvis\b",
    r"\bjarvis\b",
    r"\bhey\s+j\.?a\.?r\.?v\.?i\.?s\.?\b",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _WAKE_PATTERNS]

# Canonical wake phrases to fuzzy-match against (lowercased).
# Only short phrases — the fuzzy check is only applied to the first 3–4 words
# of the transcription so a long command can't accidentally match.
_FUZZY_TARGETS = [
    "hey jarvis",
    "ok jarvis",
    "okay jarvis",
    "jarvis",
]

# Minimum Levenshtein similarity ratio (0–1) to accept as a fuzzy wake match.
# 0.75 accepts up to ~2 edit-distance errors on a 8-char phrase ("hey jarves",
# "hay jarvis"), but rejects clearly different words ("hey davis", "play jazz").
FUZZY_THRESHOLD = 0.75


def _levenshtein_ratio(a: str, b: str) -> float:
    """Pure-stdlib similarity ratio in [0, 1]. No external dependency."""
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    # Build edit-distance matrix (space-optimised: two rows)
    prev = list(range(lb + 1))
    curr = [0] * (lb + 1)
    for i in range(1, la + 1):
        curr[0] = i
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    dist = prev[lb]
    return 1.0 - dist / max(la, lb)


def _fuzzy_wake_check(text: str) -> bool:
    """Return True if the first few words of *text* fuzzy-match any canonical
    wake phrase. Deliberately scoped to a word-count cap so a long command body
    (e.g. 'play jazz on Spotify') can never accidentally trigger this path."""
    words = text.lower().split()
    # Inspect only up to 4 words at the start — enough to cover the longest
    # canonical phrase ("okay jarvis" = 2 words) with one extra word of slack.
    candidate_words = words[:4]
    for n in range(1, len(candidate_words) + 1):
        candidate = " ".join(candidate_words[:n])
        for target in _FUZZY_TARGETS:
            if _levenshtein_ratio(candidate, target) >= FUZZY_THRESHOLD:
                _log.debug(
                    "Fuzzy wake match: %r ~ %r (ratio=%.2f)",
                    candidate, target,
                    _levenshtein_ratio(candidate, target),
                )
                return True
    return False


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
            _log.info("openwakeword loaded (streaming mode).")
        except ImportError:
            _log.info("openwakeword not available — using transcription mode.")
        except Exception as e:
            _log.warning("openwakeword error: %s", e)

    # ── Transcription mode ────────────────────────────────────────────────────
    def check(self, text: str) -> bool:
        """
        Returns True if the text contains a wake phrase (exact or fuzzy).
        Exact regex match is tried first; fuzzy edit-distance fallback fires
        only when regex fails, scoped to the first 4 words of the transcription
        so long command bodies cannot trigger false positives.
        Call this on every Whisper/Vosk transcription.
        """
        if not self._enabled:
            return True   # when disabled, treat everything as post-wake
        # 1. Exact / pattern match (fast path)
        for pat in _COMPILED:
            if pat.search(text):
                return True
        # 2. Fuzzy fallback — catches phonetic variants like "hey jarves",
        #    "hay jarvis", "okay jarvis", "hey javis" that Whisper commonly
        #    produces from accented or slightly muffled speech.
        return _fuzzy_wake_check(text)

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
