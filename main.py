"""
JARVIS-XL — Enhanced Local AI Assistant
========================================
Improvements over MARK XL:
  • Face recognition  — camera identifies YOU on startup; greets by name
  • Voice recognition — speaker diarisation so JARVIS knows who is talking
  • Smarter system prompt that injects user identity automatically
  • Wake-word detection ("Hey JARVIS") so mic is always listening
  • Emotion/tone detection from voice (calm, urgent, stressed)
  • Conversation context window extended to 20 turns
  • Auto-summarisation when context limit is reached
  • All original 18 tools retained + new `identify_user` tool
"""

# ── Force UTF-8 console output (Windows defaults stdout/stderr to cp1252,
#    which crashes on the emoji used throughout our log messages) ────────────
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(_sys.stderr, "reconfigure"):
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Structured logging — configured before anything else so every module
#    (including the two bootstrap prints just below, which are a
#    deliberate exception — see the comment at _bootstrap()) can use it
#    from the moment it's imported. Stdlib-only; safe before _bootstrap()
#    installs anything. ────────────────────────────────────────────────────
from core.logging_setup import init_logging
init_logging()

# ── Silence verbose logs ────────────────────────────────────────────────────
import os as _os
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",  "3")
_os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
_os.environ.setdefault("GRPC_VERBOSITY",         "ERROR")
_os.environ.setdefault("USE_TF",                 "0")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_os.environ.setdefault("HF_HUB_OFFLINE",         "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE",   "1")
_os.environ.setdefault("HF_DATASETS_OFFLINE",    "1")

import warnings as _warnings
_warnings.filterwarnings("ignore")

# ── Bootstrap ───────────────────────────────────────────────────────────────
import importlib.util as _ilu
import subprocess     as _sp
import sys            as _sys

_BASE_PKGS = [
    ("PyQt6",       "PyQt6"),
    ("psutil",      "psutil"),
    ("numpy",       "numpy"),
    ("sounddevice", "sounddevice"),
    ("PIL",         "pillow"),
    ("requests",    "requests"),
    ("cv2",         "opencv-python"),
]

def _bootstrap() -> None:
    # Deliberately left as print(), not logging: os.execv() below replaces
    # the process image within milliseconds of the second message — there's
    # no guarantee the QueueListener's background thread gets scheduled to
    # drain the queue (console or DB) before that happens, so a logger call
    # here risks the message silently never appearing at all. The only
    # audience for this message is a developer watching the terminal during
    # first-run setup; print() guarantees they actually see it before the
    # restart.
    need = [pkg for mod, pkg in _BASE_PKGS if _ilu.find_spec(mod) is None]
    if not need:
        return
    print(f"\n[JARVIS-XL] First-run setup — installing: {', '.join(need)}")
    _sp.run([_sys.executable, "-m", "pip", "install", *need], check=True)
    print("\n[JARVIS-XL] Base packages ready — restarting…\n")
    _os.execv(_sys.executable, [_sys.executable] + _sys.argv)

_bootstrap()

# ── Standard imports ─────────────────────────────────────────────────────────
import json
import logging
import queue
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd

from ui import JarvisUI
from memory.memory_manager import load_memory, update_memory, format_memory_for_prompt
from core.llm_client import call_llm, call_llm_stream, get_llm_settings
from recognition.face_id import FaceIdentifier
from recognition.voice_id import VoiceIdentifier
from recognition.wake_word import WakeWordDetector

from core.tool_dispatch import TOOL_DISPATCH
from core.tool_gate import dispatch_tool
from core.policy import DESKTOP as _DESKTOP_CALLER, is_agent_task_allowed
from core.confirm import CONFIRM
from core.task_approval import TASK_APPROVAL
from core.proactive import ProactiveLoop
from config import load_config, BASE_DIR

_log      = logging.getLogger("jarvis.main")
_wake_log = logging.getLogger("jarvis.wakeword")

# ── Paths ────────────────────────────────────────────────────────────────────
PROMPT_PATH     = BASE_DIR / "core" / "prompt.txt"
FACES_DIR       = BASE_DIR / "recognition" / "faces"
VOICES_DIR      = BASE_DIR / "recognition" / "voices"

SAMPLE_RATE_IN  = 16_000
BLOCK_SIZE      = 1_024
CHANNELS        = 1


# ── Tool declarations ────────────────────────────────────────────────────────
# Moved to core/tool_declarations.py so it's a shared source of truth —
# core/tool_contracts.py derives input_schema from it, and
# agent/planner.py's PLANNER_PROMPT is generated from it too, instead of
# each maintaining its own independently-drifting copy.
from core.tool_declarations import TOOL_DECLARATIONS, OLLAMA_TOOLS


# ── Helpers ──────────────────────────────────────────────────────────────────
def _load_system_prompt() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except Exception:
        return (
            "You are JARVIS, an elite AI assistant. "
            "You know the user personally — always address them by name when you know it. "
            "Be concise, direct, warm but efficient. "
            "Use provided tools immediately — never guess or simulate results."
        )


# ── Voice Activity Detection ──────────────────────────────────────────────────
class _VADBuffer:
    def __init__(self, sample_rate=16_000, silence_sec=5.0,
                 barge_in_speech_sec=0.18,
                 speech_thresh=0.008, silence_thresh=0.004,
                 min_speech_sec=0.3, max_speech_sec=30.0):
        self._sr           = sample_rate
        self._sil_n        = int(silence_sec * sample_rate)
        # Shorter threshold used ONLY to confirm a barge-in (sustained
        # speech-level RMS while JARVIS is talking) — kept separate from
        # silence_sec so lengthening end-of-utterance patience doesn't also
        # make barge-in feel sluggish. Not used by process() below.
        self._barge_in_speech_sec = barge_in_speech_sec
        self._speech_thresh = speech_thresh
        self._sil_thresh   = silence_thresh
        self._min_n        = int(min_speech_sec * sample_rate)
        self._max_n        = int(max_speech_sec * sample_rate)
        self._buf:         list = []
        self._in_spch      = False
        self._sil_cnt      = 0
        self._nudge_shown  = False

    def process(self, chunk: np.ndarray):
        rms     = float(np.sqrt(np.mean(chunk ** 2)))
        total_n = sum(len(c) for c in self._buf)
        if rms > self._speech_thresh:
            self._in_spch     = True
            self._sil_cnt     = 0
            self._nudge_shown = False
            self._buf.append(chunk.copy())
        elif self._in_spch:
            self._buf.append(chunk.copy())
            if rms < self._sil_thresh:
                self._sil_cnt += len(chunk)
            if self._sil_cnt >= self._sil_n or total_n >= self._max_n:
                audio         = np.concatenate(self._buf)
                self._buf     = []
                self._in_spch = False
                self._sil_cnt = 0
                self._nudge_shown = False
                if len(audio) >= self._min_n:
                    return audio
        return None

    def should_nudge(self, nudge_sec: float = 2.0) -> bool:
        """Fires once per utterance, ~nudge_sec into a mid-utterance pause —
        lets the caller show a 'still listening' UI cue well before the
        full silence_sec timeout ends the utterance."""
        if not self._in_spch or self._nudge_shown:
            return False
        if self._sil_cnt >= int(nudge_sec * self._sr):
            self._nudge_shown = True
            return True
        return False


class _BargeInDetector:
    """Watches raw mic RMS while JARVIS is speaking and fires once
    ~confirm_sec of continuous speech-level RMS is seen — long enough to
    tell a real interruption apart from a single loud click, but short
    enough that barge-in still feels immediate. Buffers the hot audio so
    it can be handed to a fresh _VADBuffer instead of being discarded."""

    def __init__(self, speech_thresh: float = 0.008, confirm_sec: float = 0.18,
                 sample_rate: int = 16_000):
        self._speech_thresh = speech_thresh
        self._confirm_n     = int(confirm_sec * sample_rate)
        self._hot_n         = 0
        self._hot_chunks: list = []

    def reset(self) -> None:
        self._hot_n      = 0
        self._hot_chunks = []

    def check(self, chunk: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms <= self._speech_thresh:
            self.reset()
            return False
        self._hot_n += len(chunk)
        self._hot_chunks.append(chunk.copy())
        return self._hot_n >= self._confirm_n

    def drain(self) -> np.ndarray:
        audio = np.concatenate(self._hot_chunks) if self._hot_chunks else np.empty(0, dtype=np.float32)
        self.reset()
        return audio


# ── Main assistant ────────────────────────────────────────────────────────────
class JarvisXL:
    """
    JARVIS-XL — Enhanced local assistant with face + voice recognition.

    On startup:
      1. Camera scans for a known face → greet by name immediately
      2. Voice samples are cross-checked against enrolled voice prints
      3. Unknown user → ask their name, register them for next time
    """

    MAX_HISTORY     = 20   # turns (up from 10 in MARK XL)
    MAX_TOOL_ROUNDS = 6
    WAKE_WINDOW_SEC = 15.0  # how long JARVIS stays "awake" after the wake word / last command, before requiring it again

    def __init__(self, ui: JarvisUI):
        self.ui                = ui
        self._config           = load_config()
        self._stt              = None
        self._tts              = None
        self._tts_ready        = threading.Event()
        self._speaking         = False
        self._speaking_lock    = threading.Lock()
        # Echo-gating: suppress mic input while JARVIS is speaking, plus a
        # short grace period after TTS finishes, so speaker->mic bleed
        # (very common without a headset — no acoustic echo cancellation
        # in this pipeline) doesn't get transcribed and re-processed as a
        # phantom user command.
        self._speech_end_time: float = 0.0
        self._echo_grace_sec:  float = 0.6
        # Set by _handle_barge_in() and consumed by _tts_worker's finally
        # block — without this, the worker's normal "TTS just ended, arm
        # the grace period" cleanup would immediately re-gate the mic right
        # after a barge-in, undoing the whole point of interrupting.
        self._interrupted_by_barge_in: bool = False
        self._text_queue:      queue.Queue = queue.Queue()
        self._tts_queue:       queue.Queue = queue.Queue()
        self._conversation:    list[dict]  = []

        # Recognition subsystems
        self._face_id:  FaceIdentifier  = FaceIdentifier(FACES_DIR)
        self._voice_id: VoiceIdentifier = VoiceIdentifier(VOICES_DIR)
        self._wake:       WakeWordDetector = WakeWordDetector(on_wake=self._on_wake_detected)
        # epoch time until which the wake-word gate is open; 0.0 = asleep,
        # requires the wake word before the next mic-derived command is
        # accepted (see _wake_gate). Typed text bypasses this entirely.
        self._wake_until: float = 0.0
        # Set by _handle_barge_in(); consumed (one-shot) by the next
        # _wake_gate() call so an interrupt-driven utterance always bypasses
        # the wake-word check. Deliberately separate from
        # _interrupted_by_barge_in below — that flag is consumed almost
        # immediately by _tts_worker's finally block, often well before the
        # interrupted utterance finishes transcribing and reaches
        # _wake_gate, so it can't double as this signal.
        self._barge_in_pending: bool = False

        # Current-session user state
        self._current_user:   str | None = None   # name once identified
        self._user_confidence: float     = 0.0
        self._last_voice_buf: np.ndarray | None = None  # most recent utterance audio

        # Duplicate-command guard — split producer/consumer state.
        # Producer-side (_last_produced_*) is written by whichever of the 4
        # entry points (typed text, whisper, vosk, deepgram) is calling in,
        # each on its own thread, so it's protected by _produce_lock.
        # Consumer-side (_last_command_*) is only ever touched by the single
        # _text_command_loop thread, so it needs no lock. Keeping these two
        # separate (instead of one shared pair checked from both sides)
        # avoids the producer and consumer racing over the same fields.
        self._produce_lock:       threading.Lock = threading.Lock()
        self._last_produced_text: str   = ""
        self._last_produced_time: float = 0.0
        self._last_command_text:  str   = ""
        self._last_command_time:  float = 0.0
        self._dedup_window_sec:   float = 2.5

        self._proactive = ProactiveLoop(
            speak       = self.speak,
            is_speaking = lambda: self._speaking,
            is_muted    = lambda: self.ui.muted,
        )

        self.ui.on_text_command = self._on_text_command

    # ── System prompt ─────────────────────────────────────────────────────────
    def _build_system_prompt(self) -> str:
        sys_p   = _load_system_prompt()
        memory  = load_memory()
        mem_str = format_memory_for_prompt(memory)
        now     = datetime.now()

        # Inject current user identity into every prompt
        user_ctx = ""
        if self._current_user:
            user_ctx = (
                f"[ACTIVE USER]\n"
                f"The person currently talking to you is: {self._current_user} "
                f"(face/voice confidence: {self._user_confidence:.0%}). "
                f"Always address them by name unless the conversation is already ongoing.\n"
            )

        time_ctx = (
            f"[CURRENT DATE & TIME]\n"
            f"{now.strftime('%A, %B %d, %Y — %I:%M %p')}\n"
        )

        # Persistent reminder while a file is loaded — the one-time
        # [FILE_UPLOADED] chat message only appears once at drop time, so
        # without this a later "what is this?" a few turns on has no signal
        # telling the model a file (not the live screen/webcam) is the subject.
        file_ctx = ""
        cur_file = self.ui.current_file
        if cur_file:
            file_ctx = (
                f"[UPLOADED FILE]\n"
                f"The user has a file loaded and ready: '{Path(cur_file).name}'. "
                f"If they ask about \"this\", \"the file\", what they uploaded, or its "
                f"content, call file_processor — do NOT call screen_process, which "
                f"captures a brand-new live screen/webcam image instead of the "
                f"uploaded file.\n"
            )

        parts = [sys_p]
        if user_ctx:  parts.append(user_ctx)
        if mem_str:   parts.append(mem_str)
        if file_ctx:  parts.append(file_ctx)
        parts.append(time_ctx)
        return "\n\n".join(parts)

    # ── Speaking ──────────────────────────────────────────────────────────────
    def _tts_worker(self) -> None:
        self._tts_ready.wait(timeout=120)
        while True:
            text = self._tts_queue.get()
            try:
                if text and self._tts:
                    with self._speaking_lock:
                        self._speaking = True
                    self.ui.set_state("SPEAKING")
                    self._tts.speak(text)
            except Exception as e:
                _log.error("TTS playback error: %s", e)
            finally:
                self._tts_queue.task_done()
                if self._tts_queue.empty():
                    with self._speaking_lock:
                        was_barge_in = self._interrupted_by_barge_in
                        self._interrupted_by_barge_in = False
                        self._speaking = False
                        if not was_barge_in:
                            self._speech_end_time = time.time() + self._echo_grace_sec
                        # else: a barge-in already zeroed _speech_end_time —
                        # don't re-arm the grace window right on top of it,
                        # that would re-gate the mic immediately after the
                        # interruption we just made room for.
                    if not self.ui.muted:
                        self.ui.set_state("LISTENING")

    def _echo_gated(self) -> bool:
        """True while JARVIS is speaking, or within the grace window right
        after — listen loops should drop audio during this time."""
        if self._speaking:
            return True
        return time.time() < self._speech_end_time

    def speak(self, text: str) -> None:
        if not text or not self._tts:
            return
        with self._speaking_lock:
            self._speaking = True
        self._tts_queue.put(text)

    def speak_error(self, tool_name: str, error) -> None:
        self.ui.write_log(f"ERR: {tool_name} — {str(error)[:120]}")
        self.speak(f"{tool_name} encountered an error.")

    # ── Barge-in ───────────────────────────────────────────────────────────────
    def _handle_barge_in(self) -> None:
        """Called the moment sustained speech-level audio is detected while
        JARVIS is talking. Cuts playback short and flips state so the words
        that triggered the interrupt are captured as the start of a new
        utterance instead of being dropped by the echo gate."""
        if self._tts:
            try:
                self._tts.stop()
            except Exception as e:
                _log.warning("Barge-in: tts.stop() failed: %s", e)

        # Drop any sentences still queued from the interrupted response —
        # otherwise the TTS worker would just move on to the next one right
        # after stop() cuts the current chunk off.
        while True:
            try:
                self._tts_queue.get_nowait()
                self._tts_queue.task_done()
            except queue.Empty:
                break

        with self._speaking_lock:
            self._speaking                 = False
            self._speech_end_time          = 0.0   # don't leave the tail grace window active
            self._interrupted_by_barge_in  = True
            self._barge_in_pending         = True   # consumed by the next _wake_gate() call — see __init__

        if not self.ui.muted:
            self.ui.set_state("LISTENING")

    # ── Wake word ──────────────────────────────────────────────────────────────
    def _on_wake_detected(self) -> None:
        """Callback for WakeWordDetector's streaming/openwakeword mode
        (env var USE_OPENWAKEWORD=1) — opens the awake window. Note:
        streaming mode also requires raw audio chunks to be fed to
        self._wake.process_audio_chunk(), which nothing in this codebase
        currently does (that would mean a second concurrent mic stream
        alongside whichever STT engine is active). This callback is wired
        for when that's added later; it's inert until then."""
        self._wake_until = time.time() + self.WAKE_WINDOW_SEC
        _wake_log.info("Wake word detected (streaming mode).")

    def _wake_gate(self, text: str) -> str | None:
        """Wake-word gate for mic-derived transcriptions (default
        'transcription' mode — check() is run against text STT already
        produced, no extra mic stream involved). While within an open
        awake window, utterances pass straight through and slide the
        window forward, so an ongoing back-and-forth doesn't need the
        wake word every turn. Otherwise only an utterance containing the
        wake phrase is let through, with the phrase stripped, which also
        opens a new window. Set 'wake_word_enabled': false in config to
        disable and restore the old always-on behaviour.
        Returns the (possibly stripped) text to enqueue, or None to drop
        the utterance silently."""
        if not self._config.get("wake_word_enabled", True):
            return text
        if self._barge_in_pending:
            # The user interrupted JARVIS's own speech — that's already
            # unambiguous proof of intent to engage, so this utterance
            # bypasses the wake-word check outright (but still opens/
            # refreshes the awake window like a normal wake-word hit would).
            self._barge_in_pending = False
            self._wake_until = time.time() + self.WAKE_WINDOW_SEC
            _wake_log.debug("Gate: barge-in bypass, window extended")
            return text
        now = time.time()
        if now < self._wake_until:
            self._wake_until = now + self.WAKE_WINDOW_SEC
            _wake_log.debug("Gate: within awake window, pass-through")
            return text
        if not self._wake.check(text):
            _wake_log.debug("Gate: no wake word, dropped: %r", text)
            return None
        self._wake_until = now + self.WAKE_WINDOW_SEC
        _wake_log.info("Gate: wake word matched, window opened")
        stripped = self._wake.strip_wake_word(text)
        return stripped if stripped.strip() else None

    # ── Text command entry ────────────────────────────────────────────────────
    def _enqueue_command(self, text: str) -> None:
        """Single entry point for all 4 producers (typed text, whisper, vosk,
        deepgram). Dedupes at the point of production so a near-duplicate
        from the same source (e.g. Deepgram splitting one utterance into two
        final transcripts) never reaches the queue at all."""
        normalized = text.strip().lower()
        if not normalized:
            return

        # A code-execution confirmation gate (core/confirm.py) is waiting
        # on the next utterance as its yes/no answer — route it there
        # instead of treating it as a new independent command.
        if CONFIRM.is_pending():
            CONFIRM.answer(text)
            return

        with self._produce_lock:
            now = time.time()
            if (normalized == self._last_produced_text
                    and (now - self._last_produced_time) < self._dedup_window_sec):
                _log.debug("Dedup: ignored duplicate at producer: %r", text)
                return
            self._last_produced_text = normalized
            self._last_produced_time = now
        self._text_queue.put(text)

    def _on_text_command(self, text: str) -> None:
        self._enqueue_command(text)

    # ── Recognition tools ─────────────────────────────────────────────────────
    def _do_identify_user(self, method: str = "both") -> str:
        """Run face/voice recognition and update self._current_user."""
        results = []
        confidence = 0.0

        if method in ("face", "both"):
            name, conf = self._face_id.identify()
            if name:
                results.append((name, conf, "face"))
                confidence = max(confidence, conf)

        if method in ("voice", "both") and self._last_voice_buf is not None:
            name, conf = self._voice_id.identify(self._last_voice_buf)
            if name:
                results.append((name, conf, "voice"))
                confidence = max(confidence, conf)

        if not results:
            return "unknown"

        # Pick highest-confidence result
        best = max(results, key=lambda x: x[1])
        name, conf, source = best
        self._current_user    = name
        self._user_confidence = conf
        self.ui.write_log(f"REC: Identified '{name}' via {source} ({conf:.0%})")
        return name

    def _do_register_user(self, name: str, method: str = "both") -> str:
        """Capture and enroll a new user's face/voice."""
        msgs = []
        if method in ("face", "both"):
            ok = self._face_id.register(name)
            msgs.append(f"face {'registered' if ok else 'failed'}")
        if method in ("voice", "both") and self._last_voice_buf is not None:
            ok = self._voice_id.register(name, self._last_voice_buf)
            msgs.append(f"voice {'registered' if ok else 'needs more audio'}")
        if msgs:
            self._current_user = name
            update_memory({"identity": {"name": {"value": name}}})
        return f"Registration complete for {name}: {', '.join(msgs)}."

    # ── Tool execution ────────────────────────────────────────────────────────
    def _execute_tool(self, name: str, args: dict, caller_class: str = _DESKTOP_CALLER) -> str:
        """caller_class defaults to 'desktop' so the one existing call
        site (this method's own text/voice tool-calling loop) behaves
        exactly as before — see core/policy.py's is_agent_task_allowed()
        for why agent_task specifically needs its own check here rather
        than through the normal dispatch_tool()/permission_policy path."""
        _log.info("Tool call: %s %r", name, args)
        self.ui.set_state("THINKING")

        if name == "save_memory":
            category = args.get("category", "notes")
            key      = args.get("key", "")
            value    = args.get("value", "")
            if key and value:
                update_memory({category: {key: {"value": value}}})
                logging.getLogger("jarvis.memory").info("%s/%s = %s", category, key, value)
                # Auto-update current user if name is saved
                if category == "identity" and key == "name":
                    self._current_user = value
            if not self.ui.muted:
                self.ui.set_state("LISTENING")
            return "__SILENT__"

        if name == "identify_user":
            method = args.get("method", "both")
            identified = self._do_identify_user(method)
            if not self.ui.muted:
                self.ui.set_state("LISTENING")
            return f"Identified user: {identified}" if identified != "unknown" else "User not recognised."

        if name == "register_user":
            result = self._do_register_user(args.get("name", "User"), args.get("method", "both"))
            if not self.ui.muted:
                self.ui.set_state("LISTENING")
            return result

        if name == "agent_task":
            if not is_agent_task_allowed(caller_class):
                # Hard-denied unconditionally for every service:* caller
                # at this phase, regardless of anything in
                # permission_policy — agent_task doesn't route through
                # dispatch_tool() at all, so this is the only gate it has.
                return "'agent_task' is not permitted for this caller."

            from agent.task_queue import get_queue, TaskPriority
            goal = args.get("goal", "").strip()
            if not goal:
                return "No goal was provided for the background task."
            prio_map = {
                "high":   TaskPriority.HIGH,
                "normal": TaskPriority.NORMAL,
                "low":    TaskPriority.LOW,
            }
            priority = prio_map.get(str(args.get("priority", "normal")).lower(),
                                    TaskPriority.NORMAL)

            def _on_done(task_id: str, result: str) -> None:
                self.ui.write_log(f"AGENT: [{task_id}] complete")
                if result:
                    self.speak(str(result))

            task_id = get_queue().submit(
                goal        = goal,
                priority    = priority,
                speak       = self.speak,
                on_complete = _on_done,
                submitted_interactively = True,
            )
            self.ui.write_log(f"AGENT: [{task_id}] started — {goal[:60]}")
            if not self.ui.muted:
                self.ui.set_state("LISTENING")
            return f"Working on that in the background, sir. Task {task_id}."

        if name == "list_pending_approvals":
            pending = TASK_APPROVAL.list_pending()
            if not pending:
                return "No background tasks are waiting on approval right now."
            lines = [
                f"#{p['approval_id']} (task {p['task_id']}): {p['prompt']}"
                for p in pending
            ]
            return f"{len(pending)} pending approval(s):\n" + "\n".join(lines)

        if name == "approve_task":
            approval_id = args.get("approval_id")
            try:
                approval_id = int(approval_id)
            except (TypeError, ValueError):
                return "Please give me a valid approval ID — say 'list pending approvals' to see them."
            approve = str(args.get("approve", "true")).strip().lower() not in ("false", "no", "0", "deny", "denied")
            ok = TASK_APPROVAL.answer(approval_id, approve)
            if not ok:
                return f"Approval #{approval_id} isn't waiting on a decision — it may already be resolved or never existed."
            return f"Approval #{approval_id} {'granted' if approve else 'denied'}. The task will continue shortly."

        result = "Done."
        try:
            if name == "shutdown_jarvis":
                self.ui.write_log("SYS: Shutdown requested.")
                def _shutdown():
                    import time, os
                    self.speak("Goodbye.")
                    time.sleep(2.5)
                    os._exit(0)
                threading.Thread(target=_shutdown, daemon=True).start()
                return "Shutting down."

            elif name in TOOL_DISPATCH:
                result = dispatch_tool(name, args, self.ui, self.speak, task_id=None)

            else:
                result = f"Unknown tool: {name}"

        except Exception as e:
            result = f"Tool '{name}' failed: {e}"
            _log.error("Tool '%s' failed: %s", name, e, exc_info=True)
            self.speak_error(name, e)

        if not self.ui.muted:
            self.ui.set_state("LISTENING")
        return result

    # ── LLM processing loop ───────────────────────────────────────────────────
    def _process_message(self, user_text: str) -> None:
        self.ui.set_state("THINKING")

        # Prepend speaker name if known
        display = f"{self._current_user}: {user_text}" if self._current_user else user_text
        self.ui.write_log(f"You: {display}")

        self._conversation.append({"role": "user", "content": user_text})

        # Sliding window with auto-summarisation
        if len(self._conversation) > self.MAX_HISTORY:
            self._conversation = self._auto_summarise(self._conversation)

        messages = [
            {"role": "system", "content": self._build_system_prompt()}
        ] + list(self._conversation)

        _NEEDS_LLM_ROUND = {"web_search", "screen_process", "agent_task"}

        for _round in range(self.MAX_TOOL_ROUNDS):
            final_content    = ""
            final_tool_calls: list = []
            _streamed: list[str]   = []

            try:
                for event in call_llm_stream(messages, OLLAMA_TOOLS, purpose="conversation turn"):
                    if event["type"] == "sentence":
                        _streamed.append(event["text"])
                        self.speak(event["text"])
                    elif event["type"] == "done":
                        final_content    = event["content"]
                        final_tool_calls = event["tool_calls"]
            except RuntimeError as e:
                self.speak_error("LLM", e)
                return

            if not final_tool_calls:
                assistant_msg = {"role": "assistant", "content": final_content}
                messages.append(assistant_msg)
                self._conversation.append(assistant_msg)
                self.ui.write_log(f"Jarvis: {final_content}")
                if not _streamed and final_content:
                    self.speak(final_content)
                break

            # Tool execution
            assistant_msg = {
                "role":       "assistant",
                "content":    final_content,
                "tool_calls": final_tool_calls,
            }
            messages.append(assistant_msg)
            self._conversation.append(assistant_msg)

            tool_results = []
            for tc in final_tool_calls:
                fn_name = tc.get("function", {}).get("name", "")
                fn_args = tc.get("function", {}).get("arguments", {})
                if isinstance(fn_args, str):
                    try:
                        fn_args = json.loads(fn_args)
                    except Exception:
                        fn_args = {}

                result = self._execute_tool(fn_name, fn_args)
                if result == "__SILENT__":
                    continue

                tool_results.append({
                    "role":         "tool",
                    "tool_call_id": tc.get("id", fn_name),
                    "content":      str(result),
                })

                # Speak non-LLM tool results directly
                if fn_name not in _NEEDS_LLM_ROUND and result not in ("Done.", ""):
                    self.speak(result)

            if tool_results:
                messages.extend(tool_results)
                self._conversation.extend(tool_results)
            else:
                break  # all silent tools, no second LLM pass needed

    def _auto_summarise(self, conv: list) -> list:
        """Keep the last 10 turns; prepend a brief summary of older ones."""
        old  = conv[:-10]
        keep = conv[-10:]
        if not old:
            return keep
        # Build a quick summary via LLM (non-streaming, short)
        summary_prompt = (
            "Summarise the following conversation excerpt in 3 bullet points, "
            "focusing on facts and decisions made:\n\n"
            + "\n".join(f"{m['role'].upper()}: {m.get('content','')}" for m in old)
        )
        try:
            summary_response = call_llm(
                [
                    {"role": "system", "content": "You are a concise summariser."},
                    {"role": "user", "content": summary_prompt},
                ],
                tools=[],
                purpose="auto-summarise conversation",
            )
            summary_text = summary_response.get("content", "").strip()
            if not summary_text:
                summary_text = "(earlier context omitted)"
        except Exception:
            summary_text = "(earlier context omitted)"
        summary_msg = {
            "role":    "system",
            "content": f"[CONTEXT SUMMARY — earlier conversation]\n{summary_text}",
        }
        return [summary_msg] + keep

    # ── Listen loops ──────────────────────────────────────────────────────────
    def _listen_vad_buffered(self) -> None:
        """Shared mic loop for VAD-buffered STT engines (whisper/vosk): a
        local _VADBuffer decides utterance boundaries, then self._stt
        transcribes the finished chunk. Mic stays live while JARVIS talks so
        barge-in can be detected (see _BargeInDetector)."""
        vad   = [_VADBuffer()]   # boxed so the callback can swap in a fresh buffer on barge-in
        barge = _BargeInDetector(speech_thresh=vad[0]._speech_thresh,
                                  confirm_sec=vad[0]._barge_in_speech_sec,
                                  sample_rate=SAMPLE_RATE_IN)
        nudged = [False]

        def _transcribe_and_enqueue(utterance: np.ndarray) -> None:
            self._last_voice_buf = utterance   # store for voice-ID
            t = self._stt.transcribe(utterance)
            if t and t.strip():
                gated = self._wake_gate(t.strip())
                if gated:
                    self._enqueue_command(gated)

        def _cb(indata, frames, time_info, status):
            if self.ui.muted:
                return
            audio = indata[:, 0].astype(np.float32)

            with self._speaking_lock:
                speaking_now = self._speaking

            if speaking_now:
                # Mic stays live while JARVIS talks — watch for a real
                # interruption instead of gating the audio off entirely.
                if barge.check(audio):
                    trigger_audio = barge.drain()
                    self._handle_barge_in()
                    vad[0] = _VADBuffer()
                    nudged[0] = False
                    utterance = vad[0].process(trigger_audio)
                    if utterance is not None:
                        _transcribe_and_enqueue(utterance)
                return
            barge.reset()

            # JARVIS isn't actively speaking — the short post-TTS grace
            # window still applies here (avoids speaker bleed right at the
            # tail end of a normal, non-interrupted response).
            if self._echo_gated():
                return

            utterance = vad[0].process(audio)
            if vad[0].should_nudge():
                nudged[0] = True
                if not self.ui.muted:
                    self.ui.set_state("THINKING")   # UI-only "still listening" cue
            elif nudged[0] and vad[0]._in_spch and vad[0]._sil_cnt == 0:
                # user resumed talking right after the nudge — don't leave
                # the UI looking like it's processing while they're still mid-sentence
                nudged[0] = False
                if not self.ui.muted:
                    self.ui.set_state("LISTENING")

            if utterance is not None:
                nudged[0] = False
                _transcribe_and_enqueue(utterance)

        with sd.InputStream(samplerate=SAMPLE_RATE_IN, channels=CHANNELS,
                            blocksize=BLOCK_SIZE, dtype="float32", callback=_cb):
            while True:
                threading.Event().wait(1)

    def _listen_whisper(self) -> None:
        self._listen_vad_buffered()

    def _listen_deepgram(self) -> None:
        # Deepgram is a streaming API, not VAD-buffered like whisper/vosk:
        # raw audio is forwarded to Deepgram's WebSocket continuously (that
        # part is unchanged), and Deepgram's own server-side endpointing
        # decides utterance boundaries — there's no local _VADBuffer here to
        # swap out, so barge-in and the "still listening" nudge are driven
        # off a lightweight local RMS watch inside DeepgramStreamingSTT
        # itself, in parallel with (not instead of) the transcript stream.
        from core.stt_deepgram import DeepgramStreamingSTT

        stt_language = self._config.get("stt_language", "auto")

        def _on_transcript(text: str) -> None:
            if self.ui.muted or self._echo_gated(): return
            if text and text.strip():
                gated = self._wake_gate(text.strip())
                if gated:
                    self._enqueue_command(gated)

        def _is_speaking() -> bool:
            with self._speaking_lock:
                return self._speaking

        def _is_muted() -> bool:
            return self.ui.muted

        def _on_barge_in() -> None:
            # The interrupting audio itself isn't lost: Deepgram already had
            # it (the mic is streamed to Deepgram continuously, never
            # gated), so the transcript for it just needs to not be dropped
            # by _on_transcript's echo gate — which _handle_barge_in ensures
            # by flipping self._speaking off immediately.
            self._handle_barge_in()

        def _on_silence_nudge() -> None:
            if not self.ui.muted:
                self.ui.set_state("THINKING")   # UI-only "still listening" cue

        streamer = DeepgramStreamingSTT(
            language         = stt_language,
            on_transcript    = _on_transcript,
            is_speaking      = _is_speaking,
            is_muted         = _is_muted,
            on_barge_in      = _on_barge_in,
            on_silence_nudge = _on_silence_nudge,
        )
        streamer.start()
        try:
            while True:
                threading.Event().wait(1)
        finally:
            streamer.stop()

    def _listen_vosk(self) -> None:
        self._listen_vad_buffered()

    def _text_command_loop(self) -> None:
        while True:
            try:
                text = self._text_queue.get(timeout=0.5)
                if not text.strip():
                    continue

                normalized = text.strip().lower()
                now        = time.time()

                if (normalized == self._last_command_text
                        and (now - self._last_command_time) < self._dedup_window_sec):
                    _log.debug("Dedup: skipped duplicate at consumer: %r", text)
                    continue

                self._last_command_text = normalized
                self._last_command_time = now
                self._process_message(text)
            except queue.Empty:
                pass

    # ── Startup face scan ─────────────────────────────────────────────────────
    def _startup_face_scan(self) -> None:
        """Try to identify the user from camera on startup (non-blocking)."""
        self.ui.write_log("REC: Scanning for known face…")
        name, conf = self._face_id.identify(timeout=5.0)
        if name:
            self._current_user    = name
            self._user_confidence = conf
            self.ui.write_log(f"REC: Welcome back, {name}! ({conf:.0%})")
            self.speak(f"Welcome back, {name}.")
        else:
            self.ui.write_log("REC: Face not recognised — will learn on introduction.")

    # ── Reconfigure ───────────────────────────────────────────────────────────
    def reconfigure(self, new_config: dict) -> None:
        threading.Thread(target=self._do_reconfigure, args=(new_config,), daemon=True).start()

    def _do_reconfigure(self, new_config: dict) -> None:
        old_stt = self._config.get("stt_engine", "whisper").lower()
        self._config = new_config
        try:
            from core.installer import install_for_config
            install_for_config(new_config, log=self.ui.write_log)
        except Exception as e:
            self.ui.write_log(f"ERR: Dependency install — {e}")
        try:
            from core.tts import create_tts_player
            self._tts = create_tts_player(new_config)
            self._tts_ready.set()
            self.ui.write_log("SYS: TTS reconfigured.")
        except Exception as e:
            self.ui.write_log(f"ERR: TTS reconfigure — {e}")
        new_stt = new_config.get("stt_engine", "whisper").lower()
        if old_stt == new_stt:
            try:
                lang = new_config.get("stt_language", "auto")
                if new_stt == "vosk":
                    from core.stt import VoskSTT
                    self._stt = VoskSTT(new_config.get("vosk_model_path"), language=lang)
                else:
                    from core.stt import WhisperSTT
                    self._stt = WhisperSTT(new_config.get("stt_model", "base"), language=lang)
                self.ui.write_log("SYS: STT reconfigured.")
            except Exception as e:
                self.ui.write_log(f"ERR: STT reconfigure — {e}")
        else:
            self.ui.write_log("SYS: STT engine changed — restart required.")
        self.speak("Configuration applied.")

    # ── Run ───────────────────────────────────────────────────────────────────
    def run(self) -> None:
        try:
            self.ui.on_reconfigure = self.reconfigure

            from core.llm_client import ensure_ollama_running, warmup_model
            self.ui.write_log("SYS: Checking Ollama…")
            if ensure_ollama_running():
                self.ui.write_log("SYS: Ollama OK.")
            else:
                self.ui.write_log("ERR: Ollama unavailable — run: ollama serve")

            stt_engine   = self._config.get("stt_engine",   "whisper").lower()
            stt_language = self._config.get("stt_language", "auto")
            stt_model    = self._config.get("stt_model",    "base")
            tts_engine   = self._config.get("tts_engine",   "edgetts").lower()

            self.ui.show_startup_panel()

            _warmup_done = threading.Event()
            _stt_done    = threading.Event()

            def _do_warmup():
                try:
                    static_prompt = _load_system_prompt()
                    warmup_model(system_prompt=static_prompt)
                    self.ui.write_log("SYS: LLM ready.")
                    self.ui.mark_startup_ready("llm")
                except Exception as e:
                    self.ui.write_log(f"ERR: LLM warmup — {e}")
                    self.ui.mark_startup_ready("llm", error=True)
                finally:
                    _warmup_done.set()

            def _do_stt():
                try:
                    self.ui.write_log(f"SYS: Loading {stt_engine.upper()} STT…")
                    if stt_engine == "deepgram":
                        from core.stt_deepgram import _get_api_key
                        if not _get_api_key():
                            raise ValueError("Deepgram API key not found in config/api_keys.json")
                    elif stt_engine == "vosk":
                        from core.stt import VoskSTT
                        self._stt = VoskSTT(self._config.get("vosk_model_path"), language=stt_language)
                    else:
                        from core.stt import WhisperSTT
                        self._stt = WhisperSTT(stt_model, language=stt_language)
                    self.ui.write_log("SYS: STT ready.")
                    self.ui.mark_startup_ready("stt")
                except Exception as e:
                    self.ui.write_log(f"ERR: STT — {e}")
                    self.ui.mark_startup_ready("stt", error=True)
                finally:
                    _stt_done.set()

            def _do_tts():
                try:
                    self.ui.write_log(f"SYS: Loading {tts_engine.upper()} TTS…")
                    from core.tts import create_tts_player
                    self._tts = create_tts_player(self._config)
                    self._tts_ready.set()
                    self.ui.write_log("SYS: TTS ready.")
                    self.ui.mark_startup_ready("tts")
                    self.ui.set_startup_status("● All systems ready.")
                    self.ui.hide_startup_panel()
                    # Face scan runs after TTS is ready so the greeting can be spoken
                    self._startup_face_scan()
                    if not self._current_user:
                        self.speak("JARVIS online. I don't recognise you yet — please introduce yourself.")
                except Exception as e:
                    _log.error("TTS startup failed: %s", e, exc_info=True)
                    self.ui.write_log(f"ERR: TTS — {e}")
                    self.ui.mark_startup_ready("tts", error=True)
                    self._tts_ready.set()

            self.ui.write_log("SYS: Loading systems in parallel…")
            threading.Thread(target=_do_warmup, daemon=True).start()
            threading.Thread(target=_do_stt,    daemon=True).start()
            threading.Thread(target=_do_tts,    daemon=True).start()

            _warmup_done.wait(timeout=60)
            _stt_done.wait(timeout=60)

            self.ui.write_log("SYS: JARVIS-XL online.")
            self.ui.set_state("LISTENING")

            threading.Thread(target=self._tts_worker,       daemon=True).start()
            threading.Thread(target=self._text_command_loop, daemon=True).start()
            threading.Thread(target=self._proactive.run,     daemon=True).start()

            if stt_engine == "deepgram":
                self._listen_deepgram()
            elif stt_engine == "vosk":
                self._listen_vosk()
            elif stt_engine == "whisper":
                self._listen_whisper()

        except Exception as e:
            self.ui.write_log(f"ERR: Init failed — {e}")
            _log.error("Init failed: %s", e, exc_info=True)


# ── Entry ─────────────────────────────────────────────────────────────────────
def main() -> None:
    def _preload_torch():
        try:
            import torch  # noqa
        except Exception:
            pass
    threading.Thread(target=_preload_torch, daemon=True).start()

    ui = JarvisUI("face.png")

    def runner():
        ui.wait_for_api_key()
        ui.write_log("SYS: Checking dependencies…")
        cfg = load_config(force_reload=True)
        _install_done = threading.Event()

        def _do_install():
            try:
                from core.installer import install_for_config
                install_for_config(cfg, log=ui.write_log)
            except Exception as e:
                ui.write_log(f"ERR: Dependency install — {e}")
            finally:
                _install_done.set()

        threading.Thread(target=_do_install, daemon=True).start()
        _install_done.wait()

        jarvis = JarvisXL(ui)
        try:
            jarvis.run()
        except KeyboardInterrupt:
            # print(), not logging: the QueueListener thread is a daemon
            # thread (see core/logging_setup.py) — on interpreter shutdown
            # right after this it may never get scheduled to drain the
            # queue, so a logger call here has a real chance of never
            # appearing anywhere. This message's only audience is a
            # terminal-watching developer; print() guarantees it's seen.
            print("\n[JARVIS-XL] Shutting down…")

    threading.Thread(target=runner, daemon=True).start()
    ui.root.mainloop()

if __name__ == "__main__":
    main()
