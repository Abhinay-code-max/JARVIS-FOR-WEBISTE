"""
JARVIS-XL — Enhanced Local AI Assistant with Deepgram STT
"""

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
import queue
import re
import sys
import threading
import traceback
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

from actions.file_processor    import file_processor
from actions.flight_finder     import flight_finder
from actions.open_app          import open_app
from actions.weather_report    import weather_action
from actions.daily_briefing    import daily_briefing
from actions.auto_fix_code     import auto_fix_code
from actions.vision_fix_code   import vision_fix_code
from actions.send_message      import send_message
from actions.reminder          import reminder
from actions.computer_settings import computer_settings
from actions.screen_processor  import screen_process
from actions.youtube_video     import youtube_video
from actions.desktop           import desktop_control
from actions.browser_control   import browser_control
from actions.file_controller   import file_controller
from actions.code_helper       import code_helper
from actions.dev_agent         import dev_agent
from actions.web_search        import web_search as web_search_action
from actions.computer_control  import computer_control
from actions.game_updater      import game_updater

# ── Paths ────────────────────────────────────────────────────────────────────
def _get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent

BASE_DIR        = _get_base_dir()
API_CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"
PROMPT_PATH     = BASE_DIR / "core" / "prompt.txt"
FACES_DIR       = BASE_DIR / "recognition" / "faces"
VOICES_DIR      = BASE_DIR / "recognition" / "voices"

SAMPLE_RATE_IN  = 16_000
BLOCK_SIZE      = 1_024
CHANNELS        = 1


# ── Tool declarations ────────────────────────────────────────────────────────
TOOL_DECLARATIONS = [
    {
        "name": "identify_user",
        "description": (
            "Identifies who is currently in front of the camera or speaking. "
            "Call this when: the user's identity is uncertain, someone new appears, "
            "or when asked 'do you know who I am'. Returns the recognised user profile."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "method": {
                    "type": "STRING",
                    "description": "'face' to use camera, 'voice' to use recent audio, 'both' for combined"
                }
            },
            "required": []
        }
    },
    {
        "name": "register_user",
        "description": (
            "Registers a new user by capturing their face and/or voice sample. "
            "Call when the user says 'remember me', 'learn my face', 'add me', or similar."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "name":   {"type": "STRING", "description": "The person's name"},
                "method": {"type": "STRING", "description": "'face', 'voice', or 'both'"}
            },
            "required": ["name"]
        }
    },
    {
        "name": "open_app",
        "description": (
            "Opens or launches any application, website, or program. "
            "Use when the user says: open, launch, start, run, pull up. "
            "Do NOT use send_message just because the app is a messaging app."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "app_name": {"type": "STRING", "description": "Name of the application or website"}
            },
            "required": ["app_name"]
        }
    },
    {
        "name": "web_search",
        "description": "Searches the web for any information.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query":  {"type": "STRING"},
                "mode":   {"type": "STRING", "description": "search (default) or compare"},
                "items":  {"type": "ARRAY",  "items": {"type": "STRING"}},
                "aspect": {"type": "STRING", "description": "price | specs | reviews"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "weather_report",
        "description": "Gets weather for any city.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"city": {"type": "STRING"}},
            "required": ["city"]
        }
    },
    {
        "name": "daily_briefing",
        "description": (
            "Delivers a complete daily briefing: greeting, weather, today's reminders, "
            "and top news headlines. Call this when the user says 'good morning', "
            "'give me my briefing', 'what's happening today', 'catch me up', or similar."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "city": {"type": "STRING", "description": "Optional city override for weather"}
            },
            "required": []
        }
    },
    {
        "name": "send_message",
        "description": (
            "Sends a message via WhatsApp, Telegram, etc. "
            "Only call when BOTH a recipient AND message content are given."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "receiver":     {"type": "STRING"},
                "message_text": {"type": "STRING"},
                "platform":     {"type": "STRING"}
            },
            "required": ["receiver", "message_text", "platform"]
        }
    },
    {
        "name": "reminder",
        "description": "Sets a timed reminder.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "date":    {"type": "STRING", "description": "YYYY-MM-DD"},
                "time":    {"type": "STRING", "description": "HH:MM (24h)"},
                "message": {"type": "STRING"}
            },
            "required": ["date", "time", "message"]
        }
    },
    {
        "name": "youtube_video",
        "description": "Plays, summarizes, or gets info about YouTube videos.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "play | summarize | get_info | trending"},
                "query":  {"type": "STRING"},
                "save":   {"type": "BOOLEAN"},
                "region": {"type": "STRING"},
                "url":    {"type": "STRING"}
            },
            "required": []
        }
    },
    {
        "name": "screen_process",
        "description": (
            "Captures and analyzes the screen or webcam. "
            "MUST be called when user asks what is on screen, what you see, analyze screen, etc."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "angle": {"type": "STRING", "description": "screen or camera"},
                "text":  {"type": "STRING", "description": "Question about the captured image"}
            },
            "required": ["text"]
        }
    },
    {
        "name": "computer_settings",
        "description": "Controls volume, brightness, windows, shortcuts, typing, WiFi, restart, etc.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING"},
                "description": {"type": "STRING"},
                "value":       {"type": "STRING"}
            },
            "required": []
        }
    },
    {
        "name": "browser_control",
        "description": "Controls any web browser: open sites, click, type, scroll, fill forms.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING"},
                "browser":     {"type": "STRING"},
                "url":         {"type": "STRING"},
                "query":       {"type": "STRING"},
                "selector":    {"type": "STRING"},
                "text":        {"type": "STRING"},
                "description": {"type": "STRING"},
                "direction":   {"type": "STRING"},
                "amount":      {"type": "INTEGER"},
                "key":         {"type": "STRING"},
                "incognito":   {"type": "BOOLEAN"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "file_controller",
        "description": "Manages files: list, create, delete, move, copy, rename, read, write, find.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING"},
                "path":        {"type": "STRING"},
                "destination": {"type": "STRING"},
                "new_name":    {"type": "STRING"},
                "content":     {"type": "STRING"},
                "name":        {"type": "STRING"},
                "extension":   {"type": "STRING"},
                "count":       {"type": "INTEGER"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "desktop_control",
        "description": "Controls the desktop: wallpaper, organize, clean, stats.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING"},
                "path":   {"type": "STRING"},
                "url":    {"type": "STRING"},
                "mode":   {"type": "STRING"},
                "task":   {"type": "STRING"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "code_helper",
        "description": "Writes, edits, explains, or runs code.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING"},
                "description": {"type": "STRING"},
                "language":    {"type": "STRING"},
                "output_path": {"type": "STRING"},
                "file_path":   {"type": "STRING"},
                "code":        {"type": "STRING"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "dev_agent",
        "description": "Builds complete multi-file projects from scratch.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "description":  {"type": "STRING"},
                "language":     {"type": "STRING"},
                "project_name": {"type": "STRING"},
            },
            "required": ["description"]
        }
    },
    {
        "name": "computer_control",
        "description": "Direct mouse/keyboard control: click, type, scroll, hotkeys.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING"},
                "text":        {"type": "STRING"},
                "x":           {"type": "INTEGER"},
                "y":           {"type": "INTEGER"},
                "keys":        {"type": "STRING"},
                "direction":   {"type": "STRING"},
                "amount":      {"type": "INTEGER"},
                "seconds":     {"type": "NUMBER"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "game_updater",
        "description": "Steam / Epic Games install, update, and management.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":    {"type": "STRING"},
                "platform":  {"type": "STRING"},
                "game_name": {"type": "STRING"},
                "app_id":    {"type": "STRING"},
                "hour":      {"type": "INTEGER"},
                "minute":    {"type": "INTEGER"},
                "shutdown_when_done": {"type": "BOOLEAN"},
            },
            "required": []
        }
    },
    {
        "name": "flight_finder",
        "description": "Searches Google Flights and speaks the best options.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "origin":      {"type": "STRING"},
                "destination": {"type": "STRING"},
                "date":        {"type": "STRING"},
                "return_date": {"type": "STRING"},
                "passengers":  {"type": "INTEGER"},
                "cabin":       {"type": "STRING"},
            },
            "required": ["origin", "destination", "date"]
        }
    },
    {
        "name": "file_processor",
        "description": "Processes uploaded files: images, PDFs, CSV, audio, video.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "file_path":   {"type": "STRING"},
                "action":      {"type": "STRING"},
                "instruction": {"type": "STRING"},
                "format":      {"type": "STRING"},
            },
            "required": []
        }
    },
    {
        "name": "auto_fix_code",
        "description": (
            "Runs a Python script, captures the REAL error if it crashes, "
            "and automatically fixes the bug by editing the file directly. "
            "Call this when the user says 'fix this code', 'fix the error', "
            "'run and fix', 'debug this file', or wants JARVIS to directly "
            "correct a script rather than just explain the error."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "file_path": {"type": "STRING", "description": "Full path to the Python file to run and fix"},
                "max_attempts": {"type": "INTEGER", "description": "Max fix attempts, default 2"}
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "vision_fix_code",
        "description": (
            "Looks at whatever code is currently visible on the screen (in an editor "
            "or terminal), automatically identifies the file and the bug, and directly "
            "fixes the file on disk. Call this when the user says things like "
            "'fix this', 'fix this code', 'fix the error', 'correct it', 'fix it for me', "
            "without needing them to specify a file path — you figure it out from the screen."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {}
        }
    },
    {
        "name": "save_memory",
        "description": (
            "Save a personal fact about the user. "
            "Call IMMEDIATELY when user states name, age, city, job, preference, or relationship. "
            "Call SILENTLY — never announce you are saving."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "category": {"type": "STRING",
                             "description": "identity | preferences | projects | relationships | wishes | notes"},
                "key":      {"type": "STRING"},
                "value":    {"type": "STRING"},
            },
            "required": ["category", "key", "value"]
        }
    },
    {
        "name": "shutdown_jarvis",
        "description": "Shuts down JARVIS when the user says goodbye or stop.",
        "parameters": {"type": "OBJECT", "properties": {}}
    },
]


# ── Type conversion ──────────────────────────────────────────────────────────
_TYPE_MAP = {
    "OBJECT": "object", "STRING": "string", "ARRAY": "array",
    "INTEGER": "integer", "BOOLEAN": "boolean", "NUMBER": "number",
}

def _convert_type(t): return _TYPE_MAP.get(t, t.lower()) if isinstance(t, str) else t

def _convert_props(props):
    out = {}
    for k, v in props.items():
        nv = dict(v)
        if "type" in nv: nv["type"] = _convert_type(nv["type"])
        if "items" in nv and isinstance(nv["items"], dict):
            nv["items"] = {"type": _convert_type(nv["items"].get("type", "string"))}
        out[k] = nv
    return out

def _to_ollama_tools(decls):
    tools = []
    for d in decls:
        params = d.get("parameters", {})
        new_params = {
            "type": "object",
            "properties": _convert_props(params.get("properties", {})),
        }
        req = params.get("required")
        if req: new_params["required"] = req
        tools.append({
            "type": "function",
            "function": {
                "name":        d["name"],
                "description": d["description"],
                "parameters":  new_params,
            },
        })
    return tools

OLLAMA_TOOLS = _to_ollama_tools(TOOL_DECLARATIONS)


# ── Helpers ──────────────────────────────────────────────────────────────────
def _load_config() -> dict:
    try:
        with open(API_CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

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


# ── Voice Activity Detection ─────────────────────────────────────────────────
class _VADBuffer:
    def __init__(self, sample_rate=16_000, silence_sec=0.7,
                 speech_thresh=0.008, silence_thresh=0.004,
                 min_speech_sec=0.3, max_speech_sec=30.0):
        self._sr            = sample_rate
        self._sil_n         = int(silence_sec * sample_rate)
        self._speech_thresh = speech_thresh
        self._sil_thresh    = silence_thresh
        self._min_n         = int(min_speech_sec * sample_rate)
        self._max_n         = int(max_speech_sec * sample_rate)
        self._buf:          list = []
        self._in_spch       = False
        self._sil_cnt       = 0

    def process(self, chunk: np.ndarray):
        rms     = float(np.sqrt(np.mean(chunk ** 2)))
        total_n = sum(len(c) for c in self._buf)
        if rms > self._speech_thresh:
            self._in_spch = True
            self._sil_cnt = 0
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
                if len(audio) >= self._min_n:
                    return audio
        return None


# ── Main assistant ────────────────────────────────────────────────────────────
class JarvisXL:

    MAX_HISTORY     = 20
    MAX_TOOL_ROUNDS = 6

    def __init__(self, ui: JarvisUI):
        self.ui                = ui
        self._config           = _load_config()
        self._stt              = None
        self._tts              = None
        self._tts_ready        = threading.Event()
        self._speaking         = False
        self._speaking_lock    = threading.Lock()
        self._text_queue:      queue.Queue = queue.Queue()
        self._tts_queue:       queue.Queue = queue.Queue()
        self._conversation:    list[dict]  = []

        self._face_id:  FaceIdentifier  = FaceIdentifier(FACES_DIR)
        self._voice_id: VoiceIdentifier = VoiceIdentifier(VOICES_DIR)
        self._wake:     WakeWordDetector = WakeWordDetector()

        self._current_user:    str | None = None
        self._user_confidence: float      = 0.0
        self._last_voice_buf:  np.ndarray | None = None

        # Continuous conversation mode
        self._conversation_active_until: float = 0.0
        self._conversation_window_sec:   float = 8.0

        # Duplicate-command guard
        self._last_command_text: str = ""
        self._last_command_time: float = 0.0
        self._dedup_window_sec:  float = 2.5

        self.ui.on_text_command = self._on_text_command

    # ── System prompt ─────────────────────────────────────────────────────────
    def _build_system_prompt(self) -> str:
        sys_p   = _load_system_prompt()
        memory  = load_memory()
        mem_str = format_memory_for_prompt(memory)
        now     = datetime.now()

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

        parts = [sys_p]
        if user_ctx:  parts.append(user_ctx)
        if mem_str:   parts.append(mem_str)
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
                print(f"[TTS] {e}")
            finally:
                self._tts_queue.task_done()
                if self._tts_queue.empty():
                    with self._speaking_lock:
                        self._speaking = False
                    if not self.ui.muted:
                        self.ui.set_state("LISTENING")

    def speak(self, text: str) -> None:
        if not text or not self._tts:
            return
        with self._speaking_lock:
            self._speaking = True
        self._tts_queue.put(text)
        # Open continuous-conversation window once this finishes speaking
        import time as _time
        self._conversation_active_until = _time.time() + self._conversation_window_sec + (len(text) * 0.05)

    def speak_error(self, tool_name: str, error) -> None:
        self.ui.write_log(f"ERR: {tool_name} — {str(error)[:120]}")
        self.speak(f"{tool_name} encountered an error.")

    def _on_text_command(self, text: str) -> None:
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
        self._text_queue.put(text)

    def _is_conversation_active(self) -> bool:
        """True if we're within the post-response window (no wake word needed)."""
        import time as _time
        return _time.time() < self._conversation_active_until

    def _extend_conversation_window(self) -> None:
        """Call after processing a follow-up to keep the window open a bit longer."""
        import time as _time
        self._conversation_active_until = _time.time() + self._conversation_window_sec

    # ── Recognition ───────────────────────────────────────────────────────────
    def _do_identify_user(self, method: str = "both") -> str:
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
        best = max(results, key=lambda x: x[1])
        name, conf, source = best
        self._current_user    = name
        self._user_confidence = conf
        self.ui.write_log(f"REC: Identified '{name}' via {source} ({conf:.0%})")
        return name

    def _do_register_user(self, name: str, method: str = "both") -> str:
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
    def _execute_tool(self, name: str, args: dict) -> str:
        print(f"[JARVIS-XL] 🔧 {name}  {args}")
        self.ui.set_state("THINKING")

        if name == "save_memory":
            category = args.get("category", "notes")
            key      = args.get("key", "")
            value    = args.get("value", "")
            if key and value:
                update_memory({category: {key: {"value": value}}})
                print(f"[Memory] 💾 {category}/{key} = {value}")
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

        result = "Done."
        try:
            if name == "open_app":
                r = open_app(parameters=args, response=None, player=self.ui)
                result = r or f"Opened {args.get('app_name')}."
            elif name == "weather_report":
                r = weather_action(parameters=args, player=self.ui)
                result = r or "Weather delivered."

            elif name == "daily_briefing":
                r = daily_briefing(parameters=args, player=self.ui, speak=self.speak)
                briefing_text = r or "I don't have anything to brief you on right now."
                self.speak(briefing_text)
                self.ui.write_log(f"Jarvis: {briefing_text}")
                if not self.ui.muted:
                    self.ui.set_state("LISTENING")
                return "__SILENT__"
            elif name == "browser_control":
                r = browser_control(parameters=args, player=self.ui)
                result = r or "Done."
            elif name == "file_controller":
                r = file_controller(parameters=args, player=self.ui)
                result = r or "Done."
            elif name == "send_message":
                r = send_message(parameters=args, response=None, player=self.ui, session_memory=None)
                result = r or f"Message sent to {args.get('receiver')}."
            elif name == "reminder":
                r = reminder(parameters=args, response=None, player=self.ui)
                result = r or "Reminder set."
            elif name == "youtube_video":
                r = youtube_video(parameters=args, response=None, player=self.ui)
                result = r or "Done."
            elif name == "screen_process":
                r = screen_process(parameters=args, response=None, player=self.ui, session_memory=None)
                result = r if isinstance(r, str) and r else "Screen analyzed."
            elif name == "computer_settings":
                r = computer_settings(parameters=args, response=None, player=self.ui)
                result = r or "Done."
            elif name == "desktop_control":
                r = desktop_control(parameters=args, player=self.ui)
                result = r or "Done."
            elif name == "code_helper":
                r = code_helper(parameters=args, player=self.ui, speak=self.speak)
                result = r or "Done."
            elif name == "dev_agent":
                r = dev_agent(parameters=args, player=self.ui, speak=self.speak)
                result = r or "Done."
            elif name == "web_search":
                r = web_search_action(parameters=args, player=self.ui)
                result = r or "Done."
            elif name == "file_processor":
                if not args.get("file_path") and self.ui.current_file:
                    args["file_path"] = self.ui.current_file
                r = file_processor(parameters=args, player=self.ui, speak=self.speak)
                result = r or "Done."
            elif name == "computer_control":
                r = computer_control(parameters=args, player=self.ui)
                result = r or "Done."
            elif name == "game_updater":
                r = game_updater(parameters=args, player=self.ui, speak=self.speak)
                result = r or "Done."
            elif name == "flight_finder":
                r = flight_finder(parameters=args, player=self.ui)
                result = r or "Done."
            elif name == "auto_fix_code":
                r = auto_fix_code(parameters=args, player=self.ui, speak=self.speak)
                result = r or "Done attempting to fix the code."

            elif name == "vision_fix_code":
                r = vision_fix_code(parameters=args, player=self.ui, speak=self.speak)
                result = r or "Done looking at the code."

            elif name == "shutdown_jarvis":
                self.ui.write_log("SYS: Shutdown requested.")
                def _shutdown():
                    import time, os
                    self.speak("Goodbye.")
                    time.sleep(2.5)
                    os._exit(0)
                threading.Thread(target=_shutdown, daemon=True).start()
                return "Shutting down."
            else:
                result = f"Unknown tool: {name}"
        except Exception as e:
            result = f"Tool '{name}' failed: {e}"
            traceback.print_exc()
            self.speak_error(name, e)

        if not self.ui.muted:
            self.ui.set_state("LISTENING")
        return result

    # ── LLM processing ────────────────────────────────────────────────────────
    def _process_message(self, user_text: str) -> None:
        self.ui.set_state("THINKING")
        display = f"{self._current_user}: {user_text}" if self._current_user else user_text
        self.ui.write_log(f"You: {display}")
        self._conversation.append({"role": "user", "content": user_text})
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
                for event in call_llm_stream(messages, OLLAMA_TOOLS):
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
                if fn_name not in _NEEDS_LLM_ROUND and result not in ("Done.", ""):
                    self.speak(result)

            if tool_results:
                messages.extend(tool_results)
                self._conversation.extend(tool_results)
            else:
                break

    def _auto_summarise(self, conv: list) -> list:
        old  = conv[:-10]
        keep = conv[-10:]
        if not old:
            return keep
        summary_prompt = (
            "Summarise the following conversation in 3 bullet points:\n\n"
            + "\n".join(f"{m['role'].upper()}: {m.get('content','')}" for m in old)
        )
        try:
            summary_text = call_llm(
                [{"role": "user", "content": summary_prompt}],
                tools=[], system="You are a concise summariser."
            )
        except Exception:
            summary_text = "(earlier context omitted)"
        return [{"role": "system", "content": f"[CONTEXT SUMMARY]\n{summary_text}"}] + keep

    # ── Listen loops ──────────────────────────────────────────────────────────
    def _listen_deepgram(self) -> None:
        from core.stt_deepgram import DeepgramStreamingSTT
        api_key  = self._config.get("deepgram_api_key", "")
        language = self._config.get("stt_language", "en")

        def on_transcript(text: str):
            if not text or not text.strip():
                return
            if self.ui.muted:
                return
            print(f"[Deepgram] 🎤 {text}")
            self._text_queue.put(text.strip())

        self.ui.write_log("SYS: Deepgram Nova-2 streaming STT active.")
        streamer = DeepgramStreamingSTT(
            api_key=api_key,
            language=language,
            on_transcript=on_transcript,
        )
        streamer.start()
        threading.Event().wait()

    def _listen_whisper(self) -> None:
        vad = _VADBuffer()
        def _cb(indata, frames, time_info, status):
            if self.ui.muted:
                return
            audio = indata[:, 0].astype(np.float32)
            utterance = vad.process(audio)
            if utterance is not None:
                self._last_voice_buf = utterance
                t = self._stt.transcribe(utterance)
                if t and t.strip():
                    self._text_queue.put(t.strip())
        with sd.InputStream(samplerate=SAMPLE_RATE_IN, channels=CHANNELS,
                            blocksize=BLOCK_SIZE, dtype="float32", callback=_cb):
            while True:
                threading.Event().wait(1)

    def _listen_vosk(self) -> None:
        vad = _VADBuffer()
        def _cb(indata, frames, time_info, status):
            if self.ui.muted:
                return
            audio = indata[:, 0].astype(np.float32)
            utterance = vad.process(audio)
            if utterance is not None:
                self._last_voice_buf = utterance
                t = self._stt.transcribe(utterance)
                if t and t.strip():
                    self._text_queue.put(t.strip())
        with sd.InputStream(samplerate=SAMPLE_RATE_IN, channels=CHANNELS,
                            blocksize=BLOCK_SIZE, dtype="float32", callback=_cb):
            while True:
                threading.Event().wait(1)

    def _text_command_loop(self) -> None:
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
                pass

    def _startup_face_scan(self) -> None:
        self.ui.write_log("REC: Scanning for known face...")
        name, conf = self._face_id.identify(timeout=5.0)
        if name:
            self._current_user    = name
            self._user_confidence = conf
            self.ui.write_log(f"REC: Welcome back, {name}! ({conf:.0%})")
            self.speak(f"Welcome back, {name}.")
        else:
            self.ui.write_log("REC: Face not recognised — will learn on introduction.")

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
        if old_stt == new_stt and new_stt != "deepgram":
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
            self.ui.write_log("SYS: Checking Ollama...")
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
                    self.ui.write_log(f"SYS: Loading {stt_engine.upper()} STT...")
                    if stt_engine == "deepgram":
                        from core.stt_deepgram import DeepgramSTT
                        self._stt = DeepgramSTT(
                            api_key=self._config.get("deepgram_api_key", ""),
                            language=stt_language,
                        )
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
                    self.ui.write_log(f"SYS: Loading {tts_engine.upper()} TTS...")
                    from core.tts import create_tts_player
                    self._tts = create_tts_player(self._config)
                    self._tts_ready.set()
                    self.ui.write_log("SYS: TTS ready.")
                    self.ui.mark_startup_ready("tts")
                    self.ui.set_startup_status("● All systems ready.")
                    self.ui.hide_startup_panel()
                    self._startup_face_scan()
                    if not self._current_user:
                        self.speak("JARVIS online. I don't recognise you yet — please introduce yourself.")
                except Exception as e:
                    import traceback as _tb
                    _tb.print_exc()
                    self.ui.write_log(f"ERR: TTS — {e}")
                    self.ui.mark_startup_ready("tts", error=True)
                    self._tts_ready.set()

            self.ui.write_log("SYS: Loading systems in parallel...")
            threading.Thread(target=_do_warmup, daemon=True).start()
            threading.Thread(target=_do_stt,    daemon=True).start()
            threading.Thread(target=_do_tts,    daemon=True).start()

            _warmup_done.wait(timeout=60)
            _stt_done.wait(timeout=60)

            self.ui.write_log("SYS: JARVIS-XL online.")
            self.ui.set_state("LISTENING")

            threading.Thread(target=self._tts_worker,        daemon=True).start()
            threading.Thread(target=self._text_command_loop,  daemon=True).start()

            if stt_engine == "deepgram":
                self._listen_deepgram()
            elif stt_engine == "vosk":
                self._listen_vosk()
            else:
                self._listen_whisper()

        except Exception as e:
            self.ui.write_log(f"ERR: Init failed — {e}")
            traceback.print_exc()


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
        ui.write_log("SYS: Checking dependencies...")
        cfg = _load_config()
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
            print("\n[JARVIS-XL] Shutting down...")

    threading.Thread(target=runner, daemon=True).start()
    ui.root.mainloop()

if __name__ == "__main__":
    main()
