# config/__init__.py
"""
Single source of truth for reading/writing config/api_keys.json.
Every module in JARVIS-XL imports load_config()/save_config() from here
instead of re-implementing its own JSON read.
"""
import json
import platform
import sys
from pathlib import Path


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR    = _base_dir()
CONFIG_DIR  = BASE_DIR / "config"
CONFIG_FILE = CONFIG_DIR / "api_keys.json"

_cache: dict | None = None


def load_config(force_reload: bool = False) -> dict:
    """Read config/api_keys.json (cached after the first successful read)."""
    global _cache
    if _cache is None or force_reload:
        try:
            _cache = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            _cache = {}
    return _cache


def get_config() -> dict:
    return load_config()


def ensure_config_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def config_exists() -> bool:
    return CONFIG_FILE.exists()


def save_config(cfg: dict) -> None:
    """Merge `cfg` into the existing file on disk and refresh the cache."""
    global _cache
    ensure_config_dir()
    existing = load_config(force_reload=True)
    existing.update(cfg)
    CONFIG_FILE.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    _cache = existing


def set_config_key(key: str, value) -> None:
    save_config({key: value})


def is_configured() -> bool:
    cfg = load_config(force_reload=True)
    return bool(cfg.get("llm_model")) and bool(cfg.get("stt_engine")) and bool(cfg.get("tts_engine"))


def get_os() -> str:
    """Returns 'windows' | 'mac' | 'linux' — detected at runtime, not from config."""
    s = platform.system().lower()
    if s == "darwin":
        return "mac"
    if s == "windows":
        return "windows"
    return "linux"


def is_windows() -> bool: return get_os() == "windows"
def is_mac()     -> bool: return get_os() == "mac"
def is_linux()   -> bool: return get_os() == "linux"
