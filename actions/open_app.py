import os
import re
import json
import time
import subprocess
import platform
import shutil
import logging

from config import BASE_DIR

_log = logging.getLogger("jarvis.open_app")

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

_SYSTEM = platform.system()

_APP_ALIASES: dict[str, dict[str, str]] = {

    "chrome":             {"Windows": "chrome",                  "Darwin": "Google Chrome",        "Linux": "google-chrome"},
    "google chrome":      {"Windows": "chrome",                  "Darwin": "Google Chrome",        "Linux": "google-chrome"},
    "firefox":            {"Windows": "firefox",                 "Darwin": "Firefox",              "Linux": "firefox"},
    "edge":               {"Windows": "msedge",                  "Darwin": "Microsoft Edge",       "Linux": "microsoft-edge"},
    "brave":              {"Windows": "brave",                   "Darwin": "Brave Browser",        "Linux": "brave-browser"},
    "safari":             {"Windows": "msedge",                  "Darwin": "Safari",               "Linux": "firefox"},
    "opera":              {"Windows": "opera",                   "Darwin": "Opera",                "Linux": "opera"},
    "whatsapp":           {"Windows": "whatsapp:",               "Darwin": "WhatsApp",             "Linux": "whatsapp"},
    "telegram":           {"Windows": "Telegram",                "Darwin": "Telegram",             "Linux": "telegram"},
    "discord":            {"Windows": "Discord",                 "Darwin": "Discord",              "Linux": "discord"},
    "slack":              {"Windows": "Slack",                   "Darwin": "Slack",                "Linux": "slack"},
    "zoom":               {"Windows": "Zoom",                    "Darwin": "zoom.us",              "Linux": "zoom"},
    "teams":              {"Windows": "msteams",                 "Darwin": "Microsoft Teams",      "Linux": "teams"},
    "skype":              {"Windows": "skype",                   "Darwin": "Skype",                "Linux": "skype"},
    "signal":             {"Windows": "signal",                  "Darwin": "Signal",               "Linux": "signal"},
    "spotify":            {"Windows": "Spotify",                 "Darwin": "Spotify",              "Linux": "spotify"},
    "vlc":                {"Windows": "vlc",                     "Darwin": "VLC",                  "Linux": "vlc"},
    "netflix":            {"Windows": "Netflix",                 "Darwin": "Netflix",              "Linux": "firefox"},
    "vscode":             {"Windows": "code",                    "Darwin": "Visual Studio Code",   "Linux": "code"},
    "visual studio code": {"Windows": "code",                    "Darwin": "Visual Studio Code",   "Linux": "code"},
    "code":               {"Windows": "code",                    "Darwin": "Visual Studio Code",   "Linux": "code"},
    "terminal":           {"Windows": "wt",                      "Darwin": "Terminal",             "Linux": "gnome-terminal"},
    "cmd":                {"Windows": "cmd.exe",                 "Darwin": "Terminal",             "Linux": "bash"},
    "powershell":         {"Windows": "powershell.exe",          "Darwin": "Terminal",             "Linux": "bash"},
    "postman":            {"Windows": "Postman",                 "Darwin": "Postman",              "Linux": "postman"},
    "git":                {"Windows": "git-bash",                "Darwin": "Terminal",             "Linux": "bash"},
    "figma":              {"Windows": "Figma",                   "Darwin": "Figma",                "Linux": "figma"},
    "blender":            {"Windows": "blender",                 "Darwin": "Blender",              "Linux": "blender"},
    "word":               {"Windows": "winword",                 "Darwin": "Microsoft Word",       "Linux": "libreoffice --writer"},
    "excel":              {"Windows": "excel",                   "Darwin": "Microsoft Excel",      "Linux": "libreoffice --calc"},
    "powerpoint":         {"Windows": "powerpnt",                "Darwin": "Microsoft PowerPoint", "Linux": "libreoffice --impress"},
    "libreoffice":        {"Windows": "soffice",                 "Darwin": "LibreOffice",          "Linux": "libreoffice"},
    "notepad":            {"Windows": "notepad.exe",             "Darwin": "TextEdit",             "Linux": "gedit"},
    "textedit":           {"Windows": "notepad.exe",             "Darwin": "TextEdit",             "Linux": "gedit"},
    "explorer":           {"Windows": "explorer.exe",            "Darwin": "Finder",               "Linux": "nautilus"},
    "file explorer":      {"Windows": "explorer.exe",            "Darwin": "Finder",               "Linux": "nautilus"},
    "finder":             {"Windows": "explorer.exe",            "Darwin": "Finder",               "Linux": "nautilus"},
    "task manager":       {"Windows": "taskmgr.exe",             "Darwin": "Activity Monitor",     "Linux": "gnome-system-monitor"},
    "settings":           {"Windows": "ms-settings:",            "Darwin": "System Preferences",   "Linux": "gnome-control-center"},
    "calculator":         {"Windows": "calc.exe",                "Darwin": "Calculator",           "Linux": "gnome-calculator"},
    "paint":              {"Windows": "mspaint.exe",             "Darwin": "Preview",              "Linux": "gimp"},
    "instagram":          {"Windows": "Instagram",               "Darwin": "Instagram",            "Linux": "firefox"},
    "tiktok":             {"Windows": "TikTok",                  "Darwin": "TikTok",               "Linux": "firefox"},
    "notion":             {"Windows": "Notion",                  "Darwin": "Notion",               "Linux": "notion"},
    "obsidian":           {"Windows": "Obsidian",                "Darwin": "Obsidian",             "Linux": "obsidian"},
    "capcut":             {"Windows": "CapCut",                  "Darwin": "CapCut",               "Linux": "capcut"},
    "steam":              {"Windows": "steam",                   "Darwin": "Steam",                "Linux": "steam"},
    "epic":               {"Windows": "EpicGamesLauncher",       "Darwin": "Epic Games Launcher",  "Linux": "legendary"},
    "epic games":         {"Windows": "EpicGamesLauncher",       "Darwin": "Epic Games Launcher",  "Linux": "legendary"},
}


def _load_app_paths() -> dict:
    p = BASE_DIR / "config" / "app_paths.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {k.lower().strip(): v for k, v in data.items()}
    except Exception as e:
        _log.warning("Could not load config/app_paths.json: %s", e)
    return {}


def _normalize(raw: str) -> str | dict:
    key = raw.lower().strip()

    # 1. Custom app_paths.json lookup
    custom_paths = _load_app_paths()
    if key in custom_paths:
        return custom_paths[key]
    for ck, cv in custom_paths.items():
        if ck in key or key in ck:
            return cv

    # 2. Built-in alias map
    if key in _APP_ALIASES:
        return _APP_ALIASES[key].get(_SYSTEM, raw)

    for alias_key, os_map in _APP_ALIASES.items():
        if alias_key in key or key in alias_key:
            return os_map.get(_SYSTEM, raw)

    return raw


def _probe_windows_exe(app_name: str) -> str | None:
    """Probes well-known Windows installation directories for common applications."""
    key = app_name.lower().strip()

    known_probes = {
        "chrome": [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        ],
        "google chrome": [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        ],
        "firefox": [
            r"C:\Program Files\Mozilla Firefox\firefox.exe",
            r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe",
        ],
        "edge": [
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        ],
        "msedge": [
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        ],
        "brave": [
            r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\BraveSoftware\Brave-Browser\Application\brave.exe"),
        ],
        "spotify": [
            os.path.expandvars(r"%APPDATA%\Spotify\Spotify.exe"),
            os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WindowsApps\Spotify.exe"),
        ],
        "discord": [
            os.path.expandvars(r"%LOCALAPPDATA%\Discord\Update.exe"),
        ],
        "telegram": [
            os.path.expandvars(r"%APPDATA%\Telegram Desktop\Telegram.exe"),
            r"C:\Program Files\Telegram Desktop\Telegram.exe",
        ],
        "vscode": [
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe"),
            r"C:\Program Files\Microsoft VS Code\Code.exe",
        ],
        "visual studio code": [
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe"),
            r"C:\Program Files\Microsoft VS Code\Code.exe",
        ],
        "code": [
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe"),
            r"C:\Program Files\Microsoft VS Code\Code.exe",
        ],
        "notepad": [
            r"C:\Windows\notepad.exe",
            r"C:\Windows\System32\notepad.exe",
        ],
        "notepad.exe": [
            r"C:\Windows\notepad.exe",
            r"C:\Windows\System32\notepad.exe",
        ],
        "explorer": [
            r"C:\Windows\explorer.exe",
        ],
        "task manager": [
            r"C:\Windows\System32\taskmgr.exe",
        ],
        "calc": [
            r"C:\Windows\System32\calc.exe",
        ],
        "calculator": [
            r"C:\Windows\System32\calc.exe",
        ],
        "paint": [
            r"C:\Windows\System32\mspaint.exe",
        ],
    }

    if key in known_probes:
        for candidate in known_probes[key]:
            if os.path.isfile(candidate):
                return candidate

    # Generic probing in Programs directories
    base_dirs = [
        os.path.expandvars(r"%LOCALAPPDATA%\Programs"),
        os.environ.get("ProgramFiles", r"C:\Program Files"),
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
    ]
    app_base = key.replace(".exe", "")
    for b in base_dirs:
        if not b or not os.path.isdir(b):
            continue
        direct_exe = os.path.join(b, f"{app_base}.exe")
        if os.path.isfile(direct_exe):
            return direct_exe
        nested_exe = os.path.join(b, app_base, f"{app_base}.exe")
        if os.path.isfile(nested_exe):
            return nested_exe

    return None


def _launch_windows(target: str | dict, raw_app_name: str = "") -> bool:
    # 1. Structure configuration (e.g. UWP protocol/AUMID from app_paths.json)
    if isinstance(target, dict):
        app_type = target.get("type", "path").lower()
        if app_type == "uwp":
            protocol = target.get("protocol")
            if protocol:
                try:
                    subprocess.Popen(["cmd", "/c", "start", "", protocol])
                    return True
                except Exception as e:
                    _log.debug(f"Protocol launch failed for {protocol}: {e}")
            aumid = target.get("aumid")
            if aumid:
                try:
                    subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{aumid}"])
                    return True
                except Exception as e:
                    _log.debug(f"AUMID launch failed for {aumid}: {e}")
            return False

        elif app_type == "path":
            path_val = target.get("path", "")
            if path_val and os.path.isfile(path_val):
                try:
                    subprocess.Popen([path_val], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return True
                except Exception as e:
                    _log.warning(f"Config path launch failed for {path_val}: {e}")

    # 2. String target resolution
    if isinstance(target, str):
        # Direct file path
        if os.path.isfile(target):
            try:
                subprocess.Popen([target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except Exception as e:
                _log.warning(f"File path launch failed for {target}: {e}")

        # Protocol URI (e.g. "whatsapp:", "ms-settings:")
        if target.endswith(":") or (":" in target and "\\" not in target and "/" not in target):
            try:
                subprocess.Popen(["cmd", "/c", "start", "", target])
                return True
            except Exception:
                pass

        # Standard installation directory probing
        probed = _probe_windows_exe(target) or (raw_app_name and _probe_windows_exe(raw_app_name))
        if probed:
            try:
                subprocess.Popen([probed], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except Exception as e:
                _log.warning(f"Probed path launch failed for {probed}: {e}")

        # PATH resolution via shutil.which
        resolved = shutil.which(target) or shutil.which(target.split(".")[0])
        if not resolved and raw_app_name:
            resolved = shutil.which(raw_app_name) or shutil.which(raw_app_name.split(".")[0])
        if resolved:
            try:
                subprocess.Popen([resolved], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except Exception as e:
                _log.warning(f"PATH launch failed for {resolved}: {e}")

    # 3. Fallback to Start Menu search via PyAutoGUI
    search_term = raw_app_name or (target if isinstance(target, str) else "")
    if search_term and not str(search_term).endswith(":"):
        try:
            import pyautogui
            pyautogui.PAUSE = 0.1
            pyautogui.press("win")
            time.sleep(0.7)
            pyautogui.write(search_term, interval=0.05)
            time.sleep(0.9)
            pyautogui.press("enter")
            time.sleep(1.2)
            return True
        except Exception as e:
            _log.warning(f"Start Menu search failed: {e}")

    return False


def _launch_macos(app_name: str) -> bool:
    target = app_name if isinstance(app_name, str) else str(app_name)
    try:
        result = subprocess.run(
            ["open", "-a", target],
            capture_output=True, timeout=8
        )
        if result.returncode == 0:
            time.sleep(1.0)
            return True
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["open", "-a", f"{target}.app"],
            capture_output=True, timeout=8
        )
        if result.returncode == 0:
            time.sleep(1.0)
            return True
    except Exception:
        pass

    binary = shutil.which(target) or shutil.which(target.lower())
    if binary:
        try:
            subprocess.Popen(
                [binary],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(1.0)
            return True
        except Exception:
            pass

    try:
        import pyautogui
        pyautogui.hotkey("command", "space")
        time.sleep(0.3)
        pyautogui.write(target, interval=0.05)
        time.sleep(0.8)
        pyautogui.press("enter")
        time.sleep(0.8)
        return True
    except Exception as e:
        _log.warning(f"Spotlight failed: {e}")

    return False


def _launch_linux(app_name: str) -> bool:
    target = app_name if isinstance(app_name, str) else str(app_name)
    binary = (
        shutil.which(target) or
        shutil.which(target.lower()) or
        shutil.which(target.lower().replace(" ", "-")) or
        shutil.which(target.lower().replace(" ", "_"))
    )
    if binary:
        try:
            subprocess.Popen(
                [binary],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(1.0)
            return True
        except Exception:
            pass

    try:
        subprocess.run(
            ["xdg-open", target],
            capture_output=True, timeout=5
        )
        return True
    except Exception:
        pass

    for desktop_name in [
        target.lower(),
        target.lower().replace(" ", "-"),
        target.lower().replace(" ", ""),
    ]:
        try:
            result = subprocess.run(
                ["gtk-launch", desktop_name],
                capture_output=True, timeout=5
            )
            if result.returncode == 0:
                return True
        except Exception:
            pass

    return False


def _extract_app_name_from_text(text: str) -> str:
    """
    Extract an application name from natural language query/description.
    e.g. 'Open up WhatsApp', 'launch Google Chrome', 'pull up Spotify', 'start notepad'.
    """
    if not text or not isinstance(text, str):
        return ""
    cleaned = text.strip()
    patterns = [
        r"^(?:please\s+)?(?:open\s+up|open|launch|start|run|pull\s+up|bring\s+up)\s+(?:the\s+)?([A-Za-z0-9\s\-._+]+?)(?:\s+(?:app|application|program|window|browser|please|now)|[?.!,;]|$)",
        r"\b(?:open\s+up|open|launch|start|run|pull\s+up|bring\s+up)\s+(?:the\s+)?([A-Za-z0-9\s\-._+]+?)(?:\s+(?:app|application|program|window|browser|please|now)|[?.!,;]|$)",
    ]
    for pat in patterns:
        m = re.search(pat, cleaned, re.IGNORECASE)
        if m:
            cand = m.group(1).strip()
            if cand.lower() not in ("it", "this", "that", "something", "anything", "app", "application", "program"):
                return cand
    return cleaned


_OS_LAUNCHERS = {
    "Windows": _launch_windows,
    "Darwin":  _launch_macos,
    "Linux":   _launch_linux,
}

def open_app(
    parameters=None,
    response=None,
    player=None,
    session_memory=None,
) -> str:
    params = parameters or {}
    app_name = params.get("app_name", "").strip()

    if not app_name:
        for field in ("query", "description", "goal", "prompt", "text", "name"):
            val = params.get(field)
            if val and isinstance(val, str):
                extracted = _extract_app_name_from_text(val) or val.strip()
                if extracted:
                    app_name = extracted
                    break

    if not app_name:
        return "No application name provided."

    launcher = _OS_LAUNCHERS.get(_SYSTEM)
    if launcher is None:
        return f"Unsupported operating system: {_SYSTEM}"

    normalized = _normalize(app_name)
    _log.debug(f"Launching: '{app_name}' → '{normalized}' ({_SYSTEM})")

    if player:
        player.write_log(f"[open_app] {app_name}")

    try:
        if _SYSTEM == "Windows":
            if _launch_windows(normalized, raw_app_name=app_name):
                return f"Opened {app_name}."
            if isinstance(normalized, str) and normalized.lower() != app_name.lower():
                if _launch_windows(app_name, raw_app_name=app_name):
                    return f"Opened {app_name}."
        else:
            if launcher(normalized if isinstance(normalized, str) else app_name):
                return f"Opened {app_name}."
            if isinstance(normalized, str) and normalized.lower() != app_name.lower():
                if launcher(app_name):
                    return f"Opened {app_name}."
        return (
            f"Could not confirm that {app_name} launched. "
            f"It may still be loading, or it might not be installed."
        )
    except Exception as e:
        _log.warning(f"Error: {e}")
        return f"Failed to open {app_name}: {e}"