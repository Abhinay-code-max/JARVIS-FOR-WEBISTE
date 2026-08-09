import json
import subprocess
import time
from pathlib import Path

from config import BASE_DIR, get_os
import logging

_log = logging.getLogger("jarvis.send_message")

try:
    import pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE    = 0.06
    _PYAUTOGUI = True
except ImportError:
    _PYAUTOGUI = False

try:
    import pyperclip
    _PYPERCLIP = True
except ImportError:
    _PYPERCLIP = False


def _load_contacts() -> dict:
    """Load contacts from config/contacts.json — {alias: exact_whatsapp_name}"""
    try:
        p = BASE_DIR / "config" / "contacts.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _resolve_contact(name: str) -> str:
    """
    Fuzzy-match the spoken name against contacts.json.
    Returns the exact WhatsApp contact name, or the original if no match.
    """
    contacts = _load_contacts()
    if not contacts:
        return name

    name_lower = name.lower().strip()

    # Exact match first
    if name_lower in contacts:
        return contacts[name_lower]

    # Partial match — spoken name contained in key or key contained in spoken name
    for alias, exact in contacts.items():
        if alias in name_lower or name_lower in alias:
            _log.debug(f"Matched '{name}' → '{exact}'")
            return exact

    # Character-level similarity (simple)
    best_match = None
    best_score = 0
    for alias, exact in contacts.items():
        score = _similarity(name_lower, alias)
        if score > best_score:
            best_score = score
            best_match = exact

    if best_score > 0.6 and best_match:
        _log.debug(f"Fuzzy matched '{name}' → '{best_match}' ({best_score:.0%})")
        return best_match

    return name  # fallback: use as-is


def _similarity(a: str, b: str) -> float:
    """Simple character overlap similarity."""
    if not a or not b:
        return 0.0
    set_a = set(a)
    set_b = set(b)
    return len(set_a & set_b) / len(set_a | set_b)


def _require_pyautogui():
    if not _PYAUTOGUI:
        raise RuntimeError("PyAutoGUI not installed. Run: pip install pyautogui")


def _paste_text(text: str) -> None:
    _require_pyautogui()
    os_name = get_os()
    paste_hotkey = ("command", "v") if os_name == "mac" else ("ctrl", "v")
    if _PYPERCLIP:
        pyperclip.copy(text)
        time.sleep(0.15)
        pyautogui.hotkey(*paste_hotkey)
        time.sleep(0.1)
    else:
        pyautogui.write(text, interval=0.03)


def _clear_and_paste(text: str) -> None:
    _require_pyautogui()
    os_name = get_os()
    select_all = ("command", "a") if os_name == "mac" else ("ctrl", "a")
    pyautogui.hotkey(*select_all)
    time.sleep(0.1)
    pyautogui.press("delete")
    time.sleep(0.1)
    _paste_text(text)


def _open_whatsapp_windows() -> bool:
    """Open WhatsApp Desktop on Windows using the whatsapp: protocol."""
    try:
        subprocess.Popen(
            ["powershell", "-command", "Start-Process 'whatsapp:'"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1.2)  # give WhatsApp time to open and load
        return True
    except Exception as e:
        _log.warning(f"Could not open WhatsApp: {e}")
        return False


def _send_whatsapp(receiver: str, message: str) -> str:
    _require_pyautogui()

    # Resolve contact name from contacts.json
    exact_name = _resolve_contact(receiver)
    _log.debug(f"WhatsApp → resolved '{receiver}' to '{exact_name}'")

    os_name = get_os()

    # Open WhatsApp
    if os_name == "windows":
        if not _open_whatsapp_windows():
            return "Could not open WhatsApp."
    elif os_name == "mac":
        subprocess.run(["open", "-a", "WhatsApp"], timeout=10)
        time.sleep(0.4)
    else:
        subprocess.Popen(["whatsapp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.4)

    # Use Ctrl+N or click the New Chat / Search button
    # WhatsApp Desktop: Ctrl+N opens new chat search
    time.sleep(0.6)
    pyautogui.hotkey("ctrl", "n")
    time.sleep(0.4)

    # Type the contact name in the search box
    _paste_text(exact_name)
    time.sleep(0.6)  # wait for search results

    # Press Down arrow to select first result, then Enter to open chat
    pyautogui.press("down")
    time.sleep(0.2)
    pyautogui.press("enter")
    time.sleep(0.4)

    # Type and send the message
    _paste_text(message)
    time.sleep(0.2)
    pyautogui.press("enter")
    time.sleep(0.2)

    return f"Message sent to {exact_name} via WhatsApp."


def _open_app(app_name: str) -> bool:
    _require_pyautogui()
    os_name = get_os()
    try:
        if os_name == "windows":
            pyautogui.press("win")
            time.sleep(0.2)
            _paste_text(app_name)
            time.sleep(0.6)
            pyautogui.press("enter")
            time.sleep(1.2)
            return True
        elif os_name == "mac":
            result = subprocess.run(
                ["open", "-a", app_name],
                capture_output=True, text=True, timeout=10,
            )
            time.sleep(1.2)
            return result.returncode == 0
        else:
            subprocess.Popen(
                [app_name.lower()],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(1.2)
            return True
    except Exception as e:
        _log.warning(f"Could not open {app_name}: {e}")
        return False


def _open_browser_url(url: str) -> bool:
    import webbrowser
    try:
        webbrowser.open(url)
        time.sleep(1.2)
        return True
    except Exception as e:
        _log.warning(f"Could not open browser: {e}")
        return False


def _desktop_send(app_name: str, receiver: str, message: str) -> str:
    exact_name = _resolve_contact(receiver)
    if not _open_app(app_name):
        return f"Could not open {app_name}."
    time.sleep(0.6)
    # Use Ctrl+F for search in generic apps
    pyautogui.hotkey("ctrl", "f")
    time.sleep(0.2)
    _clear_and_paste(exact_name)
    time.sleep(0.6)
    pyautogui.press("enter")
    time.sleep(0.4)
    _paste_text(message)
    time.sleep(0.2)
    pyautogui.press("enter")
    time.sleep(0.2)
    return f"Message sent to {exact_name} via {app_name}."


def _send_telegram(receiver: str, message: str) -> str:
    return _desktop_send("Telegram", receiver, message)


def _send_signal(receiver: str, message: str) -> str:
    return _desktop_send("Signal", receiver, message)


def _send_discord(receiver: str, message: str) -> str:
    return _desktop_send("Discord", receiver, message)


def _send_instagram(receiver: str, message: str) -> str:
    _require_pyautogui()
    exact_name = _resolve_contact(receiver)
    if not _open_browser_url("https://www.instagram.com/direct/new/"):
        return "Could not open Instagram in browser."
    _paste_text(exact_name)
    time.sleep(0.4)
    pyautogui.press("down")
    time.sleep(0.2)
    pyautogui.press("enter")
    time.sleep(0.4)
    for _ in range(4):
        pyautogui.press("tab")
        time.sleep(0.15)
    pyautogui.press("enter")
    time.sleep(0.6)
    _paste_text(message)
    time.sleep(0.2)
    pyautogui.press("enter")
    time.sleep(0.2)
    return f"Message sent to {exact_name} via Instagram."


def _send_messenger(receiver: str, message: str) -> str:
    _require_pyautogui()
    exact_name = _resolve_contact(receiver)
    if not _open_browser_url("https://www.messenger.com/"):
        return "Could not open Messenger in browser."
    pyautogui.hotkey("ctrl", "f")
    time.sleep(0.2)
    _paste_text(exact_name)
    time.sleep(0.2)
    pyautogui.press("down")
    time.sleep(0.2)
    pyautogui.press("enter")
    time.sleep(0.6)
    _paste_text(message)
    time.sleep(0.2)
    pyautogui.press("enter")
    time.sleep(0.2)
    return f"Message sent to {exact_name} via Messenger."


_PLATFORM_MAP = [
    ({"whatsapp", "wp", "wapp"},          _send_whatsapp),
    ({"telegram", "tg"},                   _send_telegram),
    ({"instagram", "ig", "insta"},         _send_instagram),
    ({"signal"},                            _send_signal),
    ({"discord"},                           _send_discord),
    ({"messenger", "facebook", "fb"},      _send_messenger),
]


def _resolve_platform(platform_str: str):
    key = platform_str.lower().strip()
    for keywords, handler in _PLATFORM_MAP:
        if any(k in key for k in keywords):
            return handler
    return lambda r, m: _desktop_send(platform_str.strip().title(), r, m)


def send_message(
    parameters: dict,
    response=None,
    player=None,
    session_memory=None,
) -> str:
    params       = parameters or {}
    receiver     = params.get("receiver", "").strip()
    message_text = params.get("message_text", "").strip()
    platform     = params.get("platform", "whatsapp").strip()

    if not receiver:
        return "Please specify a recipient."
    if not message_text:
        return "Please specify the message content."
    if not _PYAUTOGUI:
        return "PyAutoGUI is not installed — cannot control the desktop."

    preview = message_text[:50] + ("…" if len(message_text) > 50 else "")
    _log.debug(f"📨 {platform} → {receiver}: {preview}")
    if player:
        player.write_log(f"SYS: sending message to {receiver} via {platform}")

    try:
        handler = _resolve_platform(platform)
        result  = handler(receiver, message_text)
    except Exception as e:
        result = f"Could not send message: {e}"

    if "sent" in result.lower():
        _log.info(result)
    else:
        _log.warning(result)
    if player:
        player.write_log(f"[msg] {result}")

    return result