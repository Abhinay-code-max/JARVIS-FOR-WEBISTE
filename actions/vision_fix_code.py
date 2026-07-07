"""
actions/vision_fix_code.py
============================
Simple flow:
  1. Take a screenshot
  2. Ask vision model to identify: what file is open, what the bug is
  3. Find that file on disk (checks common locations including OneDrive-redirected folders)
  4. Read the actual file content
  5. Ask LLM to fix it based on what's visible
  6. Write the fix directly to the file
  7. Tell the user what was fixed — no questions asked, no back-and-forth
"""

from __future__ import annotations
import base64
import io
import json
import re
from pathlib import Path
from typing import Optional

import requests

try:
    import mss
    import mss.tools
    _MSS = True
except ImportError:
    _MSS = False

try:
    import PIL.Image
    _PIL = True
except ImportError:
    _PIL = False

try:
    import pygetwindow as gw
    _PYGETWINDOW = True
except ImportError:
    _PYGETWINDOW = False


from config import load_config as _load_config, BASE_DIR


def _capture_screen() -> bytes:
    if not _MSS:
        raise RuntimeError("mss not installed — cannot capture screen.")
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        sct_img = sct.grab(monitor)
        if _PIL:
            img = PIL.Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            img.thumbnail((1280, 720))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            return buf.getvalue()
        else:
            return mss.tools.to_png(sct_img.rgb, sct_img.size)


def _get_active_window_title() -> str:
    if not _PYGETWINDOW:
        return ""
    try:
        win = gw.getActiveWindow()
        if win:
            return win.title or ""
    except Exception:
        pass
    return ""


def _get_search_roots() -> list[str]:
    """
    Builds a list of likely locations, automatically detecting OneDrive-redirected
    folders (very common on Windows — Desktop/Documents/Downloads often live under
    OneDrive instead of directly under the user folder).
    """
    home = Path.home()
    roots = []

    # Standard locations
    for folder in ("Desktop", "Documents", "Downloads"):
        roots.append(str(home / folder))

    # OneDrive-redirected locations (very common)
    onedrive_candidates = [
        home / "OneDrive",
        home / "OneDrive - Personal",
    ]
    for od in onedrive_candidates:
        if od.exists():
            for folder in ("Desktop", "Documents", "Downloads"):
                roots.append(str(od / folder))

    # Also check for any OneDrive* folder directly under home
    try:
        for item in home.iterdir():
            if item.is_dir() and item.name.startswith("OneDrive"):
                for folder in ("Desktop", "Documents", "Downloads"):
                    candidate = item / folder
                    if candidate.exists():
                        roots.append(str(candidate))
    except Exception:
        pass

    # JARVIS project dir itself
    roots.append(str(BASE_DIR))

    # De-duplicate while preserving order
    seen = set()
    unique_roots = []
    for r in roots:
        if r not in seen:
            seen.add(r)
            unique_roots.append(r)

    return unique_roots


def _find_file_on_disk(hint_filename: str, search_roots: list[str]) -> Optional[Path]:
    if not hint_filename:
        return None

    hint_filename = hint_filename.strip()

    for root in search_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        candidate = root_path / hint_filename
        if candidate.exists():
            return candidate
        try:
            for p in root_path.rglob(hint_filename):
                if any(skip in str(p) for skip in ("node_modules", ".git", "__pycache__", "venv", ".backup")):
                    continue
                return p
        except Exception:
            continue

    return None


def _extract_filename_from_title(title: str) -> str:
    if not title:
        return ""
    m = re.search(r'([a-zA-Z0-9_\-\.]+\.(py|js|ts|java|cpp|c|cs|go|rb|php|html|css))', title)
    if m:
        return m.group(1)
    return ""


def _call_vision_model(image_bytes: bytes, prompt: str, config: dict) -> str:
    url   = config.get("llm_url", "http://localhost:11434").rstrip("/")
    model = config.get("vision_model") or "qwen2.5vl:7b"

    b64 = base64.b64encode(image_bytes).decode("ascii")
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [b64],
            }
        ],
    }
    resp = requests.post(f"{url}/api/chat", json=payload, timeout=60)
    resp.raise_for_status()
    return (resp.json().get("message", {}).get("content") or "").strip()


def _call_llm_to_fix(file_content: str, bug_description: str, config: dict) -> Optional[str]:
    url   = config.get("llm_url", "http://localhost:11434").rstrip("/")
    model = config.get("llm_model", "qwen2.5:3b")

    prompt = f"""Fix this Python file based on the identified bug.

BUG DESCRIPTION: {bug_description}

CURRENT FILE:
```python
{file_content}
```

Output ONLY the complete corrected file content. No markdown fences, no explanation — just the raw fixed source code."""

    try:
        payload = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": "You fix code bugs. Output only raw corrected code, nothing else."},
                {"role": "user", "content": prompt},
            ],
        }
        resp = requests.post(f"{url}/api/chat", json=payload, timeout=60)
        resp.raise_for_status()
        fixed = resp.json().get("message", {}).get("content", "").strip()
        fixed = re.sub(r"^```(?:python)?\n", "", fixed)
        fixed = re.sub(r"\n```$", "", fixed)
        return fixed.strip()
    except Exception as e:
        print(f"[VisionFixCode] LLM fix error: {e}")
        return None


def vision_fix_code(parameters: dict = None, player=None, speak=None) -> str:
    parameters = parameters or {}
    config = _load_config()

    def _log(msg: str):
        print(f"[VisionFixCode] {msg}")
        if player:
            try:
                player.write_log(f"[fix] {msg}")
            except Exception:
                pass

    try:
        _log("Capturing screen...")
        image_bytes = _capture_screen()
    except Exception as e:
        return f"Could not capture the screen: {e}"

    _log("Analyzing code on screen...")
    vision_prompt = (
        "Look at this screenshot of code in an editor. Carefully read EVERY line, "
        "top to bottom, before answering. Pay special attention to: "
        "loop ranges and boundaries (off-by-one errors like range(len(x)+1)), "
        "list/array indexing that might go out of bounds, "
        "uninitialized or undefined variables, "
        "incorrect comparisons or logic. "
        "Trace through the code mentally with the sample data shown (if visible) "
        "to find which line would actually crash or misbehave first. "
        "Identify: 1) the exact filename shown in the tab or breadcrumb, "
        "2) the SINGLE most likely bug that would cause a runtime error, "
        "including the specific line number. "
        "Respond in this exact format:\n"
        "FILENAME: <filename or 'unknown'>\n"
        "LINE: <line number or 'unknown'>\n"
        "BUG: <one precise sentence describing the exact bug and why it fails, or 'none found'>"
    )
    try:
        vision_result = _call_vision_model(image_bytes, vision_prompt, config)
    except Exception as e:
        return f"Vision analysis failed: {e}"

    _log(f"Vision result: {vision_result[:250]}")

    filename_match = re.search(r"FILENAME:\s*(.+)", vision_result)
    bug_match      = re.search(r"BUG:\s*(.+)", vision_result)

    filename = filename_match.group(1).strip() if filename_match else ""
    bug_desc = bug_match.group(1).strip() if bug_match else ""

    if filename.lower() in ("unknown", "none", "n/a", ""):
        title = _get_active_window_title()
        filename = _extract_filename_from_title(title)
        _log(f"Vision couldn't read filename, trying window title: '{title}' -> '{filename}'")

    if not filename:
        return (
            "I can see code on your screen but couldn't determine the filename. "
            "Could you tell me the file path, or make sure the filename is visible "
            "in the editor tab or window title?"
        )

    if bug_desc.lower() in ("none found", "none", "n/a", ""):
        return f"I looked at {filename} and didn't spot any obvious bugs. The code looks fine to me."

    search_roots = _get_search_roots()
    _log(f"Searching for '{filename}' across {len(search_roots)} locations (including OneDrive)...")
    fpath = _find_file_on_disk(filename, search_roots)

    if not fpath:
        return (
            f"I identified the bug in {filename} ({bug_desc}), but couldn't locate "
            f"the file on disk to fix it automatically. Could you tell me its full path?"
        )

    _log(f"Found file: {fpath}")

    try:
        file_content = fpath.read_text(encoding="utf-8")
    except Exception as e:
        return f"Found {fpath.name} but couldn't read it: {e}"

    _log("Generating fix...")
    fixed_code = _call_llm_to_fix(file_content, bug_desc, config)

    if not fixed_code:
        return f"I identified the bug in {fpath.name}: {bug_desc}. But I couldn't generate a fix automatically."

    try:
        backup_path = fpath.with_suffix(fpath.suffix + ".backup")
        backup_path.write_text(file_content, encoding="utf-8")
        fpath.write_text(fixed_code, encoding="utf-8")
        _log(f"Fixed and saved {fpath.name} (backup: {backup_path.name})")
    except Exception as e:
        return f"Found the bug ({bug_desc}) but couldn't write the fix to disk: {e}"

    return f"Fixed it. The bug was: {bug_desc}. I've corrected {fpath.name} directly — a backup of the original is saved as {fpath.name}.backup."
