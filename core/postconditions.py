"""
core/postconditions.py
=========================
Registry of postcondition checkers for tools/actions with a real,
independently-checkable claim about the outside world — did the file
file_controller claimed to write/delete/move actually change, did the OS
scheduler actually register the task reminder claimed to set, etc. Wired
into core/tool_gate.py's dispatch_tool(), run once after a tool call
returns (only on paths that actually invoked the tool — nothing to check
on a rejected/denied/timed-out/hard-denied call).

Deliberately narrow — see the verification-phase investigation. Most
tools have no checkable postcondition at all (weather_report, web_search,
screen_process, ...) and forcing a fake one onto them would produce false
confidence, which is worse than no check. Only the 7 tools below get one;
everything else is intentionally absent from POSTCONDITION_CHECKS, not an
oversight.

Each checker: (args: dict, result: str) -> str | None. None means the
postcondition holds, OR doesn't apply to this specific action/args (e.g.
file_controller's `list` action, or a param combination the tool itself
already refuses before doing anything). A non-None string describes what
was expected but wasn't found. check_postcondition() never lets a
checker's own bug (import error, unexpected shape) masquerade as a real
postcondition failure — it's caught and logged as "couldn't verify", not
"verified false".
"""
from __future__ import annotations

import logging
import subprocess
from datetime import datetime
from pathlib import Path

_log = logging.getLogger("jarvis.postconditions")


# ---------------------------------------------------------------------------
# file_controller — filesystem existence at the resolved target(s)
# ---------------------------------------------------------------------------

def _check_file_controller(args: dict, result: str) -> str | None:
    from actions.file_controller import _resolve_path

    action = (args.get("action") or "").strip().lower()
    name   = args.get("name", "")

    if action in ("write", "create_file", "create_folder"):
        base   = _resolve_path(args.get("path", "desktop"))
        target = (base / name) if name else base
        return None if target.exists() else f"'{target}' does not exist after {action}."

    if action == "delete":
        base   = _resolve_path(args.get("path", "desktop"))
        target = (base / name) if name else base
        return None if not target.exists() else f"'{target}' still exists after delete."

    if action in ("move", "copy"):
        base     = _resolve_path(args.get("path", "desktop"))
        src      = (base / name) if name else base
        dst_base = _resolve_path(args.get("destination", ""))
        if not args.get("destination"):
            return None  # tool itself already refuses this case
        # Best-effort: the destination may have been an existing directory
        # (file_controller lands the file inside it, named after the
        # source) or an exact target path — which one applied isn't
        # knowable here after the fact, so both candidates are checked.
        candidates = (dst_base / src.name, dst_base)
        if not any(c.exists() for c in candidates):
            return f"Neither '{candidates[0]}' nor '{candidates[1]}' exists after {action}."
        if action == "move" and src.exists():
            return f"Source '{src}' still exists after move."
        return None

    if action == "rename":
        new_name = args.get("new_name", "")
        if not new_name:
            return None  # tool itself already refuses this case
        base     = _resolve_path(args.get("path", "desktop"))
        target   = (base / name) if name else base
        new_path = target.parent / new_name
        return None if new_path.exists() else f"'{new_path}' does not exist after rename."

    return None  # list/read/find/largest/disk_usage/info — nothing changed, nothing to verify


# ---------------------------------------------------------------------------
# file_processor — output file exists, for the specific actions that
# produce one (file_processor.py has many more file-producing actions
# than covered here — see the investigation; only the ones explicitly
# scoped in are checked, not the whole surface)
# ---------------------------------------------------------------------------

def _fp_output_path(src: Path, suffix: str, new_ext: str | None = None) -> Path:
    """Mirrors actions/file_processor.py's own _output_path() naming
    convention exactly. Reimplemented rather than imported — that helper
    isn't part of the module's public surface — but it's a single,
    stable, three-line formula, not the kind of thing likely to drift
    silently out from under this."""
    ext  = new_ext or src.suffix
    name = f"{src.stem}_{suffix}{ext}"
    return src.parent / name


def _check_file_processor(args: dict, result: str) -> str | None:
    action    = (args.get("action") or "").strip().lower()
    file_path = args.get("file_path", "")
    if not file_path:
        return None
    path = Path(file_path)

    if action == "extract_text" and path.suffix.lower() == ".pptx":
        out = _fp_output_path(path, "text", ".txt")
        return None if out.exists() else f"'{out}' does not exist after extract_text."

    if action == "extract" and path.suffix.lower() in (".zip", ".tar", ".gz", ".bz2", ".xz"):
        dest = Path(args.get("destination", str(path.parent / path.stem)))
        # dest.mkdir() runs unconditionally before unpack_archive() is even
        # attempted, so mere existence proves nothing — only non-empty is
        # meaningful, and even that can't distinguish this run's contents
        # from a previous run's leftovers if `dest` was already populated.
        if not dest.exists() or not any(dest.iterdir()):
            return f"'{dest}' is empty after archive extract."
        return None

    if path.suffix.lower() not in (".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".m4v"):
        return None  # not a video — no other action is covered

    video_checks = {
        "extract_audio": lambda: _fp_output_path(path, "audio", ".mp3"),
        "compress":      lambda: _fp_output_path(
            path, f"compressed_crf{int(args.get('quality', 28))}", ".mp4"
        ),
        "extract_frame": lambda: _fp_output_path(
            path, f"frame_{str(args.get('timestamp', '00:00:01')).replace(':', '')}", ".jpg"
        ),
        "convert": lambda: _fp_output_path(
            path, "converted", f".{str(args.get('format', 'mp4')).lstrip('.')}"
        ),
    }
    checker = video_checks.get(action)
    if checker is None:
        return None  # info/trim/transcribe — not covered

    out = checker()
    return None if out.exists() else f"'{out}' does not exist after {action}."


# ---------------------------------------------------------------------------
# code_helper — target file exists, for the actions that save one
# ---------------------------------------------------------------------------

def _check_code_helper(args: dict, result: str) -> str | None:
    from actions.code_helper import _resolve_save_path

    action = (args.get("action") or "").strip().lower()
    if action not in ("write", "edit", "optimize"):
        return None  # run/build already self-verify internally via _has_error();
                      # explain/screen_debug/auto have no file side effect

    file_path = args.get("file_path", "")
    if action == "edit":
        if not file_path:
            return None  # tool itself already refuses this case
        target = Path(file_path)
    elif action == "optimize" and file_path:
        target = Path(file_path)
    else:  # write, or optimize with no file_path given
        target = _resolve_save_path(args.get("output_path", ""), args.get("language", "python"))

    return None if target.exists() else f"'{target}' does not exist after {action}."


# ---------------------------------------------------------------------------
# dev_agent — best-effort only. Weaker signal than the checks above: only
# checkable when project_name was explicitly given: the actual folder
# name otherwise comes from dev_agent's own internal planning LLM call
# and isn't knowable here without re-parsing its return string, which
# isn't attempted. No opinion (None), not a false failure, when it wasn't.
# ---------------------------------------------------------------------------

def _check_dev_agent(args: dict, result: str) -> str | None:
    import re as _re
    from actions.dev_agent import PROJECTS_DIR

    project_name = (args.get("project_name") or "").strip()
    if not project_name:
        return None

    proj_dir = PROJECTS_DIR / _re.sub(r"[^\w\-]", "_", project_name)
    if not proj_dir.exists() or not any(proj_dir.iterdir()):
        return f"'{proj_dir}' does not exist (or is empty) after dev_agent."
    return None


# ---------------------------------------------------------------------------
# reminder / game_updater (schedule action) — OS scheduler re-query
# ---------------------------------------------------------------------------

def _scheduled_task_exists_windows(task_name: str) -> bool:
    try:
        result = subprocess.run(
            ["schtasks", "/Query", "/TN", task_name],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def _scheduled_task_exists_linux(task_name: str) -> bool | None:
    """None (not False) if neither check is available/conclusive — unlike
    Windows/macOS there's no single reliable place to re-query on Linux
    (reminder.py itself falls back between systemd-run and at depending
    on what's installed), and a real 'not found' isn't safely
    distinguishable here from 'couldn't check'."""
    try:
        result = subprocess.run(["atq"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and task_name in result.stdout:
            return True
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["systemctl", "--user", "list-timers", "--all"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return task_name in result.stdout
    except Exception:
        pass
    return None


def _check_reminder(args: dict, result: str) -> str | None:
    from config import get_os

    date_str = (args.get("date") or "").strip()
    time_str = (args.get("time") or "").strip()
    if not date_str or not time_str:
        return None  # tool itself already refuses this case

    try:
        target_dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    except ValueError:
        return None  # tool itself already refuses this case

    task_name = f"JARVISReminder_{target_dt.strftime('%Y%m%d_%H%M%S')}"
    os_name   = get_os()

    if os_name == "windows":
        found = _scheduled_task_exists_windows(task_name)
    elif os_name == "mac":
        plist_path = Path.home() / "Library" / "LaunchAgents" / f"com.jarvis.reminder.{task_name}.plist"
        found = plist_path.exists()
    else:
        found = _scheduled_task_exists_linux(task_name)
        if found is None:
            return None

    return None if found else f"Scheduled task '{task_name}' was not found in the OS scheduler."


def _check_game_updater(args: dict, result: str) -> str | None:
    action = (args.get("action") or "update").strip().lower()
    if action != "schedule":
        return None  # install/update/list/etc — not checked, see the investigation

    from config import get_os
    os_name   = get_os()
    task_name = "JARVIS_GameUpdater"  # fixed constant in actions/game_updater.py, not derived from params

    if os_name == "windows":
        found = _scheduled_task_exists_windows(task_name)
    elif os_name == "mac":
        plist_path = Path.home() / "Library" / "LaunchAgents" / "com.jarvis.gameupdater.plist"
        found = plist_path.exists()
    else:
        found = _scheduled_task_exists_linux(task_name)
        if found is None:
            return None

    return None if found else f"Scheduled task '{task_name}' was not found in the OS scheduler."


# ---------------------------------------------------------------------------
# desktop_control — wallpaper action only, reusing the existing
# get_current_wallpaper() readback that already exists in actions/desktop.py
# but was never used as a postcondition before this phase.
# ---------------------------------------------------------------------------

# actions/desktop.py's set_wallpaper() converts these to a temp .bmp on
# Windows before actually setting it, so the *original* input path never
# appears in the readback for these formats even on genuine success —
# comparing against it would be a false failure, not a real one.
_WALLPAPER_CONVERTED_FORMATS = {".webp", ".png"}

_WALLPAPER_READBACK_FAILURE_MARKERS = (
    "could not get wallpaper",
    "wallpaper path retrieval not supported",
)


def _check_desktop_control(args: dict, result: str) -> str | None:
    action = (args.get("action") or "").strip().lower()
    if action != "wallpaper":
        # wallpaper_url: set_wallpaper_from_url() downloads to a temp
        # file, sets it, then deletes the temp file before returning —
        # nothing is left on disk to compare against by the time this
        # check runs, so there's no reliable check for this action.
        # organize/clean/list/stats/task: not covered — organize/clean
        # would need a pre-call snapshot to avoid false positives from
        # files already present in the target folders; task is arbitrary
        # LLM-generated code with no defined postcondition.
        return None

    path = (args.get("path") or "").strip()
    if not path:
        return None  # tool itself already refuses this case

    from actions.desktop import get_current_wallpaper

    current = get_current_wallpaper()
    lowered = current.lower()
    if any(marker in lowered for marker in _WALLPAPER_READBACK_FAILURE_MARKERS):
        return f"could not confirm wallpaper change — readback failed: {current}"

    if Path(path).suffix.lower() not in _WALLPAPER_CONVERTED_FORMATS:
        stem = Path(path).stem.lower()
        if stem and stem not in lowered:
            return f"current wallpaper ({current}) does not appear to reference '{path}'."
    return None


# ---------------------------------------------------------------------------
# computer_control — screen_click only. Re-screenshots after a short settle
# delay and asks the vision model an independent outcome question derived
# from the click's own `description` — "does the screen now look like X
# happened" — rather than re-confirming the original locate call's (x, y)
# guess with the same fallible process that produced it in the first place.
# That distinction is exactly why this is scoped to screen_click alone: it's
# the one computer_control action whose arguments carry a natural-language
# claim about the *end state* it should produce, which is genuinely
# checkable independently of how the click was targeted. Raw click/type/
# hotkey/drag/press have no such derivable claim from their arguments alone
# (click(500, 300) has no "expected outcome" to ask a vision model about)
# and stay deliberately unchecked — this was the same reasoning that
# excluded computer_control entirely in the earlier verification phase; a
# vision-based outcome check for screen_click doesn't share the rejected
# "self-verification" shape, but nothing else about that exclusion changes.
# ---------------------------------------------------------------------------

_SCREEN_CLICK_SETTLE_SECONDS = 0.6


def _check_computer_control(args: dict, result: str) -> str | None:
    action = (args.get("action") or "").strip().lower()
    if action != "screen_click":
        return None  # every other action has no independently-checkable claim

    description = (args.get("description") or "").strip()
    if not description or result.startswith("Element not found"):
        return None  # nothing was clicked — the tool's own result already says so

    try:
        import pyautogui
    except ImportError:
        return None  # can't verify without a screenshot backend — no opinion

    import time as _time
    import io as _io
    from config import load_config as _load_config
    from core.llm_client import call_llm_vision as _llm_vision

    _time.sleep(_SCREEN_CLICK_SETTLE_SECONDS)

    try:
        cfg          = _load_config()
        vision_model = cfg.get("vision_model") or cfg.get("llm_model", "llava")
        img          = pyautogui.screenshot()
        buf          = _io.BytesIO()
        img.save(buf, format="PNG")

        prompt = (
            f"A user just tried to click on '{description}' on this screen. "
            f"Looking at this CURRENT screenshot, does it look like that "
            f"click succeeded — i.e. does the screen now show the effect of "
            f"interacting with '{description}' (it opened, activated, "
            f"changed state, highlighted, navigated, etc.)? "
            f"Reply with ONLY one word: YES, NO, or UNCLEAR if you cannot "
            f"tell from this screenshot alone."
        )
        # timeout=60 matches every other vision call site in this codebase
        # (screen_processor.py, vision_fix_code.py) — a shorter value was
        # tried first and observed to time out on a cold model load (the
        # vision model not yet resident in Ollama), producing a spurious
        # "couldn't verify" instead of a real answer.
        answer = _llm_vision(prompt, buf.getvalue(), model=vision_model, timeout=60).strip().upper()
    except Exception as e:
        _log.warning("computer_control postcondition check couldn't run: %s", e)
        return None  # couldn't verify -> no opinion, never a false failure

    if answer.startswith("NO"):
        return f"screen_click on '{description}' does not appear to have taken effect (vision check: NO)."
    return None  # YES, UNCLEAR, or an unparsed response -> no opinion, not a failure


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

POSTCONDITION_CHECKS = {
    "file_controller":  _check_file_controller,
    "file_processor":   _check_file_processor,
    "code_helper":      _check_code_helper,
    "dev_agent":        _check_dev_agent,
    "reminder":         _check_reminder,
    "game_updater":     _check_game_updater,
    "desktop_control":  _check_desktop_control,
    "computer_control": _check_computer_control,
}


def check_postcondition(tool: str, args: dict, result: str) -> str | None:
    """Returns None if the postcondition holds (or none is registered for
    `tool`), else a short description of what was expected but not
    found. A checker raising is never mistaken for the postcondition
    itself failing — logged and treated as 'couldn't verify'."""
    checker = POSTCONDITION_CHECKS.get(tool)
    if checker is None:
        return None
    try:
        return checker(args, result)
    except Exception as e:
        _log.warning("Postcondition checker for '%s' raised: %s", tool, e, exc_info=True)
        return None
