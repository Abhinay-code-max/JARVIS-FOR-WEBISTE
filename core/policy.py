"""
core/policy.py
================
The permission_policy table: which of the four levels — auto-allow,
notify-only, ask-and-wait, hard-deny — applies to each TOOL_DISPATCH tool,
optionally overridden per action for tools whose action set spans a real
risk range.

`action=None` is a tool's default row, used when no more specific
(tool_name, action) row matches the action actually invoked. hard-deny has
no rows in this phase — hooked up for a future need, not forced onto
anything now.

Where a tool's action set wasn't explicitly enumerated in the resolved
policy, the classification below is this module's own best-effort split
(documented inline) — flagged to the user rather than silently assumed.
"""
from __future__ import annotations

import logging
import threading

from core.db import get_conn

_log = logging.getLogger("jarvis.policy")

AUTO_ALLOW   = "auto-allow"
NOTIFY_ONLY  = "notify-only"
ASK_AND_WAIT = "ask-and-wait"
HARD_DENY    = "hard-deny"

LEVELS = {AUTO_ALLOW, NOTIFY_ONLY, ASK_AND_WAIT, HARD_DENY}

# (tool_name, action, level) — action=None is the tool-wide default row.
DEFAULT_POLICY: list[tuple[str, str | None, str]] = [
    # ── auto-allow — read-only / no meaningful side effect ──────────────────
    ("weather_report", None, AUTO_ALLOW),
    ("web_search",      None, AUTO_ALLOW),
    ("open_app",         None, AUTO_ALLOW),
    ("youtube_video",    None, AUTO_ALLOW),
    ("screen_process",   None, AUTO_ALLOW),
    ("daily_briefing",   None, AUTO_ALLOW),
    ("flight_finder",    None, AUTO_ALLOW),

    # ── notify-only defaults — proceeds, user told after the fact ──────────
    ("file_processor",    None, NOTIFY_ONLY),   # non-extract ops (summarize/convert/transcribe/etc.)
    ("browser_control",   None, NOTIFY_ONLY),   # navigation/read: switch, list_browsers, close_all,
                                                 # go_to, search, scroll, get_text, get_url, new_tab,
                                                 # close_tab, screenshot, back, forward, reload, close
    ("computer_control",  None, NOTIFY_ONLY),   # passive/read: copy, screenshot, screen_find, wait,
                                                 # random_data, user_data, focus_window, move
    ("computer_settings", None, NOTIFY_ONLY),   # volume/brightness/window-mgmt/tab-nav/clipboard
                                                 # shortcuts — canned OS/browser hotkeys, not raw
                                                 # coordinate input; everything except restart/shutdown
    ("file_controller",   None, NOTIFY_ONLY),   # list, read, find, largest, disk_usage, info
    ("game_updater",      None, NOTIFY_ONLY),   # list, download_status, schedule*, update without
                                                 # shutdown_when_done

    # ── ask-and-wait — whole tool (single or delegated confirmation) ───────
    ("desktop_control", None, ASK_AND_WAIT),
    ("code_helper",      None, ASK_AND_WAIT),
    ("dev_agent",         None, ASK_AND_WAIT),  # delegated — core/tool_gate.DELEGATED_TOOLS
    ("vision_fix_code",   None, ASK_AND_WAIT),  # delegated — core/tool_gate.DELEGATED_TOOLS
    ("send_message",       None, ASK_AND_WAIT),
    # Missed in the original permission-model pass — found while building
    # per-tool contracts (core/tool_contracts.py) and cross-checking every
    # TOOL_DISPATCH tool has a policy row. Writes a notify script to disk
    # and registers a real OS-level scheduled task/cron/launchd job — same
    # risk class as file_controller's write/create_file, not a read.
    ("reminder", None, ASK_AND_WAIT),

    # ── ask-and-wait — action-specific overrides ────────────────────────────
    # file_controller: explicitly called out (write/delete/move/rename/
    # organize_desktop) plus create_file/create_folder/copy, which are the
    # same "creates/duplicates something on disk" risk class as write/move
    # but weren't named explicitly — added here, flagged to the user.
    ("file_controller", "write",            ASK_AND_WAIT),
    ("file_controller", "delete",           ASK_AND_WAIT),
    ("file_controller", "move",             ASK_AND_WAIT),
    ("file_controller", "rename",           ASK_AND_WAIT),
    ("file_controller", "organize_desktop", ASK_AND_WAIT),
    ("file_controller", "create_file",      ASK_AND_WAIT),
    ("file_controller", "create_folder",    ASK_AND_WAIT),
    ("file_controller", "copy",             ASK_AND_WAIT),

    ("computer_settings", "restart",  ASK_AND_WAIT),
    ("computer_settings", "shutdown", ASK_AND_WAIT),

    # game_updater's shutdown path isn't a distinct action string — it's the
    # shutdown_when_done=true flag on an update/download action. See
    # ACTION_EXTRACTORS below for how that's turned into a synthetic action.
    ("game_updater", "shutdown_when_done", ASK_AND_WAIT),

    ("file_processor", "extract", ASK_AND_WAIT),

    # browser_control: fill_form/smart_type/click were named explicitly.
    # smart_click/type/press carry the same risk (arbitrary data entry or a
    # keypress that can submit a form) and were added here, flagged.
    ("browser_control", "fill_form",   ASK_AND_WAIT),
    ("browser_control", "smart_type",  ASK_AND_WAIT),
    ("browser_control", "click",       ASK_AND_WAIT),
    ("browser_control", "smart_click", ASK_AND_WAIT),
    ("browser_control", "type",        ASK_AND_WAIT),
    ("browser_control", "press",       ASK_AND_WAIT),

    # computer_control: click/type/hotkey/drag were named explicitly. The
    # remaining input-injection actions (same character, same risk) were
    # added here, flagged: left_click/double_click/right_click (click
    # variants), press, scroll, paste, clear_field, screen_click (AI-finder
    # + click composite).
    ("computer_control", "type",         ASK_AND_WAIT),
    ("computer_control", "smart_type",   ASK_AND_WAIT),
    ("computer_control", "click",        ASK_AND_WAIT),
    ("computer_control", "left_click",   ASK_AND_WAIT),
    ("computer_control", "double_click", ASK_AND_WAIT),
    ("computer_control", "right_click",  ASK_AND_WAIT),
    ("computer_control", "drag",         ASK_AND_WAIT),
    ("computer_control", "hotkey",       ASK_AND_WAIT),
    ("computer_control", "press",        ASK_AND_WAIT),
    ("computer_control", "scroll",       ASK_AND_WAIT),
    ("computer_control", "paste",        ASK_AND_WAIT),
    ("computer_control", "clear_field",  ASK_AND_WAIT),
    ("computer_control", "screen_click", ASK_AND_WAIT),

    # ── hard-deny — no rows assigned in this phase ──────────────────────────
]

_seed_lock = threading.Lock()
_seeded    = False


def seed_default_policy() -> None:
    """Inserts any DEFAULT_POLICY row not already present — additive, not a
    one-shot "only if the table is empty" gate. A row-count gate would mean
    a tool added to DEFAULT_POLICY after a deployment's first run (like
    'reminder', missed in the original pass and added later) would never
    get seeded on existing installs, silently falling back to
    get_policy_level()'s ask-and-wait default forever instead of the
    intended row. Checked per-row rather than relying on a UNIQUE
    constraint because SQLite doesn't dedupe NULL in (tool_name, action) —
    see core/db.py's permission_policy comment. Table stays small (order
    of DEFAULT_POLICY's length), so a SELECT per candidate row is cheap."""
    conn = get_conn()
    added = 0
    with conn:
        for tool_name, action, level in DEFAULT_POLICY:
            if action is None:
                exists = conn.execute(
                    "SELECT 1 FROM permission_policy WHERE tool_name = ? AND action IS NULL",
                    (tool_name,),
                ).fetchone()
            else:
                exists = conn.execute(
                    "SELECT 1 FROM permission_policy WHERE tool_name = ? AND action = ?",
                    (tool_name, action),
                ).fetchone()
            if exists:
                continue
            conn.execute(
                "INSERT INTO permission_policy (tool_name, action, level) VALUES (?, ?, ?)",
                (tool_name, action, level),
            )
            added += 1
    if added:
        _log.info("Seeded %d new permission_policy row(s).", added)


def _ensure_seeded() -> None:
    global _seeded
    if _seeded:
        return
    with _seed_lock:
        if _seeded:
            return
        seed_default_policy()
        _seeded = True


def get_policy_level(tool_name: str, action: str | None) -> str:
    """Most specific match wins: (tool_name, action) row if one exists,
    else the tool's (tool_name, NULL) default row, else ask-and-wait —
    an unlisted tool fails toward asking, not toward auto-allow."""
    _ensure_seeded()
    conn = get_conn()
    if action:
        row = conn.execute(
            "SELECT level FROM permission_policy WHERE tool_name = ? AND action = ?",
            (tool_name, action),
        ).fetchone()
        if row:
            return row["level"]
    row = conn.execute(
        "SELECT level FROM permission_policy WHERE tool_name = ? AND action IS NULL",
        (tool_name,),
    ).fetchone()
    return row["level"] if row else ASK_AND_WAIT


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


# Per-tool extractor: raw TOOL_DISPATCH args -> the `action` string to look
# up in permission_policy (None = use the tool's default row). Only tools
# actually split by action need an entry here.
ACTION_EXTRACTORS = {
    "computer_control":  lambda args: (args.get("action") or "").strip().lower() or None,
    "browser_control":   lambda args: (args.get("action") or "").strip().lower() or None,
    "file_controller":   lambda args: (args.get("action") or "").strip().lower() or None,
    "computer_settings": lambda args: (args.get("action") or "").strip().lower() or None,
    "file_processor":    lambda args: (
        "extract" if (args.get("action") or "").strip().lower() == "extract" else None
    ),
    # shutdown_when_done is a boolean flag on update/download actions, not
    # its own action string — mapped to a synthetic action name so it can
    # still get its own permission_policy row.
    "game_updater": lambda args: (
        "shutdown_when_done" if _truthy(args.get("shutdown_when_done", "")) else None
    ),
}
