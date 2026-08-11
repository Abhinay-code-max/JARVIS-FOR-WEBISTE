"""
core/policy.py
================
The permission_policy table: which of the four levels — auto-allow,
notify-only, ask-and-wait, hard-deny — applies to each TOOL_DISPATCH tool,
per caller_class, optionally overridden per action for tools whose action
set spans a real risk range.

`action=None` is a tool's default row, used when no more specific
(tool_name, action, caller_class) row matches the action actually invoked.

Two caller-class models, not one (see get_policy_level()'s docstring for
the fallback each uses when no row matches at all):
  - 'desktop' (the local PyQt6 client — DEFAULT_POLICY below): unlisted
    tool falls back to ask-and-wait. hard-deny has no rows in this phase
    for desktop — hooked up for a future need, not forced onto anything
    now. Unchanged from before the headless-extraction phase — every
    existing desktop row and lookup behavior stays exactly as it was.
  - 'service:<name>' (SERVICE_POLICY below, one of the four backend-agent
    classes calling in over the future network boundary — see
    core/proactive.py-style module docs for the extraction-phase
    context): default-deny, explicit allow-list up. A class with no row
    for a given tool has never been granted it, not merely left
    unconfigured — the opposite fallback direction from desktop.

Where a tool's action set wasn't explicitly enumerated in the resolved
policy, the classification below is this module's own best-effort split
(documented inline) — flagged to the user rather than silently assumed.

DELEGATED_TOOLS caveat (dev_agent, vision_fix_code): core/tool_gate.py's
dispatch_tool() checks `tool in DELEGATED_TOOLS` *before* it ever
evaluates a permission_policy level — see that module's docstring. That
means a SERVICE_POLICY row for either tool is NOT what actually gates
access for a service:* caller; it's effectively a no-op at the
policy-table level regardless of what it says. The real protection for
any caller with no live UI (every service:* class, and today's own
background task queue) is core/confirm.py's CONFIRM.request(): it fails
closed unconditionally whenever `player is None`, which is always true
off the desktop UI thread. This is a structural gap in the same family as
the agent_task one (see core/proactive.py's Step 1.2b-equivalent
investigation) — flagged here, not silently paved over by pretending the
policy row is the enforcement point.
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

DESKTOP              = "desktop"
SERVICE_SUPPORT      = "service:support"
SERVICE_BUGFIX       = "service:bugfix"
SERVICE_PROMOTIONS   = "service:promotions"
SERVICE_PERSONAL     = "service:personal"

CALLER_CLASSES = {DESKTOP, SERVICE_SUPPORT, SERVICE_BUGFIX, SERVICE_PROMOTIONS, SERVICE_PERSONAL}
SERVICE_CALLER_CLASSES = CALLER_CLASSES - {DESKTOP}

# (tool_name, action, level) — action=None is the tool-wide default row.
# Desktop's rows only — seeded with caller_class='desktop' by
# seed_default_policy(). Unchanged from before the headless-extraction
# phase.
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

# caller_class -> [(tool_name, action, level), ...] — each service
# class's explicit allow-list. Everything NOT listed here falls to
# get_policy_level()'s hard-deny default for service:* callers (see this
# module's docstring) — absence from this list is a real, intentional
# denial, not an oversight to fix later.
#
# Every row here is ask-and-wait, deliberately more conservative than
# just inheriting desktop's level for the same tool (e.g. file_processor
# defaults to notify-only for desktop — proceeds, user told after the
# fact — which is the wrong shape for a caller with no live human present
# to be told anything). A new, less-trusted caller class starting at the
# strictest gate available and being loosened later with a specific
# reason beats starting loose. This wasn't spelled out as a numeric
# level in the phase-1 plan, so it's flagged here as a deliberate choice,
# not a silent default.
#
# agent_task is deliberately absent from every list below and cannot be
# added here at all today — it doesn't go through dispatch_tool()/
# permission_policy in the first place (see main.py's _execute_tool()
# special case). Gated separately, unconditionally hard-denied for every
# service:* class regardless of this table — see the agent_task-gap
# investigation (core/proactive.py's Step-1.2b-equivalent commit).
SERVICE_POLICY: dict[str, list[tuple[str, str | None, str]]] = {
    SERVICE_BUGFIX: [
        ("code_helper",     None, ASK_AND_WAIT),
        # dev_agent / vision_fix_code: allow-listed per the phase-1 plan,
        # but see this module's docstring — DELEGATED_TOOLS bypasses this
        # row entirely at dispatch_tool(). The row is kept here for
        # completeness/documentation of intent, not because it's what
        # actually gates these two for this class.
        ("dev_agent",       None, ASK_AND_WAIT),
        ("vision_fix_code", None, ASK_AND_WAIT),
        ("file_processor",  None, ASK_AND_WAIT),
        # file_controller: additionally scoped to project directories
        # only for this caller class — see core/tool_gate.py's
        # SERVICE_PATH_SCOPED_TOOLS / _validate_service_path_scope().
        # The policy row alone only gets the tool past the *permission*
        # gate; the path-prefix check is a separate, additional
        # validation step, same relationship file_controller's own
        # existing _is_safe_path()/_SAFE_ROOTS has to desktop callers.
        ("file_controller", None, ASK_AND_WAIT),
    ],
    # service:support: support_agent_service.py's own 5-tool registry
    # (lookup_user_trips, lookup_booking, get_refund_policy,
    # escalate_to_human, create_or_append_ticket) is entirely
    # EYV-internal — confirmed zero overlap with TOOL_DISPATCH. Nothing
    # to allow here; every Hermes tool falls to the hard-deny default.
    SERVICE_SUPPORT: [],
    # service:promotions: EYV's own analytics API already covers this
    # agent's needs. Nothing to allow here.
    SERVICE_PROMOTIONS: [],
    SERVICE_PERSONAL: [
        ("reminder",       None, ASK_AND_WAIT),
        ("file_processor", None, ASK_AND_WAIT),
        # web_search / flight_finder: open scope question per the phase-1
        # plan — deliberately left out, not decided silently either way.
    ],
}

_seed_lock = threading.Lock()
_seeded    = False


def _row_exists(conn, tool_name: str, action: str | None, caller_class: str) -> bool:
    if action is None:
        return conn.execute(
            "SELECT 1 FROM permission_policy WHERE tool_name = ? AND action IS NULL AND caller_class = ?",
            (tool_name, caller_class),
        ).fetchone() is not None
    return conn.execute(
        "SELECT 1 FROM permission_policy WHERE tool_name = ? AND action = ? AND caller_class = ?",
        (tool_name, action, caller_class),
    ).fetchone() is not None


def seed_default_policy() -> None:
    """Inserts any DEFAULT_POLICY (caller_class='desktop') or
    SERVICE_POLICY row not already present — additive, not a one-shot
    "only if the table is empty" gate. A row-count gate would mean a tool
    added to either policy list after a deployment's first run (like
    'reminder', missed in the original pass and added later) would never
    get seeded on existing installs, silently falling back to
    get_policy_level()'s own default forever instead of the intended row.
    Checked per-row rather than relying on a UNIQUE constraint because
    SQLite doesn't dedupe NULL in (tool_name, action, caller_class) — see
    core/db.py's permission_policy comment. Table stays small (order of
    DEFAULT_POLICY's + SERVICE_POLICY's combined length), so a SELECT per
    candidate row is cheap."""
    conn = get_conn()
    added = 0
    with conn:
        for tool_name, action, level in DEFAULT_POLICY:
            if _row_exists(conn, tool_name, action, DESKTOP):
                continue
            conn.execute(
                "INSERT INTO permission_policy (tool_name, action, level, caller_class) VALUES (?, ?, ?, ?)",
                (tool_name, action, level, DESKTOP),
            )
            added += 1

        for caller_class, rows in SERVICE_POLICY.items():
            for tool_name, action, level in rows:
                if _row_exists(conn, tool_name, action, caller_class):
                    continue
                conn.execute(
                    "INSERT INTO permission_policy (tool_name, action, level, caller_class) VALUES (?, ?, ?, ?)",
                    (tool_name, action, level, caller_class),
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


def get_policy_level(tool_name: str, action: str | None, caller_class: str = DESKTOP) -> str:
    """Most specific match wins: (tool_name, action, caller_class) row if
    one exists, else the tool's (tool_name, NULL, caller_class) default
    row, else a caller-class-dependent fallback:
      - 'desktop': ask-and-wait — an unlisted tool fails toward asking,
        not toward auto-allow. Exact pre-existing behavior, unchanged.
      - any 'service:*' class: hard-deny — default-deny, explicit
        allow-list up (see SERVICE_POLICY / this module's docstring). A
        tool with no row for this caller class was never granted it, not
        merely left unconfigured.
    `caller_class` defaults to 'desktop' so every pre-existing call site
    (main.py, agent/executor.py's _call_tool) keeps behaving exactly as
    before without needing to change yet."""
    _ensure_seeded()
    conn = get_conn()
    if action:
        row = conn.execute(
            "SELECT level FROM permission_policy WHERE tool_name = ? AND action = ? AND caller_class = ?",
            (tool_name, action, caller_class),
        ).fetchone()
        if row:
            return row["level"]
    row = conn.execute(
        "SELECT level FROM permission_policy WHERE tool_name = ? AND action IS NULL AND caller_class = ?",
        (tool_name, caller_class),
    ).fetchone()
    if row:
        return row["level"]
    return ASK_AND_WAIT if caller_class == DESKTOP else HARD_DENY


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
