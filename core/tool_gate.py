"""
core/tool_gate.py
==================
Single chokepoint for every TOOL_DISPATCH call. main.py's interactive
_execute_tool (player=self.ui) and agent/executor.py's _call_tool
(player=None, always — see core/confirm.py) both route through
dispatch_tool() instead of indexing TOOL_DISPATCH directly, so policy
evaluation, input validation, execution timeout, and audit logging all
happen in exactly one place instead of being scattered across 19 modules.

DELEGATED_TOOLS keep their own internal CONFIRM.request() call(s) instead
of being gated here — see the constant's docstring for why those two
specifically can't be replaced by a single entry-level ask. They still go
through input validation and the timeout wrapper below.

Thread-affinity note (see _invoke_with_timeout): timeout enforcement runs
the underlying TOOL_DISPATCH[tool] call on a fresh daemon thread. Verified
empirically that pyautogui-based tools (computer_control, computer_settings,
desktop.py, send_message.py) have no thread-affinity issue — read-only
pyautogui calls succeed identically from a worker thread. youtube_video is
the one exception (core/tool_contracts.py sets enforce_timeout=False for
it): its _ask_for_url() creates a real tkinter Tk root, and tkinter/Tcl is
not safe to abandon mid-call from a different thread than the one that
created the root — a timeout there wouldn't cleanly cancel the dialog, it
would corrupt the process-global tk._default_root for every later call.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from core.tool_dispatch    import TOOL_DISPATCH
from core.tool_contracts   import get_contract, ToolContract
from core.confirm          import CONFIRM
from core.policy           import (
    get_policy_level, ACTION_EXTRACTORS, ASK_AND_WAIT, HARD_DENY, AUTO_ALLOW, NOTIFY_ONLY,
    DESKTOP,
)
from core.task_approval    import TASK_APPROVAL
from core.postconditions   import check_postcondition
from core.db                import get_conn

_log = logging.getLogger("jarvis.tool_gate")

# Tools additionally scoped by an allowed-path-prefix check for non-
# desktop (service:*) callers, on top of whatever permission_policy level
# applies — see SERVICE_PROJECT_ROOTS/_validate_service_path_scope()
# below. Only file_controller today: the one service:*-allow-listed tool
# whose args name a filesystem path at all.
SERVICE_PATH_SCOPED_TOOLS = {"file_controller"}

# Best-known root(s) a service:* caller's file_controller calls may
# touch — deliberately narrower than desktop's own _SAFE_ROOTS
# (Path.home(), in actions/file_controller.py), which is unchanged for
# the desktop caller class. Currently just this repo's own checkout —
# the only path verifiable from inside this repo. eyv-website-2's local
# checkout path (if service:bugfix needs to touch it too) is NOT
# included here: flagged as an open question needing confirmation before
# service:bugfix's file_controller access is actually useful for that
# repo, same as core/policy.py's web_search/flight_finder scope question
# for service:personal — not decided silently either way.
SERVICE_PROJECT_ROOTS: list[Path] = [
    Path(__file__).resolve().parent.parent,  # this repo's own root
]


def _validate_service_path_scope(tool: str, args: dict, caller_class: str) -> str | None:
    """Returns None if `args`'s path argument (if any) resolves inside
    SERVICE_PROJECT_ROOTS, else a short rejection reason. No-op for the
    desktop caller class (its own, broader path check inside
    actions/file_controller.py is unchanged) and for tools outside
    SERVICE_PATH_SCOPED_TOOLS. Reuses actions/file_controller.py's own
    _resolve_path() (shortcut keywords like "desktop"/"downloads" plus
    expanduser) rather than a separate resolution scheme, so this check
    evaluates the exact same path the tool itself will actually use —
    lazy import to keep core/tool_gate.py's own module level free of
    actions/* imports, matching core/tool_dispatch.py's existing
    per-wrapper lazy-import convention."""
    if caller_class == DESKTOP or tool not in SERVICE_PATH_SCOPED_TOOLS:
        return None

    from actions.file_controller import _resolve_path

    raw_path = args.get("path") or "desktop"  # file_controller's own default
    try:
        resolved = _resolve_path(raw_path).resolve()
    except Exception:
        return f"path {raw_path!r} could not be resolved."

    if any(resolved == root or resolved.is_relative_to(root) for root in SERVICE_PROJECT_ROOTS):
        return None
    return f"path {raw_path!r} is outside the allowed project directories for caller class {caller_class!r}."

# Tools whose own CONFIRM.request() call(s) stay in place instead of a
# single dispatch-entry gate:
#   - dev_agent: up to 3 separate decision points per invocation (pip
#     install with a specific package list, running a specific command,
#     a reactively-discovered auto-install package after a run fails).
#     None of that content exists yet at dispatch entry — a single ask
#     would fire before the project is even planned, and the auto-install
#     package genuinely can't be known until a run fails partway through.
#   - vision_fix_code: one CONFIRM call, but it fires after screenshot +
#     vision analysis + on-disk file search/read, with a prompt naming the
#     exact file and bug found. Entry-level gating would ask blind, and
#     would still ask on the early-exit paths (no bug found / filename or
#     file unresolvable / ambiguous match) where nothing would actually be
#     written — asking when there's nothing to approve yet.
DELEGATED_TOOLS = {"dev_agent", "vision_fix_code"}


def _extract_action(tool: str, args: dict) -> str | None:
    extractor = ACTION_EXTRACTORS.get(tool)
    return extractor(args) if extractor else None


def _computer_control_detail(args: dict) -> str | None:
    """computer_control's action names alone don't say *what* will be
    clicked/typed/dragged — unlike e.g. file_controller's extracted action
    string, which already names the target file. That's fine for policy
    lookup (ACTION_EXTRACTORS only needs the action name), but a human
    approving an ask-and-wait request — especially a background task's,
    answered out-of-band via approve_task with no other context — deserves
    to see the actual target, not just the word 'click'."""
    action = (args.get("action") or "").strip().lower()

    if action in ("screen_find", "screen_click"):
        desc = args.get("description")
        return f"target: {desc}" if desc else None

    if action in ("type", "smart_type", "paste"):
        text = args.get("text", "")
        if not text:
            return None
        snippet = text[:60] + ("…" if len(text) > 60 else "")
        return f"text: {snippet!r}"

    if action in ("click", "left_click", "double_click", "right_click", "move"):
        x, y = args.get("x"), args.get("y")
        return f"at ({x}, {y})" if x is not None and y is not None else None

    if action == "drag":
        return f"from ({args.get('x1')}, {args.get('y1')}) to ({args.get('x2')}, {args.get('y2')})"

    if action == "hotkey":
        keys = args.get("keys")
        return f"keys: {keys}" if keys else None

    if action == "press":
        key = args.get("key")
        return f"key: {key}" if key else None

    if action == "focus_window":
        title = args.get("title")
        return f"window: {title}" if title else None

    return None


# Per-tool detail extractor: raw dispatch args -> a short human-readable
# description of what the call will actually do, appended to the
# ask-and-wait prompt. Only computer_control has one today — see
# _computer_control_detail's docstring for why it specifically needs one.
DETAIL_EXTRACTORS = {
    "computer_control": _computer_control_detail,
}


def _extract_detail(tool: str, args: dict) -> str | None:
    extractor = DETAIL_EXTRACTORS.get(tool)
    return extractor(args) if extractor else None


def _log_decision(tool: str, action: str | None, level: str, outcome: str, task_id: str | None) -> None:
    try:
        conn = get_conn()
        with conn:
            conn.execute(
                "INSERT INTO policy_decisions "
                "(tool_name, action, level_evaluated, outcome, task_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (tool, action, level, outcome, task_id, time.time()),
            )
    except Exception as e:
        _log.warning("Could not log policy decision (%s/%s): %s", tool, action, e)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

# INTEGER/NUMBER accept numeric strings too, not just python int/float —
# matched to the actual crash risk found in the codebase (e.g.
# flight_finder's int(params.get("passengers", 1)), computer_control's
# int(params.get("x", 0))): those coercions succeed on "5" just as well as
# on 5, so rejecting a numeric string here would be stricter than the tool
# itself needs. What DOES crash them is a non-numeric value ("two", a
# list, etc.) — that's what gets caught.
def _is_integer_like(v) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return True
    if isinstance(v, str):
        try:
            int(v)
            return True
        except ValueError:
            return False
    return False


def _is_number_like(v) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, str):
        try:
            float(v)
            return True
        except ValueError:
            return False
    return False


_TYPE_CHECKERS = {
    "STRING":  lambda v: isinstance(v, str),
    "INTEGER": _is_integer_like,
    "NUMBER":  _is_number_like,
    "BOOLEAN": lambda v: isinstance(v, bool),
    "ARRAY":   lambda v: isinstance(v, list),
    "OBJECT":  lambda v: isinstance(v, dict),
}


def validate_input(tool: str, args: dict, schema: dict) -> str | None:
    """Returns None if `args` satisfies `schema` (TOOL_DECLARATIONS'
    Gemini-style {type, properties, required} shape), else a short
    human-readable description of what's wrong — required-but-missing
    field(s), or a field whose value the underlying tool would crash on
    (see _TYPE_CHECKERS). Unknown/extra keys are tolerated, not rejected —
    an LLM occasionally adding a harmless extra field isn't worth the
    friction of rejecting the whole call over."""
    if not isinstance(args, dict):
        return f"parameters must be an object, got {type(args).__name__}."

    props    = schema.get("properties", {})
    required = schema.get("required", [])

    missing = [f for f in required if args.get(f) in (None, "")]
    if missing:
        return f"missing required parameter(s): {', '.join(missing)}."

    for key, value in args.items():
        if key not in props or value is None:
            continue
        expected_type = props[key].get("type")
        checker = _TYPE_CHECKERS.get(expected_type)
        if checker and not checker(value):
            return (
                f"parameter '{key}' should be {expected_type.lower()}, "
                f"got {type(value).__name__} ({value!r})."
            )

    return None


# ---------------------------------------------------------------------------
# Execution timeout
# ---------------------------------------------------------------------------

def _invoke_with_timeout(tool: str, args: dict, player, speak, contract: ToolContract) -> str:
    """Runs TOOL_DISPATCH[tool] under contract.timeout_seconds. Spawns a
    plain daemon Thread rather than a shared ThreadPoolExecutor — a pool's
    worker threads are non-daemon by default and concurrent.futures
    registers an atexit hook that joins them, so a single genuinely hung
    tool call (see the still-bare subprocess.run() calls in
    computer_settings.py/desktop.py/game_updater.py/reminder.py, a
    separate follow-up) would hang the whole process's normal shutdown.
    A daemon thread can be abandoned on timeout without blocking exit —
    the OS reclaims it when the process ends. This does NOT kill the
    underlying subprocess/syscall the tool made; see the module docstring
    and core/tool_contracts.py for which tools are exempted (enforce_timeout)
    because abandonment itself is unsafe for them, not just incomplete."""
    if not contract.enforce_timeout:
        return TOOL_DISPATCH[tool](args, player, speak)

    box: dict = {}

    def _run():
        try:
            box["value"] = TOOL_DISPATCH[tool](args, player, speak)
        except Exception as e:
            box["error"] = e

    t = threading.Thread(target=_run, name=f"tool-exec-{tool}", daemon=True)
    t.start()
    t.join(timeout=contract.timeout_seconds)

    if t.is_alive():
        _log.warning(
            "Tool '%s' exceeded its %.0fs timeout — abandoning (thread left running).",
            tool, contract.timeout_seconds,
        )
        return f"'{tool}' did not complete within {contract.timeout_seconds:.0f}s and was abandoned."

    if "error" in box:
        raise box["error"]
    return box.get("value")


# ---------------------------------------------------------------------------
# Postcondition check
# ---------------------------------------------------------------------------

def _apply_postcondition(tool: str, args: dict, result: str) -> str:
    """Runs the postcondition check registered for `tool` (if any — see
    core/postconditions.py, most tools have none) and, on failure,
    returns a distinguishable synthetic string instead of the tool's own
    result — never raises here. Every dispatch_tool() failure path
    returns a string, matching validation/permission/timeout's own
    convention; deciding whether a postcondition failure should be
    re-raised as an exception and routed through analyze_error() is a
    retry-policy question, which belongs in agent/executor.py where
    retry decisions are actually made, not in this module."""
    unmet = check_postcondition(tool, args, result)
    if unmet is None:
        return result
    _log.warning("Postcondition unmet for '%s': %s", tool, unmet)
    return f"Postcondition unmet — '{tool}' {unmet} (tool reported: {result[:200]!r})"


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def dispatch_tool(
    tool: str,
    args: dict,
    player,
    speak,
    task_id: str | None = None,
    submitted_interactively: bool = True,
    caller_class: str = DESKTOP,
) -> str:
    """Validates input, evaluates permission_policy for (tool, action,
    caller_class), gates as needed, then calls TOOL_DISPATCH[tool] under
    its contract's timeout and checks its postcondition (if one is
    registered — see core/postconditions.py). Never call this from the
    audio/transcript thread — same rule as CONFIRM.request().

    caller_class defaults to 'desktop' so every pre-existing call site
    (main.py, agent/executor.py's _call_tool) behaves exactly as before
    without needing to pass anything new yet — see core/policy.py's
    get_policy_level() for how the two caller-class families (desktop vs.
    service:*) differ once a service:* value is actually passed in (by
    the future network boundary).

    Every path that actually invokes the tool (delegated, auto-allow,
    notify-only, ask-and-wait-approved) funnels through the single
    "invoke, then check" tail near the end — nothing after the branching
    below returns early once invocation has been decided, so the
    postcondition check always runs regardless of which permission level
    or gate path got us here, instead of being duplicated at each of the
    4 call sites that used to each return _invoke_with_timeout(...)
    directly."""
    if tool not in TOOL_DISPATCH:
        raise ValueError(
            f"Unknown tool '{tool}' — no such tool exists. "
            f"Available tools: {sorted(TOOL_DISPATCH.keys())}"
        )

    contract = get_contract(tool)
    action   = _extract_action(tool, args)
    level    = get_policy_level(tool, action, caller_class)

    validation_error = validate_input(tool, args, contract.input_schema)
    if validation_error:
        _log_decision(tool, action, level, "rejected", task_id)
        return f"Rejected — '{tool}' parameters are invalid: {validation_error}"

    path_scope_error = _validate_service_path_scope(tool, args, caller_class)
    if path_scope_error:
        _log_decision(tool, action, level, "rejected", task_id)
        return f"Rejected — '{tool}' {path_scope_error}"

    should_invoke = True
    notify_after  = False
    early_result  = None
    outcome       = None

    if tool in DELEGATED_TOOLS:
        outcome = "delegated"

    elif level == HARD_DENY:
        should_invoke = False
        outcome       = "denied"
        early_result  = f"'{tool}' is not permitted."

    elif level == AUTO_ALLOW:
        outcome = "auto-allowed"

    elif level == NOTIFY_ONLY:
        outcome      = "notified"
        notify_after = True

    else:
        assert level == ASK_AND_WAIT
        detail = _extract_detail(tool, args)
        prompt = (
            f"I'm about to run {tool}"
            + (f" — {action}" if action else "")
            + (f" ({detail})" if detail else "")
            + "."
        )

        if player is not None:
            approved = CONFIRM.request(player, prompt, speak=speak)
            outcome  = "approved" if approved else "denied"
        else:
            note = " (submitted interactively)" if submitted_interactively else " (system-submitted)"
            approved, outcome = TASK_APPROVAL.wait_for_approval(task_id, prompt + note)

        if not approved:
            should_invoke = False
            # Message text is the only signal agent/executor.py's
            # _approval_outcome() has to distinguish "timed out" from
            # "explicitly denied" (outcome itself isn't returned to the
            # caller) — keep these two phrasings and don't merge them.
            if outcome == "timeout":
                early_result = f"Cancelled — '{tool}' timed out waiting for approval."
            else:
                early_result = f"Cancelled — '{tool}' was not approved."

    if should_invoke:
        result = _invoke_with_timeout(tool, args, player, speak, contract)
        result = _apply_postcondition(tool, args, result)
        if notify_after and player is not None:
            try:
                player.write_log(f"NOTICE: ran {tool}" + (f" ({action})" if action else ""))
            except Exception:
                pass
    else:
        result = early_result

    _log_decision(tool, action, level, outcome, task_id)
    return result
