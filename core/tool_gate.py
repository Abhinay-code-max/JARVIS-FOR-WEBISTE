"""
core/tool_gate.py
==================
Single chokepoint for every TOOL_DISPATCH call. main.py's interactive
_execute_tool (player=self.ui) and agent/executor.py's _call_tool
(player=None, always — see core/confirm.py) both route through
dispatch_tool() instead of indexing TOOL_DISPATCH directly, so policy
evaluation and audit logging happen in exactly one place instead of being
scattered as per-action-module CONFIRM calls.

DELEGATED_TOOLS keep their own internal CONFIRM.request() call(s) instead
of being gated here — see the constant's docstring for why those two
specifically can't be replaced by a single entry-level ask.
"""
from __future__ import annotations

import logging
import time

from core.tool_dispatch  import TOOL_DISPATCH
from core.confirm        import CONFIRM
from core.policy         import get_policy_level, ACTION_EXTRACTORS, ASK_AND_WAIT, HARD_DENY, AUTO_ALLOW, NOTIFY_ONLY
from core.task_approval  import TASK_APPROVAL
from core.db              import get_conn

_log = logging.getLogger("jarvis.tool_gate")

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


def dispatch_tool(
    tool: str,
    args: dict,
    player,
    speak,
    task_id: str | None = None,
    submitted_interactively: bool = True,
) -> str:
    """Evaluates permission_policy for (tool, action), gates as needed,
    then calls TOOL_DISPATCH[tool]. Never call this from the audio/
    transcript thread — same rule as CONFIRM.request()."""
    if tool not in TOOL_DISPATCH:
        raise ValueError(
            f"Unknown tool '{tool}' — no such tool exists. "
            f"Available tools: {sorted(TOOL_DISPATCH.keys())}"
        )

    action = _extract_action(tool, args)
    level  = get_policy_level(tool, action)

    if tool in DELEGATED_TOOLS:
        _log_decision(tool, action, level, "delegated", task_id)
        return TOOL_DISPATCH[tool](args, player, speak)

    if level == HARD_DENY:
        _log_decision(tool, action, level, "denied", task_id)
        return f"'{tool}' is not permitted."

    if level == AUTO_ALLOW:
        result = TOOL_DISPATCH[tool](args, player, speak)
        _log_decision(tool, action, level, "auto-allowed", task_id)
        return result

    if level == NOTIFY_ONLY:
        result = TOOL_DISPATCH[tool](args, player, speak)
        if player is not None:
            try:
                player.write_log(f"NOTICE: ran {tool}" + (f" ({action})" if action else ""))
            except Exception:
                pass
        _log_decision(tool, action, level, "notified", task_id)
        return result

    # ask-and-wait
    assert level == ASK_AND_WAIT
    prompt = f"I'm about to run {tool}" + (f" — {action}" if action else "") + "."

    if player is not None:
        approved = CONFIRM.request(player, prompt, speak=speak)
        outcome  = "approved" if approved else "denied"
    else:
        note = " (submitted interactively)" if submitted_interactively else " (system-submitted)"
        approved, outcome = TASK_APPROVAL.wait_for_approval(task_id, prompt + note)

    _log_decision(tool, action, level, outcome, task_id)
    if not approved:
        # Message text is the only signal agent/executor.py's
        # _approval_outcome() has to distinguish "timed out" from
        # "explicitly denied" (outcome itself isn't returned to the
        # caller) — keep these two phrasings and don't merge them.
        if outcome == "timeout":
            return f"Cancelled — '{tool}' timed out waiting for approval."
        return f"Cancelled — '{tool}' was not approved."
    return TOOL_DISPATCH[tool](args, player, speak)
