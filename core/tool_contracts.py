"""
core/tool_contracts.py
========================
Central registry of per-tool contracts: input schema, output schema,
execution timeout, retry eligibility, and a descriptive risk level — for
every TOOL_DISPATCH tool. Enforced from core/tool_gate.py's dispatch_tool(),
the single chokepoint both main.py (interactive) and agent/executor.py
(background tasks) already route every tool call through.

Deliberately does NOT store a permission level — core/policy.py's
permission_policy table is the only source of truth for that; dispatch_tool()
keeps querying it directly. Duplicating it here would just create a second
place it could drift from.

risk_level is descriptive/audit metadata only in this phase — it does not
gate anything and does not influence `retryable`. `retryable` is set
per-tool purely on idempotency: is it safe to silently re-run this exact
call after a transient failure, or could a blind retry double-send a
message, double-click something, or re-run a partially-completed mutation?
Contracts are per-TOOL, not per-action (unlike permission_policy, which
does split some tools by action) — timeout_seconds and retryable are sized
to each tool's *worst-case* action, which makes them more generous than
the tool's typical action needs. That's a deliberate, safer-direction
coarseness: an oversized timeout never kills a call that would have
succeeded; an undersized one can.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from core.tool_declarations import get_declaration

_log = logging.getLogger("jarvis.tool_contracts")

DEFAULT_TIMEOUT_SECONDS = 120.0  # matches core/llm_client.py's own default call timeout

_STRING_OUTPUT_SCHEMA = {"type": "string"}  # every tool returns plain text today — see module docstring


@dataclass(frozen=True)
class ToolContract:
    tool_name:       str
    input_schema:    dict
    output_schema:   dict = field(default_factory=lambda: dict(_STRING_OUTPUT_SCHEMA))
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    retryable:       bool = True
    risk_level:      str  = "medium"   # low | medium | high | critical — descriptive/audit only, no behavioral effect
    # False only for tools whose internal blocking calls can't be safely
    # abandoned by dispatch_tool()'s ThreadPoolExecutor timeout wrapper —
    # see the thread-affinity investigation in core/tool_gate.py's
    # docstring for why youtube_video is the one exception.
    enforce_timeout: bool = True


def _input_schema_for(tool_name: str) -> dict:
    decl = get_declaration(tool_name)
    if decl is None:
        raise ValueError(f"No TOOL_DECLARATIONS entry for '{tool_name}' — contract can't derive input_schema.")
    return decl["parameters"]


def _contract(
    tool_name: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    retryable: bool = True,
    risk_level: str = "medium",
    enforce_timeout: bool = True,
) -> ToolContract:
    return ToolContract(
        tool_name       = tool_name,
        input_schema    = _input_schema_for(tool_name),
        timeout_seconds = timeout_seconds,
        retryable       = retryable,
        risk_level      = risk_level,
        enforce_timeout = enforce_timeout,
    )


TOOL_CONTRACTS: dict[str, ToolContract] = {
    # ── low risk, idempotent — retryable, tight-ish timeouts ───────────────
    "weather_report": _contract("weather_report", timeout_seconds=20,  retryable=True, risk_level="low"),
    "web_search":     _contract("web_search",     timeout_seconds=150, retryable=True, risk_level="low"),
    # request timeout=10 internally; DDGS itself has no explicit bound at
    # our call site + the LLM summarize call defaults to 120s — sized to
    # clear both with margin.
    "open_app":       _contract("open_app",       timeout_seconds=20,  retryable=True, risk_level="low"),
    "screen_process":  _contract("screen_process",  timeout_seconds=150, retryable=True, risk_level="low"),
    # vision call defaults to 120s internally.
    "daily_briefing": _contract("daily_briefing", timeout_seconds=60,  retryable=True, risk_level="low"),
    "flight_finder":  _contract("flight_finder",  timeout_seconds=90,  retryable=True, risk_level="low"),

    # youtube_video: retryable (idempotent — re-searching/re-summarizing is
    # harmless) but enforce_timeout=False — see the thread-affinity
    # investigation: _ask_for_url()'s tkinter dialog can't be safely
    # abandoned by a generic executor timeout without corrupting the
    # process-global tk._default_root for every later call.
    "youtube_video": _contract(
        "youtube_video", timeout_seconds=150, retryable=True, risk_level="low", enforce_timeout=False,
    ),

    # ── medium risk, still safely retryable (mostly read/analyze) ──────────
    # file_processor's one ask-and-wait action (archive extract) has minor
    # non-idempotence (re-extracts over the same destination) but no
    # external reach — outweighed by the tool's dominant read/analyze use.
    # ffmpeg convert/compress already bounds itself at 1800s internally —
    # this must clear that with margin or the wrapper would kill a
    # legitimately long-but-succeeding conversion.
    "file_processor": _contract("file_processor", timeout_seconds=1900, retryable=True, risk_level="medium"),

    # ── high risk, NOT safely retryable (real mutation / UI control) ───────
    "file_controller":   _contract("file_controller",   timeout_seconds=60,  retryable=False, risk_level="high"),
    "computer_settings": _contract("computer_settings", timeout_seconds=30,  retryable=False, risk_level="high"),
    "computer_control":  _contract("computer_control",  timeout_seconds=150, retryable=False, risk_level="high"),
    "browser_control":   _contract("browser_control",   timeout_seconds=90,  retryable=False, risk_level="high"),
    "game_updater":      _contract("game_updater",      timeout_seconds=60,  retryable=False, risk_level="high"),
    "desktop_control":   _contract("desktop_control",   timeout_seconds=150, retryable=False, risk_level="high"),
    "vision_fix_code":   _contract("vision_fix_code",   timeout_seconds=150, retryable=False, risk_level="high"),

    # code_helper/dev_agent: timeout sized to their worst action (code_helper's
    # multi-attempt "build" loop; dev_agent's multi-file generation + install
    # + up to 5 fix attempts), not their typical one — see module docstring.
    "code_helper": _contract("code_helper", timeout_seconds=600, retryable=False, risk_level="high"),
    "dev_agent":   _contract("dev_agent",   timeout_seconds=900, retryable=False, risk_level="high"),

    # reminder: writes a notify script to disk + registers a real OS-level
    # scheduled task/cron/launchd job — same risk class as file_controller's
    # write/create_file, not idempotent-safe to blind-retry (a retry after
    # a false-negative failure read would double-schedule the reminder).
    # schtasks/launchctl/systemd-run/at calls have no internal timeout of
    # their own (flagged in the subprocess-timeout follow-up), but are
    # normally fast local commands — 30s outer bound.
    "reminder": _contract("reminder", timeout_seconds=30, retryable=False, risk_level="medium"),

    # ── critical risk — irreversible and externally visible ────────────────
    "send_message": _contract("send_message", timeout_seconds=30, retryable=False, risk_level="critical"),
}


def get_contract(tool_name: str) -> ToolContract:
    """Falls back to a conservative default for any tool with no explicit
    entry above (a new TOOL_DISPATCH tool added without a contract) —
    non-retryable, default timeout, medium risk, rather than silently
    behaving as if unrestricted. Doesn't require a TOOL_DECLARATIONS entry
    to exist either (an empty/permissive schema — no required fields,
    nothing to type-check — rather than raising): dispatch_tool() already
    guarantees `tool_name` is a real TOOL_DISPATCH key before this is ever
    called, so the only way to land here is a tool that's missing from
    TOOL_CONTRACTS specifically, not from TOOL_DISPATCH."""
    contract = TOOL_CONTRACTS.get(tool_name)
    if contract is not None:
        return contract
    _log.warning("No explicit contract for tool '%s' — using conservative fallback.", tool_name)
    return ToolContract(
        tool_name       = tool_name,
        input_schema    = {"properties": {}, "required": []},
        timeout_seconds = DEFAULT_TIMEOUT_SECONDS,
        retryable       = False,
        risk_level      = "medium",
    )
