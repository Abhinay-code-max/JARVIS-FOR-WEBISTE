"""
core/hermes_client.py
=======================
Headless-extraction phase, Step 1.7: the desktop app's own client of its
local Hermes server (api/hermes_app.py) — for the two operations that
have a genuine, faithful REST equivalent: submitting/polling an
agent_task, and listing/resolving approvals. See main.py's JarvisXL for
the caller.

Deliberately NOT used for the general TOOL_DISPATCH path (weather_report,
file_controller, etc.). api/hermes_app.py's WS relay calls
dispatch_tool(tool, args, player=None, speak=None, ...) unconditionally
— every ordinary tool call the desktop's own live voice/text loop makes
needs a real player (self.ui, for on-screen state and the ask-and-wait
confirmation flow) and a real speak callback (live TTS), neither of
which a WebSocket message can carry. Routing those through /ws as-is
would silently drop that wiring rather than migrate it faithfully, so
they stay in-process — see main.py's _execute_tool. This is a reported
gap in Step 1.5's surface, not something this module works around.

Every function here is fallible by design (network_api_enabled may be
off, fastapi/uvicorn may not be installed, the local server may not have
finished starting yet, the desktop token may never have been minted) —
callers catch HermesUnavailable and fall back to the pre-Step-1.7
in-process call, logging a warning so a persistently broken Hermes path
stays discoverable rather than silently masked.
"""
from __future__ import annotations

import logging

import requests

from core import hermes_auth
from core.policy import DESKTOP

_log = logging.getLogger("jarvis.hermes_client")

# Must match api/hermes_app.py's hardcoded __main__ bind — see that
# module's docstring for why going beyond 127.0.0.1 is a separate,
# still-open decision this client does not get to make.
BASE_URL = "http://127.0.0.1:8765"
_TIMEOUT = 5.0


class HermesUnavailable(Exception):
    """Raised for any failure reaching the local Hermes server or
    resolving a desktop token. Callers fall back to the in-process path
    on this — never retried in a loop here."""


def _headers() -> dict:
    token = hermes_auth.get_token(DESKTOP)
    if not token:
        raise HermesUnavailable(
            "no desktop Hermes token minted (run: python -m core.hermes_auth mint desktop)"
        )
    return {"Authorization": f"Bearer {token}"}


def submit_task(goal: str, priority: str = "normal") -> str:
    try:
        r = requests.post(
            f"{BASE_URL}/tasks", json={"goal": goal, "priority": priority},
            headers=_headers(), timeout=_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()["task_id"]
    except HermesUnavailable:
        raise
    except Exception as e:
        raise HermesUnavailable(str(e)) from e


def get_task_status(task_id: str) -> dict | None:
    """None means 'no such task' (404) — distinct from HermesUnavailable,
    which means 'couldn't even ask'."""
    try:
        r = requests.get(f"{BASE_URL}/tasks/{task_id}", headers=_headers(), timeout=_TIMEOUT)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except HermesUnavailable:
        raise
    except Exception as e:
        raise HermesUnavailable(str(e)) from e


def list_pending_approvals() -> list[dict]:
    try:
        r = requests.get(f"{BASE_URL}/approvals", headers=_headers(), timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except HermesUnavailable:
        raise
    except Exception as e:
        raise HermesUnavailable(str(e)) from e


def resolve_approval(approval_id: int, approve: bool) -> bool:
    """False means 'not waiting on a decision' (404) — distinct from
    HermesUnavailable, matching core.task_approval.TASK_APPROVAL.answer()'s
    own bool-return contract so main.py's fallback path is a drop-in."""
    try:
        r = requests.post(
            f"{BASE_URL}/approvals/{approval_id}", json={"approve": approve},
            headers=_headers(), timeout=_TIMEOUT,
        )
        if r.status_code == 404:
            return False
        r.raise_for_status()
        return True
    except HermesUnavailable:
        raise
    except Exception as e:
        raise HermesUnavailable(str(e)) from e
