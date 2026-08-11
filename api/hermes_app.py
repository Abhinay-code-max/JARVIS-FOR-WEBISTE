"""
api/hermes_app.py
===================
Headless-extraction phase, Step 1.5: the FastAPI + WebSocket network
boundary letting EYV's backend agents call into JARVIS-XL's orchestration
core (core/tool_gate.dispatch_tool, agent/task_queue) over HTTP/WS
instead of only from the local PyQt6 UI in-process. Built on top of the
already-UI-independent core/ + agent/ packages (Step 1.1/1.4) — this
module itself imports nothing from main.py or ui.py.

*** STOP — do not bind this beyond 127.0.0.1 ***
Whether Hermes needs to be reachable from EYV's Railway deployment over
the public internet, or can stay on a private/local network, is an
explicit open question the plan requires Abhinay to answer directly
before this goes any further — see Step 1.5.x. Build and test everything
here against 127.0.0.1 (or FastAPI's TestClient, which binds no socket
at all) only. Nothing in this module decides or assumes an answer; if you
are the one about to pass host="0.0.0.0" or stand up a tunnel, stop and
get that confirmation first — this docstring existing is not that
confirmation.

Security mechanisms, each mirroring internal_tickets_api.py's pattern as
closely as this repo can verify without access to that file (EYV-side —
flagged, not silently assumed identical):
  - Auth is a router-level dependency (_authenticate, attached via
    APIRouter(dependencies=[Depends(_authenticate)])) — no REST route,
    present or future, can be registered on `rest_router` without going
    through it. This does NOT cover the one websocket route the same
    structural way (Starlette registers websocket routes differently
    from APIRoute-based HTTP routes, so router-level `dependencies=`
    doesn't reach them) — the /ws handler calls the exact same
    _resolve_caller_class() check as its first action instead. A second
    websocket route added later would need to remember to do the same;
    that's a real, narrower guarantee than the REST side gets, flagged
    here rather than glossed over.
  - Tokens are re-read from disk on every single request
    (core/hermes_auth.get_token(), never cached) — an instant-revocation
    property, same as internal_tickets_api.py's own
    _current_internal_ticket_api_token().
  - hmac.compare_digest() for the token comparison itself, never `==`.
  - Two-layer rate limiting: _AUTH_RATE_LIMITER gates the authentication
    attempt itself (successes and failures both, keyed by client IP,
    since a failed attempt has no caller_class yet) at 120/min;
    _ROUTE_RATE_LIMITER gates each (caller_class, route) pair at 60/min,
    counting only requests that already passed auth. Numbers match the
    plan's own suggested starting point from internal_tickets_api.py.
  - AuditedRoute (a custom APIRoute) logs every REST request's method,
    path, caller_class, outcome, and duration to log_events (via the
    same caller_id/caller_class/triggered_by columns Step 1.3 added) —
    structurally, at the route-class level, not a decorator someone
    could forget to add to a new endpoint.
  - WebSocket auth reads the Authorization header from the handshake
    (websocket.headers, available before .accept()), never a query
    param — query params land in logs/proxies in a way headers don't.

REST surface, deliberately minimal for this first cut (not a complete
task/approval/memory management API — flagged as an open scope question
for what's still needed, not silently declared finished):
  POST /tasks              submit a background task (gated by
                            core.policy.is_agent_task_allowed(); only
                            'desktop' can today, per Step 1.2b)
  GET  /tasks/{task_id}     status
  GET  /approvals           list pending ask-and-wait approvals
  POST /approvals/{id}      resolve one (approve/deny) — desktop caller
                            only, regardless of who triggered the
                            underlying task (Step 1.6b) — GET /approvals
                            (listing) is not similarly restricted, a
                            deliberate scope boundary, not an oversight
  GET  /memory              read-only snapshot (memory *write* semantics
                            over this boundary — which caller_class can
                            write which categories, if any — is an open
                            question left for a follow-up pass, not
                            included here)
  WS   /ws                  tool-call relay: {"tool": ..., "args": {...}}
                            -> dispatch_tool(..., caller_class=...) ->
                            {"result": ...}, one call per message
"""
from __future__ import annotations

import hmac
import json
import logging
import threading
import time
from typing import Callable

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketException
from fastapi.routing import APIRoute
from starlette.responses import Response
from starlette.status import WS_1008_POLICY_VIOLATION

from core import hermes_auth
from core.policy import CALLER_CLASSES, DESKTOP, is_agent_task_allowed
from core.tool_gate import dispatch_tool
from core.task_approval import TASK_APPROVAL

_log = logging.getLogger("jarvis.hermes_api")


# ---------------------------------------------------------------------------
# Rate limiting — simple in-memory fixed-window counters, no new
# dependency (matches this codebase's own preference for a hand-rolled
# mechanism sized to actual load over a heavier framework — see
# agent/task_queue.py's ProactiveLoop/TaskQueue for the same reasoning
# applied elsewhere). Per-process, in-memory: resets on restart, and
# doesn't coordinate across multiple processes — fine for a single-
# instance, localhost-only deployment; would need revisiting for
# anything else, which is exactly why binding beyond 127.0.0.1 is its
# own gated decision (Step 1.5.x).
# ---------------------------------------------------------------------------

class _RateLimiter:
    def __init__(self, max_calls: int, window_sec: float):
        self._max_calls = max_calls
        self._window    = window_sec
        self._lock      = threading.Lock()
        self._hits: dict[str, list[float]] = {}

    def allow(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            hits   = self._hits.setdefault(key, [])
            cutoff = now - self._window
            while hits and hits[0] < cutoff:
                hits.pop(0)
            if len(hits) >= self._max_calls:
                return False
            hits.append(now)
            return True


# 120/min on the auth gate itself (successes and failures both — a flood
# of wrong-token attempts must not be able to run unbounded even though
# none of them ever reach a real route), 60/min per (caller_class,
# route) once authenticated — the plan's own suggested starting point,
# lifted from internal_tickets_api.py.
_AUTH_RATE_LIMITER  = _RateLimiter(max_calls=120, window_sec=60.0)
_ROUTE_RATE_LIMITER = _RateLimiter(max_calls=60,  window_sec=60.0)


class HermesAuthError(Exception):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def _resolve_caller_class(auth_header: str | None, client_ip: str) -> str:
    """Shared core of both the REST and WebSocket auth checks. Raises
    HermesAuthError on any failure (missing header, malformed header,
    auth-gate rate limit, no matching token) — callers translate that
    into the right protocol-specific rejection. Never caches a token —
    core.hermes_auth.get_token() re-reads from disk every call, so a
    freshly re-minted or removed token takes effect on the very next
    request."""
    if not _AUTH_RATE_LIMITER.allow(client_ip):
        raise HermesAuthError("auth rate limit exceeded")

    if not auth_header or not auth_header.startswith("Bearer "):
        raise HermesAuthError("missing or malformed Authorization header")
    presented = auth_header[len("Bearer "):].strip()
    if not presented:
        raise HermesAuthError("empty bearer token")

    for caller_class in CALLER_CLASSES:
        stored = hermes_auth.get_token(caller_class)
        if stored is None:
            continue
        # hmac.compare_digest, never == — constant-time comparison,
        # the plan's own explicit requirement.
        if hmac.compare_digest(stored, presented):
            return caller_class

    raise HermesAuthError("no caller_class token matches")


async def _authenticate(request: Request) -> str:
    """The REST router-level dependency — see APIRouter(dependencies=...)
    below. Stores the resolved caller_class on request.state so
    AuditedRoute and the endpoint handlers can both read it without
    re-authenticating."""
    client_ip = request.client.host if request.client else "unknown"
    try:
        caller_class = _resolve_caller_class(request.headers.get("Authorization"), client_ip)
    except HermesAuthError as e:
        raise HTTPException(status_code=401, detail=str(e))

    route_key = f"{caller_class}:{request.url.path}"
    if not _ROUTE_RATE_LIMITER.allow(route_key):
        raise HTTPException(status_code=429, detail="rate limit exceeded for this route")

    request.state.caller_class = caller_class
    return caller_class


async def _require_desktop_caller(request: Request) -> None:
    """Additional router-level dependency for approval-*resolution*
    routes specifically (Step 1.6b) — an approval may only be resolved
    by the 'desktop' caller_class, regardless of which caller_class
    triggered the underlying task. This matches the approval-routing
    design locked in during Step 1.6: a single human-approval channel,
    not one whose trust forks by caller_class. Without this, once a
    second live token exists (Step 1.8), a service:support caller
    (empty tool allow-list — the least-trusted class by design) could
    resolve a service:bugfix task's pending approval, which would make
    the human-approval gate meaningless the moment that second token
    goes live.

    Runs after _authenticate (which resolves request.state.caller_class)
    — see approval_resolution_router below, where both are listed as
    router-level dependencies together, so any future route registered
    on that router gets this enforced structurally, not as a per-route
    check an endpoint author could forget to add."""
    caller_class = getattr(request.state, "caller_class", None)
    if caller_class != DESKTOP:
        raise HTTPException(status_code=403, detail="only the desktop caller may resolve approvals")


# ---------------------------------------------------------------------------
# Audited route class — every REST request logged structurally, not via
# a decorator an individual endpoint author could forget to add.
# ---------------------------------------------------------------------------

class AuditedRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_handler = super().get_route_handler()

        async def audited_handler(request: Request) -> Response:
            start = time.monotonic()
            outcome = "error"
            try:
                response = await original_handler(request)
                outcome = str(response.status_code)
                return response
            except HTTPException as e:
                outcome = str(e.status_code)
                raise
            finally:
                caller_class = getattr(request.state, "caller_class", None)
                duration_ms  = int((time.monotonic() - start) * 1000)
                _log.info(
                    "%s %s -> %s", request.method, request.url.path, outcome,
                    extra={
                        "caller_id":    caller_class,
                        "caller_class": caller_class,
                        "triggered_by": caller_class,
                        "duration_ms":  duration_ms,
                    },
                )

        return audited_handler


rest_router = APIRouter(route_class=AuditedRoute, dependencies=[Depends(_authenticate)])

# Separate router, not a per-route dependency added inline on
# resolve_approval — see _require_desktop_caller's own docstring for why
# this needs to be structural rather than something a future
# approval-resolution route could forget to repeat. _authenticate is
# listed again here (routers don't inherit sibling routers'
# dependencies) — order matters: it must resolve request.state.caller_class
# before _require_desktop_caller reads it.
approval_resolution_router = APIRouter(
    route_class=AuditedRoute,
    dependencies=[Depends(_authenticate), Depends(_require_desktop_caller)],
)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@rest_router.post("/tasks")
async def submit_task(body: dict, request: Request):
    caller_class = request.state.caller_class
    if not is_agent_task_allowed(caller_class):
        raise HTTPException(status_code=403, detail="agent_task submission is not permitted for this caller")

    goal = str(body.get("goal", "")).strip()
    if not goal:
        raise HTTPException(status_code=400, detail="'goal' is required")

    from agent.task_queue import get_queue, TaskPriority
    prio_map = {"high": TaskPriority.HIGH, "normal": TaskPriority.NORMAL, "low": TaskPriority.LOW}
    priority = prio_map.get(str(body.get("priority", "normal")).lower(), TaskPriority.NORMAL)

    task_id = get_queue().submit(
        goal=goal, priority=priority, submitted_interactively=False, caller_class=caller_class,
    )
    return {"task_id": task_id}


@rest_router.get("/tasks/{task_id}")
async def get_task(task_id: str):
    from agent.task_queue import get_queue
    status = get_queue().get_status(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail="unknown task_id")
    return status


@rest_router.get("/approvals")
async def list_approvals():
    return TASK_APPROVAL.list_pending()


@approval_resolution_router.post("/approvals/{approval_id}")
async def resolve_approval(approval_id: int, body: dict):
    approve = bool(body.get("approve", True))
    ok = TASK_APPROVAL.answer(approval_id, approve)
    if not ok:
        raise HTTPException(status_code=404, detail="approval isn't waiting on a decision")
    return {"approval_id": approval_id, "approved": approve}


@rest_router.get("/memory")
async def get_memory():
    from memory.memory_manager import load_memory
    return load_memory()


# ---------------------------------------------------------------------------
# WebSocket: tool-call relay
# ---------------------------------------------------------------------------

async def hermes_websocket(websocket: WebSocket) -> None:
    client_ip = websocket.client.host if websocket.client else "unknown"
    try:
        caller_class = _resolve_caller_class(websocket.headers.get("Authorization"), client_ip)
    except HermesAuthError as e:
        # Reject before .accept() — never establish the connection for an
        # unauthenticated caller.
        raise WebSocketException(code=WS_1008_POLICY_VIOLATION, reason=str(e))

    await websocket.accept()
    _log.info("WS connected", extra={"caller_id": caller_class, "caller_class": caller_class, "triggered_by": caller_class})

    try:
        while True:
            raw = await websocket.receive_text()
            route_key = f"{caller_class}:/ws"
            if not _ROUTE_RATE_LIMITER.allow(route_key):
                await websocket.send_json({"error": "rate limit exceeded"})
                continue

            try:
                payload = json.loads(raw)
                tool = payload["tool"]
                args = payload.get("args", {})
            except (json.JSONDecodeError, KeyError, TypeError):
                await websocket.send_json({"error": "expected {'tool': ..., 'args': {...}}"})
                continue

            start = time.monotonic()
            try:
                result = dispatch_tool(
                    tool, args, player=None, speak=None,
                    submitted_interactively=False, caller_class=caller_class,
                )
                outcome = "ok"
            except Exception as e:
                result  = str(e)
                outcome = "error"
            duration_ms = int((time.monotonic() - start) * 1000)
            _log.info(
                "WS tool_call %s -> %s", tool, outcome,
                extra={
                    "caller_id": caller_class, "caller_class": caller_class, "triggered_by": caller_class,
                    "duration_ms": duration_ms,
                },
            )
            await websocket.send_json({"result": result})
    except Exception:
        # Client disconnect (WebSocketDisconnect) or any other transport-
        # level error — nothing more to do, the connection is already
        # gone or going.
        pass


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Factory, not a module-level `app = FastAPI()` singleton — each
    call gets a fresh app (rate limiters are attached to the module,
    though, so tests that care about isolated rate-limit state should
    reset _AUTH_RATE_LIMITER/_ROUTE_RATE_LIMITER's ._hits directly; see
    tests/test_hermes_api.py)."""
    app = FastAPI(title="Hermes — JARVIS-XL network boundary")
    app.include_router(rest_router)
    app.include_router(approval_resolution_router)
    app.add_api_websocket_route("/ws", hermes_websocket)
    return app


# ---------------------------------------------------------------------------
# Local-only manual run entrypoint — 127.0.0.1 is hardcoded, not read from
# an env var/config value, specifically so this can't be silently flipped
# to 0.0.0.0 by a config change. Going beyond localhost is Step 1.5.x's
# explicit, still-open decision — see this module's docstring. Run with:
#   python -m api.hermes_app
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=8765)
