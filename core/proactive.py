"""
core/proactive.py
==================
Phase-1 proactive nudges: JARVIS surfacing things unprompted, sourced
entirely from data it already generates about itself (approvals, tasks).
No external event source (calendar, messages, CI, ...) — that's a
separate, unbuilt future phase.

Two trigger types only:
  - a background approval that's been pending too long (core/task_approval.py)
  - a background task that failed with an unhandled exception (agent/task_queue.py's
    _run_task except-block, which logs but never speaks — the one failure
    path in the app that today produces zero spoken feedback, ever, even
    when the task was submitted with a live speak callback)

Single daemon poll loop, not a job-scheduler library — this app has one
process, one user, and two trigger types; a sleep loop is the right size
(see agent/task_queue.py / core/task_approval.py for the same reasoning
applied to their own machinery).

Delivery reuses the existing TTS queue (JarvisXL.speak() / main.py) — the
same pathway the agent_task tool already uses to speak background-task
results unprompted. Nothing new there; what's new is deciding *when* to
call it, gated by:
  - not already speaking, not muted (JarvisXL's own live state)
  - quiet_hours (config/api_keys.json, optional; absent = disabled)
  - fire-once dedup per trigger (see proactive_nudges schema comment in
    core/db.py for why the two triggers need different dedup strategies)
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Callable

from config import load_config
from core.db import get_conn, get_step_outcomes, nudge_already_sent, record_nudge
from core.task_approval import TASK_APPROVAL

_log = logging.getLogger("jarvis.proactive")

POLL_INTERVAL_SEC = 60

# 15 min: half of core/task_approval.py's DEFAULT_TIMEOUT (30 min), so a
# nudge always lands with real runway left before the approval
# auto-times-out — not so short that a user who stepped away for a few
# minutes gets nudged over an ordinary delay.
STALE_APPROVAL_SEC = 900

TRIGGER_TASK_FAILED = "task_failed"


def _quiet_hours_active(cfg: dict, now: datetime | None = None) -> bool:
    """True if `now` (default: current time) falls inside the configured
    quiet_hours window. Config shape: {"quiet_hours": {"start": "HH:MM",
    "end": "HH:MM"}} in config/api_keys.json — absent or malformed means
    quiet hours are simply off, not an error."""
    qh = cfg.get("quiet_hours")
    if not isinstance(qh, dict) or not qh.get("start") or not qh.get("end"):
        return False
    try:
        start = datetime.strptime(qh["start"], "%H:%M").time()
        end   = datetime.strptime(qh["end"], "%H:%M").time()
    except (ValueError, TypeError):
        return False

    now_t = (now or datetime.now()).time()
    if start <= end:
        return start <= now_t < end
    return now_t >= start or now_t < end  # window wraps past midnight


class ProactiveLoop:
    def __init__(
        self,
        speak:       Callable[[str], None],
        is_speaking: Callable[[], bool],
        is_muted:    Callable[[], bool],
        now:         Callable[[], datetime] = datetime.now,
    ):
        self._speak       = speak
        self._is_speaking = is_speaking
        self._is_muted    = is_muted
        self._now         = now
        # In-memory only, deliberately — see core/db.py's proactive_nudges
        # schema comment. A restart wipes this set, but by then
        # _reconcile_orphaned_approvals() has already flipped every
        # leftover 'pending' row to 'expired', so nothing stale can
        # resurface through TASK_APPROVAL.list_pending() anyway.
        self._nudged_approvals: set[int] = set()
        self._running = False

    def _gated_speak(self, text: str) -> bool:
        """Returns True if the nudge was actually spoken (so callers only
        record dedup state on a real delivery, not a suppressed one)."""
        if self._is_speaking() or self._is_muted():
            return False
        if _quiet_hours_active(load_config(), now=self._now()):
            return False
        self._speak(text)
        return True

    # -- trigger: stale pending approval ------------------------------------

    def _check_stale_approvals(self) -> None:
        now = time.time()
        for row in TASK_APPROVAL.list_pending():
            approval_id = row["approval_id"]
            if approval_id in self._nudged_approvals:
                continue
            age = now - row["requested_at"]
            if age < STALE_APPROVAL_SEC:
                continue
            minutes = int(age // 60)
            msg = (
                f"Sir, approval request #{approval_id} has been waiting "
                f"{minutes} minutes: {row['prompt'][:120]}"
            )
            if self._gated_speak(msg):
                self._nudged_approvals.add(approval_id)

    # -- trigger: background task failed -------------------------------------

    def _check_failed_tasks(self) -> None:
        conn = get_conn()
        rows = conn.execute(
            "SELECT task_id, goal, error FROM tasks WHERE status = 'failed'"
        ).fetchall()

        for row in rows:
            task_id = row["task_id"]
            if nudge_already_sent(TRIGGER_TASK_FAILED, task_id):
                continue

            outcomes  = get_step_outcomes(task_id)
            last_step = outcomes[-1] if outcomes else None
            step_ctx  = f" while on \"{last_step['description']}\"" if last_step else ""
            error     = (row["error"] or "no further detail").strip()

            msg = (
                f"Sir, the background task \"{row['goal'][:60]}\" failed"
                f"{step_ctx}: {error[:150]}"
            )
            # If _gated_speak() suppressed it (quiet hours / already
            # speaking / muted), leave it un-dedup'd so the next tick
            # retries — tasks.status stays 'failed' forever, so there's no
            # narrow window to miss the way there is for approvals.
            if self._gated_speak(msg):
                record_nudge(TRIGGER_TASK_FAILED, task_id)

    def _tick(self) -> None:
        try:
            self._check_stale_approvals()
        except Exception:
            _log.warning("Stale-approval check failed", exc_info=True)
        try:
            self._check_failed_tasks()
        except Exception:
            _log.warning("Failed-task check failed", exc_info=True)

    def run(self) -> None:
        # Forces TaskQueue.start() (and its startup reconciliation pass —
        # see agent/task_queue.py) if nothing has submitted a background
        # task yet this run. Idempotent: a no-op if the queue is already
        # started.
        from agent.task_queue import get_queue
        get_queue()

        self._running = True
        while self._running:
            time.sleep(POLL_INTERVAL_SEC)
            self._tick()

    def stop(self) -> None:
        self._running = False
