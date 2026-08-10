"""
core/task_approval.py
======================
Approval gate for ask-and-wait policy decisions hit on an AgentExecutor
task-queue worker thread — i.e. player is None, which happens structurally
for every executor-originated tool call (see agent/executor.py's
_call_tool and core/confirm.py's docstring). There is no live human to
block on synchronously the way the interactive CONFIRM gate does, so this
gate pauses the task's own worker thread and waits for a human to resolve
it later, out of band, via the approve_task tool (main.py).

Unlike CONFIRM (a single global pending slot, answered from the audio/
transcript chokepoint), several background tasks can be paused on separate
approvals at once, so this is keyed per approval_id, not a single flag.

The blocked thread is always a per-task `AgentTask-{task_id}` daemon
thread (agent/task_queue.py), never the audio/transcript thread — so
blocking DB I/O here is safe, same rule CONFIRM.request() follows.
answer(), like CONFIRM.answer(), runs on the interactive text-command-loop
thread (via the approve_task tool), which also tolerates blocking I/O.

Known limitation: pending approvals live only in this in-memory registry.
A process restart loses them (the `approvals` row is left at outcome=
'pending' forever) exactly like an interrupted task's row is left at
status='running' forever — see agent/task_queue.py's
_reconcile_interrupted_tasks. Not solved here; the task itself becomes
unresumable on restart regardless, so there is no live wait to reconnect.
"""
from __future__ import annotations

import logging
import threading
import time

from core.db import get_conn

_log = logging.getLogger("jarvis.task_approval")

DEFAULT_TIMEOUT = 1800.0  # 30 min — no live human refreshing a 45s window here


class _PendingApproval:
    def __init__(self):
        self.event  = threading.Event()
        self.answer = False


class _TaskApprovalGate:
    def __init__(self):
        self._lock    = threading.Lock()
        self._pending: dict[int, _PendingApproval] = {}

    def wait_for_approval(
        self, task_id: str | None, prompt: str, timeout: float = DEFAULT_TIMEOUT,
    ) -> tuple[bool, str]:
        """Blocking. Inserts a pending `approvals` row, waits for a human to
        resolve it via answer(), returns (approved, outcome) where outcome
        is 'approved' | 'denied' | 'timeout'."""
        requested_at = time.time()
        conn = get_conn()
        with conn:
            cur = conn.execute(
                "INSERT INTO approvals (prompt, requested_at, answered_at, outcome, task_id) "
                "VALUES (?, ?, NULL, 'pending', ?)",
                (prompt, requested_at, task_id),
            )
            approval_id = cur.lastrowid

        pending = _PendingApproval()
        with self._lock:
            self._pending[approval_id] = pending

        _log.info(
            "Task %s awaiting approval #%d (timeout %ds): %s",
            task_id, approval_id, int(timeout), prompt[:120],
        )
        got_answer = pending.event.wait(timeout=timeout)

        with self._lock:
            self._pending.pop(approval_id, None)

        if not got_answer:
            outcome = "timeout"
        else:
            outcome = "approved" if pending.answer else "denied"

        with conn:
            conn.execute(
                "UPDATE approvals SET answered_at = ?, outcome = ? WHERE approval_id = ?",
                (time.time(), outcome, approval_id),
            )

        return outcome == "approved", outcome

    def answer(self, approval_id: int, approve: bool) -> bool:
        """Resolves a pending approval. Returns False if approval_id isn't
        (or is no longer) waiting — already answered, timed out, or never
        existed in this process."""
        with self._lock:
            pending = self._pending.get(approval_id)
        if not pending:
            return False
        pending.answer = approve
        pending.event.set()
        return True

    def list_pending(self) -> list[dict]:
        """Only rows this process is actually still blocked on — reads the
        in-memory registry's keys, not just `outcome='pending'` in the DB,
        so a restart-orphaned row (see module docstring) isn't listed as
        something approve_task could still resolve."""
        with self._lock:
            ids = list(self._pending.keys())
        if not ids:
            return []
        conn        = get_conn()
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT approval_id, prompt, task_id, requested_at FROM approvals "
            f"WHERE approval_id IN ({placeholders}) ORDER BY requested_at",
            ids,
        ).fetchall()
        return [dict(r) for r in rows]


TASK_APPROVAL = _TaskApprovalGate()
