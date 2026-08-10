"""
tests/test_permission_model.py
================================
Covers the two gaps closed after the initial permission-model
implementation:

  1. A background ask-and-wait approval that times out — the approvals
     row must land on a resolved, distinct 'timeout' status (never stuck
     at 'pending'), the in-memory registry entry must be cleaned up, and
     the owning task step must fail cleanly (not be silently logged as
     'done') without cascading into another blocking wait via the
     generic retry/replan-fix machinery.
  2. Restart reconciliation — a 'pending' approvals row left over from a
     previous process lifetime (no live worker thread can ever resolve
     it) must be reconciled to 'expired' by the same startup pass that
     already reconciles orphaned tasks.

Uses stdlib unittest only (no pytest in this environment) and redirects
core.db at a fresh temp sqlite file per test class so nothing here
touches the real data/jarvis.db.
"""
from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.db as db


def _use_temp_db() -> Path:
    """Points core.db at a fresh temp sqlite file. Must run before any
    thread's first get_conn() call — core.db's _local is a process-wide
    threading.local() created at import time, so this only isolates
    *this* test process, and only if called before any connection is
    opened."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="jarvis_test_db_"))
    db.DB_DIR  = tmp_dir
    db.DB_PATH = tmp_dir / "test.db"
    db._local  = threading.local()  # drop any connection opened before this call
    return db.DB_PATH


_use_temp_db()

import core.policy as policy         # noqa: E402
import core.task_approval as ta_mod  # noqa: E402
import core.tool_gate as gate        # noqa: E402
import core.tool_dispatch as tdisp   # noqa: E402
import agent.executor as executor    # noqa: E402
import agent.task_queue as task_queue  # noqa: E402


class TaskApprovalTimeoutTest(unittest.TestCase):
    """Layer 1: core/task_approval.py's Event.wait() timeout behavior,
    exercised for real (short-circuited duration, not mocked)."""

    def setUp(self):
        _use_temp_db()
        policy.seed_default_policy()
        self.gate_obj = ta_mod._TaskApprovalGate()  # fresh registry per test

    def test_timeout_resolves_cleanly_no_answer_given(self):
        approved, outcome = self.gate_obj.wait_for_approval(
            task_id="task-timeout-1", prompt="test: never answered", timeout=0.2,
        )

        self.assertFalse(approved)
        self.assertEqual(outcome, "timeout")

        # approvals row must be resolved, not stuck at 'pending'
        conn = db.get_conn()
        row = conn.execute(
            "SELECT outcome, answered_at, task_id FROM approvals WHERE task_id = ?",
            ("task-timeout-1",),
        ).fetchone()
        self.assertIsNotNone(row, "approvals row was never created")
        self.assertEqual(row["outcome"], "timeout")
        self.assertIsNotNone(row["answered_at"], "answered_at left NULL after timeout")

        # exactly one row — no duplicate insert from a retried/partial path
        count = conn.execute(
            "SELECT COUNT(*) FROM approvals WHERE task_id = ?", ("task-timeout-1",),
        ).fetchone()[0]
        self.assertEqual(count, 1)

        # registry entry must be cleaned up — nothing left to leak/answer late
        self.assertEqual(len(self.gate_obj._pending), 0)

    def test_answer_after_timeout_is_a_no_op(self):
        approved, outcome = self.gate_obj.wait_for_approval(
            task_id="task-timeout-2", prompt="test: answered too late", timeout=0.1,
        )
        self.assertEqual(outcome, "timeout")

        # a late answer() must not find anything to resolve (already popped)
        ok = self.gate_obj.answer(approval_id=self._last_approval_id("task-timeout-2"), approve=True)
        self.assertFalse(ok)

        # and the row must still read 'timeout', not flipped to 'approved'
        row = db.get_conn().execute(
            "SELECT outcome FROM approvals WHERE task_id = ?", ("task-timeout-2",),
        ).fetchone()
        self.assertEqual(row["outcome"], "timeout")

    def test_explicit_denial_is_distinct_from_timeout(self):
        result_holder = {}

        def _run():
            result_holder["r"] = self.gate_obj.wait_for_approval(
                task_id="task-denied-1", prompt="test: explicit deny", timeout=5.0,
            )

        t = threading.Thread(target=_run)
        t.start()
        time.sleep(0.1)
        approval_id = self._last_approval_id("task-denied-1")
        self.assertTrue(self.gate_obj.answer(approval_id, approve=False))
        t.join(timeout=5)

        approved, outcome = result_holder["r"]
        self.assertFalse(approved)
        self.assertEqual(outcome, "denied")  # not 'timeout' — must stay distinguishable

    def _last_approval_id(self, task_id: str) -> int:
        row = db.get_conn().execute(
            "SELECT approval_id FROM approvals WHERE task_id = ? ORDER BY approval_id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        return row["approval_id"]


class DispatchToolTimeoutTest(unittest.TestCase):
    """Layer 2: core/tool_gate.py's handling of a timeout outcome —
    message text and policy_decisions logging. TASK_APPROVAL.wait_for_approval
    is stubbed to return a timeout immediately (already covered for real in
    TaskApprovalTimeoutTest above) so this isolates dispatch_tool's own
    logic without another real wait."""

    def setUp(self):
        _use_temp_db()
        policy.seed_default_policy()
        self._real_wait = ta_mod.TASK_APPROVAL.wait_for_approval
        ta_mod.TASK_APPROVAL.wait_for_approval = lambda task_id, prompt, timeout=None: (False, "timeout")
        self._orig_tool = tdisp.TOOL_DISPATCH.get("file_controller")
        tdisp.TOOL_DISPATCH["file_controller"] = lambda args, player, speak: "SHOULD NOT RUN"

    def tearDown(self):
        ta_mod.TASK_APPROVAL.wait_for_approval = self._real_wait
        if self._orig_tool is not None:
            tdisp.TOOL_DISPATCH["file_controller"] = self._orig_tool

    def test_timeout_message_is_distinct_and_tool_never_runs(self):
        result = gate.dispatch_tool(
            "file_controller", {"action": "delete", "name": "x"},
            player=None, speak=None, task_id="task-dispatch-timeout",
        )
        self.assertIn("timed out waiting for approval", result)
        self.assertNotEqual(result, "SHOULD NOT RUN")

    def test_timeout_logged_distinctly_in_policy_decisions(self):
        gate.dispatch_tool(
            "file_controller", {"action": "delete", "name": "x"},
            player=None, speak=None, task_id="task-dispatch-timeout-2",
        )
        row = db.get_conn().execute(
            "SELECT level_evaluated, outcome FROM policy_decisions WHERE task_id = ?",
            ("task-dispatch-timeout-2",),
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["level_evaluated"], "ask-and-wait")
        self.assertEqual(row["outcome"], "timeout")  # not the generic 'denied'


class ExecutorApprovalOutcomeTest(unittest.TestCase):
    """Layer 3a: agent/executor.py's _approval_outcome() correctly tells
    timeout apart from an explicit denial, from dispatch_tool's exact
    message strings."""

    def test_timeout_message_parses_as_timeout(self):
        msg = "Cancelled — 'file_controller' timed out waiting for approval."
        self.assertEqual(executor._approval_outcome(msg), "timeout")

    def test_denied_message_parses_as_denied(self):
        msg = "Cancelled — 'file_controller' was not approved."
        self.assertEqual(executor._approval_outcome(msg), "denied")

    def test_ordinary_result_is_not_an_approval_outcome(self):
        self.assertIsNone(executor._approval_outcome("Deleted x.txt."))
        self.assertIsNone(executor._approval_outcome(None))


class ExecutorStepFailsCleanlyOnTimeoutTest(unittest.TestCase):
    """Layer 3b: the full step loop treats an approval timeout as an
    immediate, distinctly-tagged failure — not a silently-'done' step,
    and not something that cascades into analyze_error/retry/recovery
    (which would itself risk another blocking wait). create_plan and
    dispatch_tool are stubbed so this runs deterministically with no real
    LLM calls and no real 30-minute wait."""

    def setUp(self):
        _use_temp_db()
        policy.seed_default_policy()

        self._orig_create_plan = executor.create_plan
        self._orig_replan      = executor.replan
        self._orig_dispatch    = executor.dispatch_tool

        def _fake_plan(goal, task_id=None):
            return {"steps": [{
                "step": 1, "tool": "file_controller",
                "description": "delete a file", "parameters": {"action": "delete"},
            }]}

        def _fake_replan(goal, completed_steps, failed_step, failed_error, task_id=None):
            # No real LLM call — the outer task-level replanner isn't what
            # this test is about; returning no steps ends the task cleanly
            # on its next loop iteration instead of retrying the step again.
            return {"steps": []}

        def _fake_dispatch(tool, args, player, speak, task_id=None, submitted_interactively=True):
            return f"Cancelled — '{tool}' timed out waiting for approval."

        executor.create_plan   = _fake_plan
        executor.replan        = _fake_replan
        executor.dispatch_tool = _fake_dispatch

    def tearDown(self):
        executor.create_plan   = self._orig_create_plan
        executor.replan        = self._orig_replan
        executor.dispatch_tool = self._orig_dispatch

    def test_timed_out_step_fails_task_without_retry_storm(self):
        # task_events.task_id is a real FK into tasks(task_id) — insert the
        # parent row a normal TaskQueue.submit() would have created.
        conn = db.get_conn()
        with conn:
            conn.execute(
                "INSERT INTO tasks (task_id, goal, priority, status, result, error, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("task-exec-timeout", "delete a file", 2, "running", None, "", time.time(), time.time()),
            )

        calls = {"n": 0}
        real_call_tool = executor._call_tool

        def _counting_call_tool(*a, **kw):
            calls["n"] += 1
            return real_call_tool(*a, **kw)

        executor._call_tool = _counting_call_tool
        try:
            ex = executor.AgentExecutor()
            result = ex.execute(goal="delete a file", task_id="task-exec-timeout", submitted_interactively=False)
        finally:
            executor._call_tool = real_call_tool

        # exactly one attempt — the approval-outcome short-circuit must
        # bypass the attempt<=3 retry loop and analyze_error/REPLAN's
        # generate_fix cascade entirely, not retry a blocking wait blindly
        self.assertEqual(calls["n"], 1)
        self.assertIsInstance(result, str)

        # the step's own "failed" event, not the outer task-level
        # "replanned" event logged afterwards for the same step_num
        row = db.get_conn().execute(
            "SELECT status, detail FROM task_events "
            "WHERE task_id = ? AND step_num = 1 AND status = 'failed' "
            "ORDER BY event_id DESC LIMIT 1",
            ("task-exec-timeout",),
        ).fetchone()
        self.assertIsNotNone(row, "no 'failed' task_event was logged for the step")
        self.assertIn("approval_timeout", row["detail"])

        # must NOT have gone through the generic 'attempt_failed' retry
        # path — that would mean it fell into the except-Exception branch
        # (analyze_error/RETRY) instead of the immediate short-circuit
        attempt_failed_count = db.get_conn().execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND status = 'attempt_failed'",
            ("task-exec-timeout",),
        ).fetchone()[0]
        self.assertEqual(attempt_failed_count, 0)


class OrphanedApprovalReconciliationTest(unittest.TestCase):
    """Restart reconciliation: a 'pending' row with no live worker thread
    behind it gets marked 'expired', in the same pass that already
    reconciles orphaned tasks."""

    def setUp(self):
        _use_temp_db()
        policy.seed_default_policy()

    def _insert_approval(self, outcome: str, task_id: str) -> int:
        conn = db.get_conn()
        with conn:
            cur = conn.execute(
                "INSERT INTO approvals (prompt, requested_at, answered_at, outcome, task_id) "
                "VALUES (?, ?, NULL, ?, ?)",
                (f"test prompt for {task_id}", time.time(), outcome, task_id),
            )
        return cur.lastrowid

    def test_pending_row_reconciled_to_expired(self):
        approval_id = self._insert_approval("pending", "orphaned-task-1")

        task_queue._reconcile_orphaned_approvals()

        row = db.get_conn().execute(
            "SELECT outcome, answered_at FROM approvals WHERE approval_id = ?", (approval_id,),
        ).fetchone()
        self.assertEqual(row["outcome"], "expired")
        self.assertIsNotNone(row["answered_at"])

    def test_resolved_rows_are_left_untouched(self):
        approved_id = self._insert_approval("approved", "resolved-task-1")
        denied_id   = self._insert_approval("denied", "resolved-task-2")
        timeout_id  = self._insert_approval("timeout", "resolved-task-3")

        task_queue._reconcile_orphaned_approvals()

        conn = db.get_conn()
        for aid, expected in ((approved_id, "approved"), (denied_id, "denied"), (timeout_id, "timeout")):
            row = conn.execute("SELECT outcome FROM approvals WHERE approval_id = ?", (aid,)).fetchone()
            self.assertEqual(row["outcome"], expected)

    def test_wired_into_taskqueue_start(self):
        """TaskQueue.start() must call the same reconciliation pass that
        handles orphaned tasks — not a separate one-off path someone has
        to remember to call."""
        calls = {"tasks": 0, "approvals": 0}
        orig_tasks     = task_queue._reconcile_interrupted_tasks
        orig_approvals = task_queue._reconcile_orphaned_approvals

        def _count_tasks():
            calls["tasks"] += 1

        def _count_approvals():
            calls["approvals"] += 1

        task_queue._reconcile_interrupted_tasks  = _count_tasks
        task_queue._reconcile_orphaned_approvals = _count_approvals
        try:
            q = task_queue.TaskQueue()
            q.start()
            q.stop()
        finally:
            task_queue._reconcile_interrupted_tasks  = orig_tasks
            task_queue._reconcile_orphaned_approvals = orig_approvals

        self.assertEqual(calls["tasks"], 1)
        self.assertEqual(calls["approvals"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
