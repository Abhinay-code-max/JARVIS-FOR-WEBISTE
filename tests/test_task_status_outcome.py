"""
tests/test_task_status_outcome.py
===================================
Headless-extraction phase, Step 1.6c: agent/task_queue.py's Task.status
must reflect a real denial/hard-deny outcome, not silently read as
'completed' just because AgentExecutor.execute() returned without
raising.

Same shape as the earlier _short_circuit_reason() hard-deny bug from the
adversarial-benchmark work: the real safety property already held (the
tool genuinely never ran) — this is about the AUDIT TRAIL / status
consumers not being misled into thinking the task succeeded when it
didn't. Fixed via agent/task_queue.py's _last_terminal_failure_detail(),
which checks the real, durable outcome (core.db.get_step_outcomes(), the
same helper _summarize() itself trusts) rather than trusting a lack of
exception as success.

Redirects core.db at a fresh temp sqlite file per test, same pattern as
every other module in this package.
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
    tmp_dir = Path(tempfile.mkdtemp(prefix="jarvis_test_db_"))
    db.DB_DIR  = tmp_dir
    db.DB_PATH = tmp_dir / "test.db"
    db._local  = threading.local()
    return db.DB_PATH


_use_temp_db()

import core.policy        as policy         # noqa: E402
import core.tool_dispatch  as tdisp         # noqa: E402
import core.task_approval  as ta_mod        # noqa: E402
import agent.executor      as executor      # noqa: E402
import agent.task_queue    as task_queue    # noqa: E402


def _insert_task(task_id: str, goal: str = "test goal"):
    conn = db.get_conn()
    with conn:
        conn.execute(
            "INSERT INTO tasks (task_id, goal, priority, status, result, error, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (task_id, goal, 2, "running", None, "", time.time(), time.time()),
        )


class LastTerminalFailureDetailUnitTest(unittest.TestCase):
    """Direct tests of the real function against real task_events rows —
    no TaskQueue/executor machinery involved, just the outcome-reading
    logic itself."""

    def setUp(self):
        _use_temp_db()

    def test_no_outcomes_returns_none(self):
        _insert_task("t-empty")
        self.assertIsNone(task_queue._last_terminal_failure_detail("t-empty"))

    def test_last_segment_done_returns_none(self):
        _insert_task("t-done")
        db.log_task_event("t-done", 1, "weather_report", "x", "started")
        db.log_task_event("t-done", 1, "weather_report", "x", "done", "Sunny.")
        self.assertIsNone(task_queue._last_terminal_failure_detail("t-done"))

    def test_last_segment_skipped_returns_none(self):
        _insert_task("t-skip")
        db.log_task_event("t-skip", 1, "send_message", "x", "started")
        db.log_task_event("t-skip", 1, "send_message", "x", "attempt_failed", "attempt 1")
        db.log_task_event("t-skip", 1, "send_message", "x", "skipped", "skipped (non-critical)")
        self.assertIsNone(task_queue._last_terminal_failure_detail("t-skip"))

    def test_last_segment_failed_returns_its_detail(self):
        _insert_task("t-fail")
        db.log_task_event("t-fail", 1, "code_helper", "x", "started")
        db.log_task_event("t-fail", 1, "code_helper", "x", "failed", "approval_denied: Cancelled — 'code_helper' was not approved.")
        detail = task_queue._last_terminal_failure_detail("t-fail")
        self.assertIsNotNone(detail)
        self.assertIn("approval_denied", detail)

    def test_hard_deny_detail_is_recognized_too(self):
        """The fix is reason-agnostic — it checks task_events.status,
        not which specific short-circuit reason produced it — so this
        must hold for hard_denied exactly like approval_denied."""
        _insert_task("t-harddeny")
        db.log_task_event("t-harddeny", 1, "_synthetic_tool", "x", "started")
        db.log_task_event("t-harddeny", 1, "_synthetic_tool", "x", "failed", "hard_denied: '_synthetic_tool' is not permitted.")
        detail = task_queue._last_terminal_failure_detail("t-harddeny")
        self.assertIsNotNone(detail)
        self.assertIn("hard_denied", detail)

    def test_recovered_via_replan_returns_none(self):
        """The contrast case the fix must not break: a step that failed
        but was later followed by a real 'done' segment (a successful
        replan-recovery) must NOT be reported as a failure — the task
        genuinely succeeded, just not on the first attempt."""
        _insert_task("t-recovered")
        db.log_task_event("t-recovered", 1, "code_helper", "x", "started")
        db.log_task_event("t-recovered", 1, "code_helper", "x", "failed", "approval_denied: ...")
        # Replan re-numbers from 1 again — a NEW segment for a DIFFERENT,
        # now-allowed tool.
        db.log_task_event("t-recovered", 1, "reminder", "x (retry)", "started")
        db.log_task_event("t-recovered", 1, "reminder", "x (retry)", "done", "Reminder set.")
        self.assertIsNone(task_queue._last_terminal_failure_detail("t-recovered"))


class TaskStatusRealEndToEndTest(unittest.TestCase):
    """Real TaskQueue + real AgentExecutor + real dispatch_tool()/
    TASK_APPROVAL, matching every other real-mechanism test in this
    phase. create_plan/replan stubbed (planning/LLM machinery, not what
    this is testing) — dispatch_tool and the approval gate are real."""

    def setUp(self):
        _use_temp_db()
        self._orig_create_plan = executor.create_plan
        self._orig_replan      = executor.replan

    def tearDown(self):
        executor.create_plan = self._orig_create_plan
        executor.replan      = self._orig_replan

    def _wait_for_pending_approval(self, task_id: str, timeout: float = 5.0) -> dict | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            for row in ta_mod.TASK_APPROVAL.list_pending():
                if row["task_id"] == task_id:
                    return row
            time.sleep(0.02)
        return None

    def _wait_for_terminal(self, q: "task_queue.TaskQueue", task_id: str, timeout: float = 5.0) -> dict:
        deadline = time.time() + timeout
        status = None
        while time.time() < deadline:
            status = q.get_status(task_id)
            if status and status["status"] in ("completed", "failed", "cancelled"):
                return status
            time.sleep(0.02)
        self.fail(f"task {task_id} never reached a terminal status within {timeout}s: {status}")

    def test_denied_step_produces_failed_task_status_with_error_detail(self):
        executor.create_plan = lambda goal, task_id=None: {"steps": [
            {"step": 1, "tool": "code_helper", "description": "x", "parameters": {"action": "run", "file_path": "x.py"}},
        ]}
        executor.replan = lambda goal, completed_steps, failed_step, error, task_id=None: {"steps": []}
        orig_tool = tdisp.TOOL_DISPATCH.get("code_helper")
        tdisp.TOOL_DISPATCH["code_helper"] = lambda args, player, speak: "SHOULD NEVER RUN"

        q = task_queue.TaskQueue()
        q.start()
        try:
            task_id  = q.submit(goal="do a gated thing", submitted_interactively=False, caller_class=policy.SERVICE_BUGFIX)
            approval = self._wait_for_pending_approval(task_id)
            self.assertIsNotNone(approval)
            ta_mod.TASK_APPROVAL.answer(approval["approval_id"], approve=False)

            status = self._wait_for_terminal(q, task_id)
        finally:
            q.stop()
            if orig_tool is not None:
                tdisp.TOOL_DISPATCH["code_helper"] = orig_tool

        # Expected (Step 1.6c): not silently 'completed'.
        self.assertEqual(status["status"], "failed", status)
        self.assertIn("approval_denied", status["error"])

        # And durable, not just the in-memory Task object.
        row = db.get_conn().execute("SELECT status, error FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        self.assertEqual(row["status"], "failed")
        self.assertIn("approval_denied", row["error"])

    def test_approved_step_still_produces_completed_status(self):
        """Contrast case — the fix must not turn every ask-and-wait step
        into 'failed'; only a genuine denial/hard-deny the task actually
        gave up on."""
        executor.create_plan = lambda goal, task_id=None: {"steps": [
            {"step": 1, "tool": "code_helper", "description": "x", "parameters": {"action": "run", "file_path": "x.py"}},
        ]}
        orig_tool = tdisp.TOOL_DISPATCH.get("code_helper")
        tdisp.TOOL_DISPATCH["code_helper"] = lambda args, player, speak: "ran successfully"

        q = task_queue.TaskQueue()
        q.start()
        try:
            task_id  = q.submit(goal="do a gated thing", submitted_interactively=False, caller_class=policy.DESKTOP)
            approval = self._wait_for_pending_approval(task_id)
            self.assertIsNotNone(approval)
            ta_mod.TASK_APPROVAL.answer(approval["approval_id"], approve=True)

            status = self._wait_for_terminal(q, task_id)
        finally:
            q.stop()
            if orig_tool is not None:
                tdisp.TOOL_DISPATCH["code_helper"] = orig_tool

        self.assertEqual(status["status"], "completed", status)

    def test_hard_denied_step_also_produces_failed_task_status(self):
        """Generalization check: the fix isn't approval-denial-specific
        — a hard-denied step (no approval gate involved at all) must
        equally not read as 'completed'."""
        synthetic_tool = "_synthetic_hard_deny_tool_1_6c"
        orig_tool = tdisp.TOOL_DISPATCH.get(synthetic_tool)
        calls = []
        tdisp.TOOL_DISPATCH[synthetic_tool] = lambda args, player, speak: (calls.append(args), "SHOULD NEVER RUN")[1]
        conn = db.get_conn()
        with conn:
            conn.execute(
                "INSERT INTO permission_policy (tool_name, action, level, caller_class) VALUES (?, NULL, ?, ?)",
                (synthetic_tool, policy.HARD_DENY, policy.DESKTOP),
            )

        executor.create_plan = lambda goal, task_id=None: {"steps": [
            {"step": 1, "tool": synthetic_tool, "description": "x", "parameters": {}},
        ]}
        executor.replan = lambda goal, completed_steps, failed_step, error, task_id=None: {"steps": []}

        q = task_queue.TaskQueue()
        q.start()
        try:
            task_id = q.submit(goal="try the hard-denied tool", submitted_interactively=False, caller_class=policy.DESKTOP)
            status  = self._wait_for_terminal(q, task_id)
        finally:
            q.stop()
            if orig_tool is not None:
                tdisp.TOOL_DISPATCH[synthetic_tool] = orig_tool
            else:
                tdisp.TOOL_DISPATCH.pop(synthetic_tool, None)

        self.assertEqual(status["status"], "failed", status)
        self.assertIn("hard_denied", status["error"])
        self.assertEqual(calls, [], "hard-denied tool was actually invoked")


if __name__ == "__main__":
    unittest.main(verbosity=2)
