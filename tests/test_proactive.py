"""
tests/test_proactive.py
========================
Phase-1 proactive nudges (core/proactive.py): the two trigger types, the
speaking/muted/quiet-hours gate, and the dedup story for each trigger
(durable table for failed tasks, in-memory-only for stale approvals — see
core/db.py's proactive_nudges schema comment for why they differ).

Redirects core.db at a fresh temp sqlite file, same pattern as the other
test modules in this package, so nothing here touches data/jarvis.db.
"""
from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime
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

import core.proactive as proactive          # noqa: E402
from core.task_approval import TASK_APPROVAL, _PendingApproval  # noqa: E402


def _insert_task(task_id: str, status: str, error: str = "", goal: str = "test goal"):
    conn = db.get_conn()
    with conn:
        conn.execute(
            "INSERT INTO tasks (task_id, goal, priority, status, result, error, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (task_id, goal, 2, status, None, error, time.time(), time.time()),
        )


def _insert_pending_approval(approval_id: int, prompt: str, requested_at: float, task_id: str = "t1"):
    """Mirrors what TASK_APPROVAL.wait_for_approval() does, but with a
    caller-chosen approval_id/requested_at so tests can backdate it past
    the staleness threshold without a real 15-minute sleep."""
    conn = db.get_conn()
    with conn:
        conn.execute(
            "INSERT INTO approvals (approval_id, prompt, requested_at, answered_at, outcome, task_id) "
            "VALUES (?, ?, ?, NULL, 'pending', ?)",
            (approval_id, prompt, requested_at, task_id),
        )
    TASK_APPROVAL._pending[approval_id] = _PendingApproval()


class _RecordingLoop(proactive.ProactiveLoop):
    """Same gating logic as the real loop, but records what actually got
    spoken instead of needing a live JarvisXL/TTS stack."""

    def __init__(self, is_speaking=lambda: False, is_muted=lambda: False,
                 now=datetime.now):
        self.spoken: list[str] = []
        super().__init__(speak=self.spoken.append, is_speaking=is_speaking,
                          is_muted=is_muted, now=now)


class ProactiveTestCase(unittest.TestCase):
    def setUp(self):
        _use_temp_db()
        TASK_APPROVAL._pending.clear()


class StaleApprovalTriggerTest(ProactiveTestCase):
    def test_not_yet_stale_does_not_nudge(self):
        _insert_pending_approval(1, "delete these files", time.time() - 60)
        loop = _RecordingLoop()
        loop._check_stale_approvals()
        self.assertEqual(loop.spoken, [])

    def test_past_threshold_nudges_once(self):
        _insert_pending_approval(
            1, "delete these files",
            time.time() - proactive.STALE_APPROVAL_SEC - 1,
        )
        loop = _RecordingLoop()
        loop._check_stale_approvals()
        self.assertEqual(len(loop.spoken), 1)
        self.assertIn("delete these files", loop.spoken[0])

    def test_second_tick_does_not_renudge_the_same_approval(self):
        _insert_pending_approval(
            1, "delete these files",
            time.time() - proactive.STALE_APPROVAL_SEC - 1,
        )
        loop = _RecordingLoop()
        loop._check_stale_approvals()
        loop._check_stale_approvals()
        self.assertEqual(len(loop.spoken), 1)

    def test_resolved_approval_stops_matching_without_any_dedup_state(self):
        """The natural-state-exit claim from the plan: once a human
        resolves an approval via the real wait_for_approval()/answer()
        pair (the waiting thread pops itself from the in-memory pending
        registry on wake, same as production), list_pending() stops
        returning it — no durable table needed for this trigger."""
        def _wait():
            TASK_APPROVAL.wait_for_approval("t1", "delete these files", timeout=5)

        t = threading.Thread(target=_wait, daemon=True)
        t.start()
        deadline = time.time() + 2
        while time.time() < deadline and not TASK_APPROVAL.list_pending():
            time.sleep(0.01)
        pending = TASK_APPROVAL.list_pending()
        self.assertEqual(len(pending), 1)

        TASK_APPROVAL.answer(pending[0]["approval_id"], approve=True)
        t.join(timeout=2)

        loop = _RecordingLoop()
        loop._check_stale_approvals()
        self.assertEqual(loop.spoken, [])


class FailedTaskTriggerTest(ProactiveTestCase):
    def test_failed_task_nudges_with_error_detail(self):
        _insert_task("t1", "failed", error="network timeout", goal="book a flight")
        loop = _RecordingLoop()
        loop._check_failed_tasks()
        self.assertEqual(len(loop.spoken), 1)
        self.assertIn("book a flight", loop.spoken[0])
        self.assertIn("network timeout", loop.spoken[0])

    def test_completed_task_does_not_nudge(self):
        _insert_task("t1", "completed")
        loop = _RecordingLoop()
        loop._check_failed_tasks()
        self.assertEqual(loop.spoken, [])

    def test_restart_does_not_resurface_a_failed_task_already_nudged(self):
        """The fix the plan required: dedup for this trigger must survive
        a process restart (a fresh ProactiveLoop / in-memory state), since
        a failed task row has no natural state-exit like approvals do."""
        _insert_task("t1", "failed", error="network timeout", goal="book a flight")
        first_run = _RecordingLoop()
        first_run._check_failed_tasks()
        self.assertEqual(len(first_run.spoken), 1)

        # Simulate a restart: brand-new loop instance, empty in-memory
        # state, same on-disk db.
        second_run = _RecordingLoop()
        second_run._check_failed_tasks()
        self.assertEqual(second_run.spoken, [])

    def test_suppressed_nudge_is_not_dedup_recorded_and_retries_next_tick(self):
        _insert_task("t1", "failed", error="network timeout", goal="book a flight")
        loop = _RecordingLoop(is_muted=lambda: True)
        loop._check_failed_tasks()
        self.assertEqual(loop.spoken, [])
        self.assertFalse(db.nudge_already_sent(proactive.TRIGGER_TASK_FAILED, "t1"))


class GatingTest(ProactiveTestCase):
    def test_speaking_suppresses_nudge(self):
        _insert_task("t1", "failed", error="x", goal="y")
        loop = _RecordingLoop(is_speaking=lambda: True)
        loop._check_failed_tasks()
        self.assertEqual(loop.spoken, [])

    def test_muted_suppresses_nudge(self):
        _insert_task("t1", "failed", error="x", goal="y")
        loop = _RecordingLoop(is_muted=lambda: True)
        loop._check_failed_tasks()
        self.assertEqual(loop.spoken, [])

    def test_quiet_hours_suppresses_a_nudge_that_would_otherwise_fire(self):
        """Concrete proof the config-level quiet_hours gate actually
        suppresses delivery, not just that the key exists unused."""
        cfg = {"quiet_hours": {"start": "22:00", "end": "07:00"}}
        original_load_config = proactive.load_config
        proactive.load_config = lambda: cfg
        try:
            _insert_task("t1", "failed", error="x", goal="y")
            during_quiet_hours = datetime(2026, 8, 10, 23, 0)
            loop = _RecordingLoop(now=lambda: during_quiet_hours)
            self.assertTrue(proactive._quiet_hours_active(cfg, now=during_quiet_hours))
            loop._check_failed_tasks()
            self.assertEqual(loop.spoken, [])
        finally:
            proactive.load_config = original_load_config

    def test_outside_quiet_hours_window_nudge_still_fires(self):
        cfg = {"quiet_hours": {"start": "22:00", "end": "07:00"}}
        original_load_config = proactive.load_config
        proactive.load_config = lambda: cfg
        try:
            _insert_task("t1", "failed", error="x", goal="y")
            midday = datetime(2026, 8, 10, 12, 0)
            loop = _RecordingLoop(now=lambda: midday)
            self.assertFalse(proactive._quiet_hours_active(cfg, now=midday))
            loop._check_failed_tasks()
            self.assertEqual(len(loop.spoken), 1)
        finally:
            proactive.load_config = original_load_config

    def test_no_quiet_hours_configured_never_suppresses(self):
        self.assertFalse(proactive._quiet_hours_active({}))
        self.assertFalse(proactive._quiet_hours_active({"quiet_hours": {}}))
        self.assertFalse(proactive._quiet_hours_active({"quiet_hours": {"start": "bad"}}))


if __name__ == "__main__":
    unittest.main()
