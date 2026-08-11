"""
tests/test_caller_attribution.py
==================================
The headless-extraction phase's Step 1.3: caller_id/caller_class/
triggered_by columns on task_events and log_events, and the real wiring
that populates them for task_events (agent/task_queue.py's submit() ->
AgentExecutor.execute() -> the local _log_event() wrapper -> every
task_events row a task's steps produce).

Redirects core.db at a fresh temp sqlite file per test, same pattern as
every other module in this package, so nothing here touches
data/jarvis.db. Real AgentExecutor.execute() runs throughout (create_plan/
dispatch_tool stubbed, same convention as tests/test_verification.py and
tests/test_permission_model.py) — the mechanism under test is the
attribution wiring itself, not the planning/dispatch machinery those
other files already cover.
"""
from __future__ import annotations

import sqlite3
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

import core.policy      as policy    # noqa: E402
import core.llm_client              # noqa: E402
import agent.executor   as executor  # noqa: E402
import agent.task_queue as task_queue  # noqa: E402


def _insert_task(task_id: str, goal: str = "test goal"):
    conn = db.get_conn()
    with conn:
        conn.execute(
            "INSERT INTO tasks (task_id, goal, priority, status, result, error, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (task_id, goal, 2, "running", None, "", time.time(), time.time()),
        )


class LogTaskEventAttributionTest(unittest.TestCase):
    def setUp(self):
        _use_temp_db()
        _insert_task("t1")

    def test_defaults_to_null_when_not_provided(self):
        db.log_task_event("t1", 1, "web_search", "search", "done", "ok")
        row = db.get_conn().execute(
            "SELECT caller_id, caller_class, triggered_by FROM task_events WHERE task_id = 't1'"
        ).fetchone()
        self.assertIsNone(row["caller_id"])
        self.assertIsNone(row["caller_class"])
        self.assertIsNone(row["triggered_by"])

    def test_persists_all_three_when_provided(self):
        db.log_task_event(
            "t1", 1, "code_helper", "fix bug", "done", "ok",
            caller_id="service:bugfix", caller_class="service:bugfix", triggered_by="service:bugfix",
        )
        row = db.get_conn().execute(
            "SELECT caller_id, caller_class, triggered_by FROM task_events WHERE task_id = 't1'"
        ).fetchone()
        self.assertEqual(row["caller_id"], "service:bugfix")
        self.assertEqual(row["caller_class"], "service:bugfix")
        self.assertEqual(row["triggered_by"], "service:bugfix")


class LogEventsAttributionTest(unittest.TestCase):
    """core/logging_setup.py's SQLiteHandler.emit() — the actual INSERT
    logic Step 1.3 changed — called directly and synchronously against a
    real constructed LogRecord, rather than through the full
    QueueHandler/QueueListener pipeline. That pipeline runs a long-lived
    background thread behind module-level global state
    (core.logging_setup._listener) that's shared process-wide and not
    designed to be repeatedly torn down and rebuilt across test cases —
    exercising it end-to-end here bought nothing this test actually
    cares about (the record -> SQL mapping) while adding real thread-
    timing flakiness across a full suite run. emit() itself is exactly
    the same code either way."""

    def setUp(self):
        _use_temp_db()

    def _make_record(self, message: str, **extra) -> "logging.LogRecord":
        import logging as _logging
        record = _logging.LogRecord(
            name="jarvis.test_caller_attribution", level=_logging.INFO,
            pathname=__file__, lineno=1, msg=message, args=(), exc_info=None,
        )
        for key, value in extra.items():
            setattr(record, key, value)
        return record

    def test_extra_caller_fields_are_persisted(self):
        import core.logging_setup as logging_setup
        handler = logging_setup.SQLiteHandler()
        handler.emit(self._make_record(
            "attribution test message",
            caller_id="service:personal", caller_class="service:personal", triggered_by="service:personal",
        ))

        row = db.get_conn().execute(
            "SELECT caller_id, caller_class, triggered_by FROM log_events WHERE message = 'attribution test message'"
        ).fetchone()
        self.assertIsNotNone(row, "log_events row was never written")
        self.assertEqual(row["caller_id"], "service:personal")
        self.assertEqual(row["caller_class"], "service:personal")
        self.assertEqual(row["triggered_by"], "service:personal")

    def test_ordinary_record_without_extra_leaves_them_null(self):
        import core.logging_setup as logging_setup
        handler = logging_setup.SQLiteHandler()
        handler.emit(self._make_record("plain message, no caller context"))

        row = db.get_conn().execute(
            "SELECT caller_id, caller_class, triggered_by FROM log_events WHERE message = 'plain message, no caller context'"
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertIsNone(row["caller_id"])
        self.assertIsNone(row["caller_class"])
        self.assertIsNone(row["triggered_by"])


class ExecutorAttributionIntegrationTest(unittest.TestCase):
    """Real AgentExecutor.execute() runs, real _log_event() wrapper, real
    task_events writes — create_plan/dispatch_tool stubbed (the
    mechanism under test is attribution, not planning/dispatch, already
    covered elsewhere)."""

    def setUp(self):
        _use_temp_db()
        self._orig_create_plan = executor.create_plan
        self._orig_dispatch    = executor.dispatch_tool
        self._orig_replan      = executor.replan

    def tearDown(self):
        executor.create_plan   = self._orig_create_plan
        executor.dispatch_tool = self._orig_dispatch
        executor.replan        = self._orig_replan

    def test_default_caller_class_is_desktop_for_every_step_event(self):
        _insert_task("t-desktop")
        executor.create_plan   = lambda goal, task_id=None: {"steps": [
            {"step": 1, "tool": "weather_report", "description": "check weather", "parameters": {}},
        ]}
        executor.dispatch_tool = lambda tool, args, player, speak, task_id=None, submitted_interactively=True: "Sunny."

        ex = executor.AgentExecutor()
        ex.execute(goal="check weather", task_id="t-desktop", submitted_interactively=False)

        # tool != 'llm' excludes core/llm_client.py's own, separate
        # log_task_event() call site (_log_llm_outcome(), invoked from
        # _summarize()'s call_llm_text() here) — a real, deliberate scope
        # boundary found while writing this test: threading caller_class
        # into every call_llm_text()/call_llm_vision() call site across
        # agent/planner.py, agent/error_handler.py, and half a dozen
        # actions/*.py modules (15+ call sites total) is a materially
        # larger change than Step 1.3 asked for. Flagged, not silently
        # left untested — this is exactly why the assertion is scoped
        # rather than blanket over every row for this task_id.
        rows = db.get_conn().execute(
            "SELECT status, caller_id, caller_class, triggered_by FROM task_events "
            "WHERE task_id = 't-desktop' AND tool != 'llm'"
        ).fetchall()
        self.assertGreaterEqual(len(rows), 2)  # at least 'started' + 'done'
        for row in rows:
            self.assertEqual(row["caller_id"], "desktop")
            self.assertEqual(row["caller_class"], "desktop")
            self.assertEqual(row["triggered_by"], "desktop")

    def test_llm_bookkeeping_row_is_a_known_unattributed_gap_not_a_silent_one(self):
        """The flip side of the scope boundary above, made explicit and
        regression-tested rather than just a comment: an LLM-bookkeeping
        task_events row (core/llm_client.py's own call_llm_text() ->
        _log_llm_outcome() -> log_task_event(), independent of
        AgentExecutor's _log_event() wrapper) is NOT currently attributed.
        Calls the real call_llm_text() (not stubbed) — it logs its
        'llm'-tool bookkeeping row on both its 'done' and 'failed' paths
        (see core/llm_client.py), so this holds regardless of whether a
        real LLM backend is reachable in this environment; only the
        resulting status differs, which this test doesn't care about.
        If a future change threads caller_class into call_llm_text() and
        this starts failing, that's the gap actually closing — update
        this test then, don't just widen it to keep passing."""
        _insert_task("t-llm-gap")
        core.llm_client.call_llm_text(
            "irrelevant prompt", task_id="t-llm-gap", step_num=None, purpose="test bookkeeping",
        )

        llm_row = db.get_conn().execute(
            "SELECT caller_class FROM task_events WHERE task_id = 't-llm-gap' AND tool = 'llm'"
        ).fetchone()
        self.assertIsNotNone(llm_row, "expected an 'llm' bookkeeping row from call_llm_text's own log site")
        self.assertIsNone(llm_row["caller_class"], "core/llm_client.py's own log_task_event() call now populates caller_class")

    def test_service_caller_class_is_recorded_on_every_step_event(self):
        _insert_task("t-service")
        executor.create_plan   = lambda goal, task_id=None: {"steps": [
            {"step": 1, "tool": "code_helper", "description": "fix bug", "parameters": {}},
        ]}
        executor.dispatch_tool = lambda tool, args, player, speak, task_id=None, submitted_interactively=True: "Fixed."

        ex = executor.AgentExecutor()
        ex.execute(
            goal="fix bug", task_id="t-service", submitted_interactively=False,
            caller_class=policy.SERVICE_BUGFIX,
        )

        rows = db.get_conn().execute(
            "SELECT status, caller_id, caller_class, triggered_by FROM task_events "
            "WHERE task_id = 't-service' AND tool != 'llm'"
        ).fetchall()
        self.assertGreaterEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["caller_class"], policy.SERVICE_BUGFIX)
            self.assertEqual(row["triggered_by"], policy.SERVICE_BUGFIX)

    def test_replanned_event_also_carries_caller_class(self):
        # The 'replanned' log site sits outside the per-step loop — its
        # own regression coverage, since it's structurally different from
        # every other _log_event() call site (step_num/tool come from
        # failed_step, not the loop's own locals).
        _insert_task("t-replan")
        executor.create_plan = lambda goal, task_id=None: {"steps": [
            {"step": 1, "tool": "weather_report", "description": "x", "parameters": {}},
        ]}
        executor.dispatch_tool = lambda tool, args, player, speak, task_id=None, submitted_interactively=True: "Rejected — bad input."
        executor.replan = lambda goal, completed_steps, failed_step, failed_error, task_id=None: {"steps": []}

        ex = executor.AgentExecutor()
        ex.execute(
            goal="x", task_id="t-replan", submitted_interactively=False,
            caller_class=policy.SERVICE_PERSONAL,
        )

        row = db.get_conn().execute(
            "SELECT caller_class, triggered_by FROM task_events WHERE task_id = 't-replan' AND status = 'replanned'"
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["caller_class"], policy.SERVICE_PERSONAL)
        self.assertEqual(row["triggered_by"], policy.SERVICE_PERSONAL)


class TaskQueueCallerClassWiringTest(unittest.TestCase):
    """Real TaskQueue.submit() -> real worker thread -> real
    AgentExecutor.execute() (create_plan/dispatch_tool stubbed) — proves
    caller_class survives the actual queue/threading boundary, not just
    a direct execute() call."""

    def setUp(self):
        _use_temp_db()
        self._orig_create_plan = executor.create_plan
        self._orig_dispatch    = executor.dispatch_tool

    def tearDown(self):
        executor.create_plan   = self._orig_create_plan
        executor.dispatch_tool = self._orig_dispatch

    def test_caller_class_survives_submit_to_worker_thread_to_task_events(self):
        executor.create_plan   = lambda goal, task_id=None: {"steps": [
            {"step": 1, "tool": "weather_report", "description": "x", "parameters": {}},
        ]}
        executor.dispatch_tool = lambda tool, args, player, speak, task_id=None, submitted_interactively=True: "Sunny."

        q = task_queue.TaskQueue()
        q.start()
        try:
            task_id = q.submit(goal="check weather", caller_class=policy.SERVICE_SUPPORT)

            deadline = time.time() + 5
            status = None
            while time.time() < deadline:
                status = q.get_status(task_id)
                if status and status["status"] in ("completed", "failed"):
                    break
                time.sleep(0.02)
            self.assertIsNotNone(status)
            self.assertEqual(status["status"], "completed", status)
        finally:
            q.stop()

        row = db.get_conn().execute(
            "SELECT caller_class, triggered_by FROM task_events WHERE task_id = ? AND status = 'done' AND tool != 'llm'",
            (task_id,),
        ).fetchone()
        self.assertIsNotNone(row, "no 'done' task_events row was written")
        self.assertEqual(row["caller_class"], policy.SERVICE_SUPPORT)
        self.assertEqual(row["triggered_by"], policy.SERVICE_SUPPORT)

    def test_default_submit_still_attributes_desktop(self):
        executor.create_plan   = lambda goal, task_id=None: {"steps": [
            {"step": 1, "tool": "weather_report", "description": "x", "parameters": {}},
        ]}
        executor.dispatch_tool = lambda tool, args, player, speak, task_id=None, submitted_interactively=True: "Sunny."

        q = task_queue.TaskQueue()
        q.start()
        try:
            task_id = q.submit(goal="check weather")  # no caller_class passed

            deadline = time.time() + 5
            status = None
            while time.time() < deadline:
                status = q.get_status(task_id)
                if status and status["status"] in ("completed", "failed"):
                    break
                time.sleep(0.02)
            self.assertEqual(status["status"], "completed", status)
        finally:
            q.stop()

        row = db.get_conn().execute(
            "SELECT caller_class FROM task_events WHERE task_id = ? AND status = 'done' AND tool != 'llm'",
            (task_id,),
        ).fetchone()
        self.assertEqual(row["caller_class"], "desktop")


class MainPyAgentTaskWiringTest(unittest.TestCase):
    """Static source check — main.py can't be imported in tests (real
    PyQt6/audio bootstrap as an import-time side effect, same reason as
    tests/test_caller_class_policy.py's equivalent check)."""

    def test_submit_call_passes_caller_class_through(self):
        main_src = (Path(__file__).resolve().parent.parent / "main.py").read_text(encoding="utf-8")
        branch_start = main_src.find('if name == "agent_task":')
        self.assertNotEqual(branch_start, -1)
        next_branch = main_src.find('if name ==', branch_start + 10)
        branch_body = main_src[branch_start:next_branch if next_branch != -1 else branch_start + 1200]
        self.assertIn("get_queue().submit(", branch_body)
        self.assertIn("caller_class = caller_class", branch_body)


class PreExistingInstallMigrationTest(unittest.TestCase):
    """A real pre-phase-1 install: task_events/log_events tables that
    predate caller attribution, already holding real rows, on disk
    before core.db.get_conn() ever runs against them in this process."""

    def test_existing_task_events_rows_survive_and_new_columns_are_null(self):
        tmp_dir = Path(tempfile.mkdtemp(prefix="jarvis_test_migration_"))
        db_path = tmp_dir / "legacy.db"

        raw = sqlite3.connect(str(db_path))
        raw.execute(
            "CREATE TABLE task_events (event_id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "task_id TEXT NOT NULL, step_num INTEGER, tool TEXT, description TEXT, "
            "status TEXT, detail TEXT, duration_ms INTEGER, created_at REAL NOT NULL)"
        )
        raw.execute(
            "INSERT INTO task_events (task_id, step_num, tool, description, status, detail, created_at) "
            "VALUES ('old-task', 1, 'web_search', 'search', 'done', 'ok', 0)"
        )
        raw.execute(
            "CREATE TABLE log_events (event_id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT, "
            "level TEXT NOT NULL, source TEXT NOT NULL, message TEXT NOT NULL, detail TEXT, "
            "duration_ms INTEGER, created_at REAL NOT NULL)"
        )
        raw.commit()
        raw.close()

        db.DB_DIR  = tmp_dir
        db.DB_PATH = db_path
        db._local  = threading.local()

        conn = db.get_conn()  # runs the real ALTER TABLE ADD COLUMN migrations
        row = conn.execute(
            "SELECT caller_id, caller_class, triggered_by FROM task_events WHERE task_id = 'old-task'"
        ).fetchone()
        self.assertIsNotNone(row, "pre-existing task_events row was lost during migration")
        self.assertIsNone(row["caller_id"])
        self.assertIsNone(row["caller_class"])
        self.assertIsNone(row["triggered_by"])

        cols = {r["name"] for r in conn.execute("PRAGMA table_info(log_events)")}
        self.assertIn("caller_id", cols)
        self.assertIn("caller_class", cols)
        self.assertIn("triggered_by", cols)


if __name__ == "__main__":
    unittest.main(verbosity=2)
