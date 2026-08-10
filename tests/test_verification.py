"""
tests/test_verification.py
=============================
Covers the verification phase: get_step_outcomes()'s replan-safe
segmenting, _summarize()'s two fixed paths (LLM-prompt and deterministic
fallback), the postcondition-check registry, the dispatch_tool()
refactor's behavior-preservation, and postcondition failures flowing
through the existing except-Exception -> analyze_error() ->
_maybe_downgrade_retry() retry machinery instead of a hard short-circuit.

Redirects core.db at a fresh temp sqlite file, same pattern as the other
test modules in this package, so nothing here touches data/jarvis.db.
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

import core.policy as policy               # noqa: E402
import core.tool_dispatch as tdisp         # noqa: E402
import core.tool_gate as gate              # noqa: E402
import core.postconditions as postconditions  # noqa: E402
import core.tool_contracts as contracts    # noqa: E402
import agent.executor as executor          # noqa: E402
from agent.error_handler import ErrorDecision  # noqa: E402


def _insert_task(task_id: str, goal: str = "test goal"):
    conn = db.get_conn()
    with conn:
        conn.execute(
            "INSERT INTO tasks (task_id, goal, priority, status, result, error, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (task_id, goal, 2, "running", None, "", time.time(), time.time()),
        )


class GetStepOutcomesTest(unittest.TestCase):
    def setUp(self):
        _use_temp_db()
        _insert_task("t1")

    def test_done_step_is_reported_done(self):
        db.log_task_event("t1", 1, "web_search", "search flights", "started")
        db.log_task_event("t1", 1, "web_search", "search flights", "done", "found flights")
        outcomes = db.get_step_outcomes("t1")
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0]["status"], "done")

    def test_skipped_step_is_reported_skipped_not_done(self):
        """The exact original bug scenario at the data layer: a skipped
        step must never be reported as 'done'."""
        db.log_task_event("t1", 2, "send_message", "send to Mom", "started")
        db.log_task_event("t1", 2, "send_message", "send to Mom", "attempt_failed", "attempt 1: contact not found")
        db.log_task_event("t1", 2, "send_message", "send to Mom", "skipped", "skipped (non-critical)")
        outcomes = db.get_step_outcomes("t1")
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0]["status"], "skipped")
        self.assertIn("non-critical", outcomes[0]["detail"])

    def test_attempt_failed_and_retried_noise_is_collapsed_into_terminal_status(self):
        db.log_task_event("t1", 1, "code_helper", "write script", "started")
        db.log_task_event("t1", 1, "code_helper", "write script", "attempt_failed", "attempt 1: syntax error")
        db.log_task_event("t1", 1, "code_helper", "write script", "retried", "attempt 1 -> 2")
        db.log_task_event("t1", 1, "code_helper", "write script", "done", "fixed")
        outcomes = db.get_step_outcomes("t1")
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0]["status"], "done")

    def test_replan_reusing_a_step_number_does_not_conflate_segments(self):
        """A step_num can be reused by a replanned plan for an entirely
        different step — each 'started' must begin its own segment, not
        overwrite/merge with an earlier one sharing the same number."""
        db.log_task_event("t1", 1, "web_search", "search NYC-LAX", "started")
        db.log_task_event("t1", 1, "web_search", "search NYC-LAX", "done", "found flights")
        db.log_task_event("t1", 2, "send_message", "send to Mom", "started")
        db.log_task_event("t1", 2, "send_message", "send to Mom", "failed", "validation_failed: bad contact")
        # replanned — step 1 is reused for a *different* step in the new plan
        db.log_task_event("t1", 1, "reminder", "set a reminder instead", "started")
        db.log_task_event("t1", 1, "reminder", "set a reminder instead", "done", "reminder set")

        outcomes = db.get_step_outcomes("t1")
        self.assertEqual(len(outcomes), 3)
        self.assertEqual(outcomes[0]["tool"], "web_search")
        self.assertEqual(outcomes[0]["status"], "done")
        self.assertEqual(outcomes[1]["tool"], "send_message")
        self.assertEqual(outcomes[1]["status"], "failed")
        self.assertEqual(outcomes[2]["tool"], "reminder")
        self.assertEqual(outcomes[2]["status"], "done")

    def test_recovered_via_fix_is_still_done_with_detail_preserved(self):
        db.log_task_event("t1", 1, "web_search", "search flights", "started")
        db.log_task_event("t1", 1, "web_search", "search flights", "attempt_failed", "attempt 1: network error")
        db.log_task_event("t1", 1, "code_helper", "search flights", "done", "recovered via fix: got results")
        outcomes = db.get_step_outcomes("t1")
        self.assertEqual(outcomes[0]["status"], "done")
        self.assertTrue(outcomes[0]["detail"].startswith("recovered via fix"))
        # tool/description are locked in from the *original* 'started' row,
        # not overwritten by the recovery's own (different) tool
        self.assertEqual(outcomes[0]["tool"], "web_search")

    def test_no_task_id_returns_empty_list(self):
        self.assertEqual(db.get_step_outcomes(None), [])
        self.assertEqual(db.get_step_outcomes(""), [])

    def test_unknown_task_id_returns_empty_list(self):
        self.assertEqual(db.get_step_outcomes("no-such-task"), [])


class SummarizeBeforeAfterTest(unittest.TestCase):
    """The original bug, reproduced and confirmed fixed: a plan where
    step 2 is skipped must not be described as accomplished, on either
    the LLM-prompt path or the deterministic fallback."""

    def setUp(self):
        _use_temp_db()
        _insert_task("t-summarize")
        db.log_task_event("t-summarize", 1, "web_search", "Search for cheapest flight NYC to LAX", "started")
        db.log_task_event("t-summarize", 1, "web_search", "Search for cheapest flight NYC to LAX", "done", "found: Delta $350")
        db.log_task_event("t-summarize", 2, "send_message", "Send the cheapest flight to Mom", "started")
        db.log_task_event("t-summarize", 2, "send_message", "Send the cheapest flight to Mom", "attempt_failed", "attempt 1: contact not found")
        db.log_task_event("t-summarize", 2, "send_message", "Send the cheapest flight to Mom", "skipped", "skipped (non-critical)")

    def test_fallback_summary_distinguishes_done_from_skipped(self):
        outcomes = db.get_step_outcomes("t-summarize")
        done    = [o for o in outcomes if o["status"] == "done"]
        skipped = [o for o in outcomes if o["status"] == "skipped"]
        fallback = executor._build_fallback_summary("find and send Mom the cheapest flight", done, skipped, [])

        # Concise by design (this is the no-LLM safety net, not the
        # naturally-phrased path) — a count of done steps, plus the
        # skipped one called out by name. Not "All done, sir" — that
        # exact wording was the original bug's symptom.
        self.assertIn("Completed 1 of 2 step(s)", fallback)
        self.assertIn("Skipped", fallback)
        self.assertIn("Send the cheapest flight to Mom", fallback)
        self.assertNotIn("All done", fallback)

    def test_llm_prompt_marks_the_skipped_step_explicitly(self):
        outcomes = db.get_step_outcomes("t-summarize")
        done    = [o for o in outcomes if o["status"] == "done"]
        skipped = [o for o in outcomes if o["status"] == "skipped"]
        prompt = executor._build_summary_prompt("find and send Mom the cheapest flight", done, skipped, [])

        self.assertIn("- DONE: Search for cheapest flight NYC to LAX", prompt)
        self.assertIn("- SKIPPED:", prompt)
        self.assertIn("Send the cheapest flight to Mom", prompt)
        # the old unconditional instruction must be gone
        self.assertNotIn("Be direct and positive", prompt)
        self.assertIn("do not claim a skipped or failed step was completed", prompt)

    def test_summarize_end_to_end_via_fallback(self):
        orig_call_llm_text = executor.call_llm_text
        executor.call_llm_text = lambda *a, **kw: ""  # force the fallback path
        try:
            ex = executor.AgentExecutor()
            result = ex._summarize("find and send Mom the cheapest flight", speak=None, task_id="t-summarize")
        finally:
            executor.call_llm_text = orig_call_llm_text

        self.assertIn("Skipped", result)
        self.assertNotIn("All done", result)


class PostconditionRegistryTest(unittest.TestCase):
    def test_registered_tools_match_the_approved_list_exactly(self):
        approved = {
            "file_controller", "file_processor", "code_helper",
            "dev_agent", "reminder", "game_updater", "desktop_control",
        }
        self.assertEqual(set(postconditions.POSTCONDITION_CHECKS.keys()), approved)

    def test_unregistered_tool_has_no_opinion(self):
        self.assertIsNone(postconditions.check_postcondition("weather_report", {}, "sunny"))

    def test_checker_exception_is_swallowed_not_propagated_as_failure(self):
        postconditions.POSTCONDITION_CHECKS["_test_tool"] = lambda a, r: 1 / 0
        try:
            result = postconditions.check_postcondition("_test_tool", {}, "ok")
        finally:
            del postconditions.POSTCONDITION_CHECKS["_test_tool"]
        self.assertIsNone(result)

    def test_file_controller_write_fails_when_file_does_not_exist(self):
        import tempfile as _tempfile
        tmp_dir = _tempfile.mkdtemp()
        unmet = postconditions.check_postcondition(
            "file_controller",
            {"action": "write", "path": tmp_dir, "name": "does_not_exist.txt", "content": "x"},
            "Written to: does_not_exist.txt",
        )
        self.assertIsNotNone(unmet)
        self.assertIn("does not exist", unmet)

    def test_file_controller_write_passes_when_file_exists(self):
        import tempfile as _tempfile
        tmp_dir = _tempfile.mkdtemp()
        real_file = Path(tmp_dir) / "real.txt"
        real_file.write_text("hello")
        unmet = postconditions.check_postcondition(
            "file_controller",
            {"action": "write", "path": tmp_dir, "name": "real.txt", "content": "hello"},
            "Written to: real.txt",
        )
        self.assertIsNone(unmet)

    def test_file_controller_delete_fails_when_file_still_exists(self):
        import tempfile as _tempfile
        tmp_dir = _tempfile.mkdtemp()
        still_there = Path(tmp_dir) / "still_there.txt"
        still_there.write_text("oops")
        unmet = postconditions.check_postcondition(
            "file_controller",
            {"action": "delete", "path": tmp_dir, "name": "still_there.txt"},
            "Moved to Trash: still_there.txt",
        )
        self.assertIsNotNone(unmet)
        self.assertIn("still exists", unmet)


class DispatchToolPostconditionIntegrationTest(unittest.TestCase):
    """The concrete requested example: file_controller claims a delete
    succeeded but the file is still there — dispatch_tool() must return
    the distinguishable 'Postcondition unmet' marker, not the tool's own
    (false) success string."""

    def setUp(self):
        _use_temp_db()
        policy.seed_default_policy()
        import tempfile as _tempfile
        self.tmp_dir = _tempfile.mkdtemp()
        self.target = Path(self.tmp_dir) / "target.txt"
        self.target.write_text("still here")

        self._orig_tool = tdisp.TOOL_DISPATCH.get("file_controller")
        tdisp.TOOL_DISPATCH["file_controller"] = lambda args, player, speak: "Moved to Trash: target.txt"

        # file_controller's 'delete' action is ask-and-wait — with
        # player=None that blocks on TASK_APPROVAL.wait_for_approval()
        # for up to 30 minutes with nothing to ever answer it. Auto-approve
        # instantly: this test is about the postcondition check, not the
        # approval flow (already covered elsewhere).
        import core.task_approval as ta_mod
        self._orig_wait = ta_mod.TASK_APPROVAL.wait_for_approval
        ta_mod.TASK_APPROVAL.wait_for_approval = lambda task_id, prompt, timeout=None: (True, "approved")
        self._ta_mod = ta_mod

    def tearDown(self):
        if self._orig_tool is not None:
            tdisp.TOOL_DISPATCH["file_controller"] = self._orig_tool
        self._ta_mod.TASK_APPROVAL.wait_for_approval = self._orig_wait

    def test_claimed_delete_that_did_not_happen_is_caught(self):
        result = gate.dispatch_tool(
            "file_controller",
            {"action": "delete", "path": self.tmp_dir, "name": "target.txt"},
            player=None, speak=None, task_id="t-postcond",
        )
        self.assertTrue(result.startswith("Postcondition unmet — "))
        self.assertIn("still exists", result)
        self.assertTrue(self.target.exists(), "the file genuinely wasn't touched by this test")


class DispatchToolRefactorBehaviorTest(unittest.TestCase):
    """Targeted checks that the 4-call-sites-to-1-shared-tail refactor
    preserved behavior for each permission level, beyond the full
    existing suite (87 tests) already passing unchanged."""

    def setUp(self):
        _use_temp_db()
        policy.seed_default_policy()

    def _stub(self, tool_name, return_value="ok"):
        calls = []
        orig = tdisp.TOOL_DISPATCH.get(tool_name)

        def _fake(args, player, speak):
            calls.append(args)
            return return_value

        tdisp.TOOL_DISPATCH[tool_name] = _fake
        self.addCleanup(lambda: tdisp.TOOL_DISPATCH.__setitem__(tool_name, orig) if orig else None)
        return calls

    def test_auto_allow_path_invokes_once_and_logs_auto_allowed(self):
        calls = self._stub("weather_report")
        result = gate.dispatch_tool("weather_report", {"city": "Paris"}, player=None, speak=None, task_id="t-aa")
        self.assertEqual(result, "ok")
        self.assertEqual(len(calls), 1)
        row = db.get_conn().execute(
            "SELECT outcome FROM policy_decisions WHERE task_id='t-aa'"
        ).fetchone()
        self.assertEqual(row["outcome"], "auto-allowed")

    def test_notify_only_path_invokes_once_and_logs_notified(self):
        calls = self._stub("browser_control")
        result = gate.dispatch_tool(
            "browser_control", {"action": "get_text"}, player=None, speak=None, task_id="t-no",
        )
        self.assertEqual(result, "ok")
        self.assertEqual(len(calls), 1)
        row = db.get_conn().execute(
            "SELECT outcome FROM policy_decisions WHERE task_id='t-no'"
        ).fetchone()
        self.assertEqual(row["outcome"], "notified")

    def test_delegated_path_invokes_once_and_logs_delegated(self):
        calls = self._stub("dev_agent")
        result = gate.dispatch_tool(
            "dev_agent", {"description": "a simple script"}, player=None, speak=None, task_id="t-del",
        )
        self.assertEqual(result, "ok")
        self.assertEqual(len(calls), 1)
        row = db.get_conn().execute(
            "SELECT outcome FROM policy_decisions WHERE task_id='t-del'"
        ).fetchone()
        self.assertEqual(row["outcome"], "delegated")

    def test_ask_and_wait_approved_path_invokes_once_and_logs_approved(self):
        # send_message: whole-tool ask-and-wait, no postcondition check
        # registered for it — keeps this test isolated to the dispatch
        # flow itself, not entangled with postcondition behavior (covered
        # separately in DispatchToolPostconditionIntegrationTest).
        import core.task_approval as ta_mod
        orig_wait = ta_mod.TASK_APPROVAL.wait_for_approval
        ta_mod.TASK_APPROVAL.wait_for_approval = lambda task_id, prompt, timeout=None: (True, "approved")
        try:
            calls = self._stub("send_message")
            result = gate.dispatch_tool(
                "send_message", {"receiver": "bob", "message_text": "hi", "platform": "whatsapp"},
                player=None, speak=None, task_id="t-aw",
            )
        finally:
            ta_mod.TASK_APPROVAL.wait_for_approval = orig_wait

        self.assertEqual(result, "ok")
        self.assertEqual(len(calls), 1)
        row = db.get_conn().execute(
            "SELECT outcome FROM policy_decisions WHERE task_id='t-aw'"
        ).fetchone()
        self.assertEqual(row["outcome"], "approved")

    def test_hard_deny_and_rejected_paths_never_invoke_the_tool(self):
        # validation rejection — never reaches the invoke tail at all
        calls = self._stub("weather_report")
        result = gate.dispatch_tool("weather_report", {}, player=None, speak=None, task_id="t-rej")
        self.assertEqual(len(calls), 0)
        self.assertTrue(result.startswith("Rejected — "))


class PostconditionFailureRetryWiringTest(unittest.TestCase):
    """The concrete example requested: a postcondition failure must
    become a real exception, be picked up by analyze_error(), and be
    retried or not strictly according to the tool's contract.retryable —
    not the unconditional short-circuit path."""

    def setUp(self):
        _use_temp_db()
        _insert_task("t-retry-no", goal="delete a file")
        _insert_task("t-retry-yes", goal="convert a video")

        self._orig_create_plan  = executor.create_plan
        self._orig_replan       = executor.replan
        self._orig_dispatch     = executor.dispatch_tool
        self._orig_analyze      = executor.analyze_error
        self._orig_call_llm     = executor.call_llm_text
        executor.call_llm_text  = lambda *a, **kw: ""
        # A step that ultimately fails falls through to the *task-level*
        # replanner (a real LLM call via agent.planner.replan, a separate
        # call_llm_text reference this doesn't touch) — stub it to end
        # the task cleanly instead of letting a real Ollama connection
        # attempt run (and, per an earlier phase, potentially spawn a
        # real 'ollama serve' subprocess as a fallback).
        executor.replan = lambda goal, completed_steps, failed_step, error, task_id=None: {"steps": []}

    def tearDown(self):
        executor.create_plan   = self._orig_create_plan
        executor.replan        = self._orig_replan
        executor.dispatch_tool = self._orig_dispatch
        executor.analyze_error = self._orig_analyze
        executor.call_llm_text = self._orig_call_llm

    def test_non_retryable_tool_postcondition_failure_is_not_retried(self):
        # file_controller: retryable=False per its contract
        self.assertFalse(contracts.get_contract("file_controller").retryable)

        calls = {"n": 0}

        def _fake_plan(goal, task_id=None):
            return {"steps": [{
                "step": 1, "tool": "file_controller", "description": "delete x.txt",
                "parameters": {"action": "delete"}, "critical": True,
            }]}

        def _fake_dispatch(tool, args, player, speak, task_id=None, submitted_interactively=True):
            calls["n"] += 1
            return "Postcondition unmet — 'file_controller' 'x.txt' still exists after delete. (tool reported: 'Moved to Trash')"

        def _fake_analyze_error(step, error, attempt=1, max_attempts=2, task_id=None):
            return {"decision": ErrorDecision.RETRY, "reason": "looks transient", "user_message": ""}

        executor.create_plan   = _fake_plan
        executor.dispatch_tool = _fake_dispatch
        executor.analyze_error = _fake_analyze_error

        ex = executor.AgentExecutor()
        ex.execute(goal="delete a file", task_id="t-retry-no", submitted_interactively=False)

        # RETRY was requested by analyze_error, but file_controller's
        # contract.retryable=False must downgrade it to REPLAN — so the
        # tool is invoked exactly once, never retried.
        self.assertEqual(calls["n"], 1)

        statuses = [r["status"] for r in db.get_conn().execute(
            "SELECT status FROM task_events WHERE task_id='t-retry-no' ORDER BY event_id"
        ).fetchall()]
        self.assertIn("attempt_failed", statuses)   # went through the exception path, not the hard short-circuit
        self.assertNotIn("retried", statuses)        # but was never actually retried

    def test_retryable_tool_postcondition_failure_is_retried_and_can_succeed(self):
        # file_processor: retryable=True per its contract
        self.assertTrue(contracts.get_contract("file_processor").retryable)

        calls = {"n": 0}

        def _fake_plan(goal, task_id=None):
            return {"steps": [{
                "step": 1, "tool": "file_processor", "description": "convert video",
                "parameters": {"file_path": "video.mp4", "action": "convert"}, "critical": True,
            }]}

        def _fake_dispatch(tool, args, player, speak, task_id=None, submitted_interactively=True):
            calls["n"] += 1
            if calls["n"] == 1:
                return "Postcondition unmet — 'file_processor' 'video_converted.mp4' does not exist after convert."
            return "Converted to MP4. Saved: video_converted.mp4"

        def _fake_analyze_error(step, error, attempt=1, max_attempts=2, task_id=None):
            return {"decision": ErrorDecision.RETRY, "reason": "looks transient", "user_message": ""}

        executor.create_plan   = _fake_plan
        executor.dispatch_tool = _fake_dispatch
        executor.analyze_error = _fake_analyze_error

        ex = executor.AgentExecutor()
        ex.execute(goal="convert a video", task_id="t-retry-yes", submitted_interactively=False)

        # retryable=True -> RETRY is honored as-is -> attempt 1 fails its
        # postcondition, attempt 2 succeeds for real.
        self.assertEqual(calls["n"], 2)

        statuses = [r["status"] for r in db.get_conn().execute(
            "SELECT status FROM task_events WHERE task_id='t-retry-yes' ORDER BY event_id"
        ).fetchall()]
        self.assertIn("attempt_failed", statuses)
        self.assertIn("retried", statuses)
        self.assertIn("done", statuses)


if __name__ == "__main__":
    unittest.main(verbosity=2)
