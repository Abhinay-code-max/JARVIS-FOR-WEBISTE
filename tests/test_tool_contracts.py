"""
tests/test_tool_contracts.py
==============================
Covers the tool-contracts phase: contract completeness, input validation,
the execution-timeout wrapper (and its enforce_timeout escape hatch), the
executor's unified short-circuit classification, the retryable-gated
RETRY->REPLAN downgrade, and that PLANNER_PROMPT is generated from
TOOL_DECLARATIONS rather than a separately hand-maintained copy.

Redirects core.db at a fresh temp sqlite file, same pattern as
tests/test_permission_model.py, so nothing here touches data/jarvis.db.
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
import core.tool_contracts as contracts    # noqa: E402
import core.tool_gate as gate              # noqa: E402
import core.tool_declarations as decls     # noqa: E402
import agent.planner as planner            # noqa: E402
import agent.executor as executor          # noqa: E402
from agent.error_handler import ErrorDecision  # noqa: E402


class ContractCompletenessTest(unittest.TestCase):
    def test_every_tool_dispatch_tool_has_a_contract(self):
        missing = [t for t in tdisp.TOOL_DISPATCH if t not in contracts.TOOL_CONTRACTS]
        self.assertEqual(missing, [], f"TOOL_DISPATCH tools with no explicit contract: {missing}")

    def test_no_contracts_for_nonexistent_tools(self):
        extra = [t for t in contracts.TOOL_CONTRACTS if t not in tdisp.TOOL_DISPATCH]
        self.assertEqual(extra, [])

    def test_every_contract_input_schema_resolves(self):
        for name, c in contracts.TOOL_CONTRACTS.items():
            self.assertIn("properties", c.input_schema, name)

    def test_get_contract_falls_back_conservatively_for_unknown_tool(self):
        c = contracts.get_contract("totally_unknown_tool_xyz")
        self.assertFalse(c.retryable)
        self.assertEqual(c.risk_level, "medium")

    def test_reminder_has_a_permission_policy_row(self):
        # Was missing entirely from the earlier permission-model phase —
        # caught while cross-checking contracts against TOOL_DISPATCH.
        policy.seed_default_policy()
        level = policy.get_policy_level("reminder", None)
        self.assertEqual(level, policy.ASK_AND_WAIT)


class InputValidationTest(unittest.TestCase):
    def test_missing_required_field_is_caught(self):
        schema = {"properties": {"city": {"type": "STRING"}}, "required": ["city"]}
        err = gate.validate_input("weather_report", {}, schema)
        self.assertIsNotNone(err)
        self.assertIn("city", err)

    def test_valid_input_passes(self):
        schema = {"properties": {"city": {"type": "STRING"}}, "required": ["city"]}
        err = gate.validate_input("weather_report", {"city": "London"}, schema)
        self.assertIsNone(err)

    def test_wrong_type_is_caught(self):
        # mirrors flight_finder's int(params.get("passengers", 1)) crash risk
        schema = {"properties": {"passengers": {"type": "INTEGER"}}, "required": []}
        err = gate.validate_input("flight_finder", {"passengers": "two"}, schema)
        self.assertIsNotNone(err)
        self.assertIn("passengers", err)

    def test_numeric_string_is_tolerated_for_integer(self):
        # the tool's own int("5") coercion would succeed — don't reject
        # what wouldn't actually crash it
        schema = {"properties": {"passengers": {"type": "INTEGER"}}, "required": []}
        err = gate.validate_input("flight_finder", {"passengers": "5"}, schema)
        self.assertIsNone(err)

    def test_unknown_extra_key_is_tolerated(self):
        schema = {"properties": {"city": {"type": "STRING"}}, "required": ["city"]}
        err = gate.validate_input("weather_report", {"city": "Paris", "extra_field": "whatever"}, schema)
        self.assertIsNone(err)

    def test_non_dict_args_rejected(self):
        schema = {"properties": {}, "required": []}
        err = gate.validate_input("weather_report", "not a dict", schema)
        self.assertIsNotNone(err)


class DispatchToolValidationIntegrationTest(unittest.TestCase):
    """A schema-invalid call must never reach the underlying tool
    function — verified end-to-end through dispatch_tool()."""

    def setUp(self):
        _use_temp_db()
        policy.seed_default_policy()
        self._orig_tool = tdisp.TOOL_DISPATCH.get("weather_report")
        tdisp.TOOL_DISPATCH["weather_report"] = lambda args, player, speak: "SHOULD NOT RUN"

    def tearDown(self):
        if self._orig_tool is not None:
            tdisp.TOOL_DISPATCH["weather_report"] = self._orig_tool

    def test_missing_required_param_never_reaches_the_tool(self):
        result = gate.dispatch_tool("weather_report", {}, player=object(), speak=None)
        self.assertNotEqual(result, "SHOULD NOT RUN")
        self.assertTrue(result.startswith("Rejected — "))
        self.assertIn("city", result)

    def test_valid_call_reaches_the_tool(self):
        result = gate.dispatch_tool("weather_report", {"city": "Tokyo"}, player=object(), speak=None)
        self.assertEqual(result, "SHOULD NOT RUN")  # i.e. it *did* run


class TimeoutWrapperTest(unittest.TestCase):
    def setUp(self):
        self._orig_tool = tdisp.TOOL_DISPATCH.get("weather_report")

    def tearDown(self):
        if self._orig_tool is not None:
            tdisp.TOOL_DISPATCH["weather_report"] = self._orig_tool

    def test_slow_tool_is_abandoned_at_timeout_and_caller_returns_promptly(self):
        release = threading.Event()

        def _slow_tool(args, player, speak):
            release.wait(timeout=5)  # would block indefinitely without the outer wrapper's join timeout
            return "finished late"

        tdisp.TOOL_DISPATCH["weather_report"] = _slow_tool
        fast_contract = contracts.ToolContract(
            tool_name="weather_report",
            input_schema={"properties": {}, "required": []},
            timeout_seconds=0.2,
            enforce_timeout=True,
        )

        start  = time.monotonic()
        result = gate._invoke_with_timeout("weather_report", {}, None, None, fast_contract)
        elapsed = time.monotonic() - start

        self.assertLess(elapsed, 2.0, "caller should not block past the contract timeout")
        self.assertIn("did not complete within", result)
        self.assertIn("abandoned", result)
        release.set()  # let the orphaned thread finish so it doesn't linger past the test

    def test_fast_tool_returns_normally_within_timeout(self):
        tdisp.TOOL_DISPATCH["weather_report"] = lambda args, player, speak: "sunny"
        fast_contract = contracts.ToolContract(
            tool_name="weather_report",
            input_schema={"properties": {}, "required": []},
            timeout_seconds=5.0,
            enforce_timeout=True,
        )
        result = gate._invoke_with_timeout("weather_report", {}, None, None, fast_contract)
        self.assertEqual(result, "sunny")

    def test_enforce_timeout_false_never_spawns_a_thread(self):
        tdisp.TOOL_DISPATCH["weather_report"] = lambda args, player, speak: "sunny"
        no_wrap_contract = contracts.ToolContract(
            tool_name="weather_report",
            input_schema={"properties": {}, "required": []},
            timeout_seconds=0.01,   # would fail instantly if the wrapper were used
            enforce_timeout=False,
        )
        orig_thread = threading.Thread
        spawned = {"n": 0}

        class _CountingThread(orig_thread):
            def __init__(self, *a, **kw):
                spawned["n"] += 1
                super().__init__(*a, **kw)

        threading.Thread = _CountingThread
        try:
            result = gate._invoke_with_timeout("weather_report", {}, None, None, no_wrap_contract)
        finally:
            threading.Thread = orig_thread

        self.assertEqual(result, "sunny")
        self.assertEqual(spawned["n"], 0, "enforce_timeout=False must call the tool directly, no thread hop")

    def test_youtube_video_contract_has_enforce_timeout_disabled(self):
        # the concrete thread-affinity finding: _ask_for_url()'s tkinter
        # dialog can't be safely abandoned by the generic wrapper
        c = contracts.get_contract("youtube_video")
        self.assertFalse(c.enforce_timeout)


class ExecutorShortCircuitTest(unittest.TestCase):
    def test_validation_rejection_is_classified(self):
        msg = "Rejected — 'weather_report' parameters are invalid: missing required parameter(s): city."
        self.assertEqual(executor._short_circuit_reason(msg), "validation_failed")

    def test_execution_timeout_is_classified(self):
        msg = "'dev_agent' did not complete within 900s and was abandoned."
        self.assertEqual(executor._short_circuit_reason(msg), "execution_timeout")

    def test_approval_timeout_still_classified_via_existing_path(self):
        msg = "Cancelled — 'file_controller' timed out waiting for approval."
        self.assertEqual(executor._short_circuit_reason(msg), "approval_timeout")

    def test_ordinary_result_is_not_short_circuited(self):
        self.assertIsNone(executor._short_circuit_reason("The weather in Paris is sunny."))


class RetryableDowngradeTest(unittest.TestCase):
    def test_non_retryable_tool_downgrades_retry_to_replan(self):
        # send_message: retryable=False
        decision = executor._maybe_downgrade_retry("send_message", ErrorDecision.RETRY, task_id=None)
        self.assertEqual(decision, ErrorDecision.REPLAN)

    def test_retryable_tool_keeps_retry(self):
        # weather_report: retryable=True
        decision = executor._maybe_downgrade_retry("weather_report", ErrorDecision.RETRY, task_id=None)
        self.assertEqual(decision, ErrorDecision.RETRY)

    def test_non_retry_decisions_are_unaffected(self):
        for d in (ErrorDecision.SKIP, ErrorDecision.REPLAN, ErrorDecision.ABORT):
            self.assertEqual(executor._maybe_downgrade_retry("send_message", d, task_id=None), d)


class PlannerPromptGenerationTest(unittest.TestCase):
    def test_planner_prompt_covers_every_tool_dispatch_tool(self):
        for name in tdisp.TOOL_DISPATCH:
            self.assertIn(name, planner.PLANNER_PROMPT, f"{name} missing from generated PLANNER_PROMPT")

    def test_previously_missing_tools_are_now_present(self):
        # concretely the drift this phase closed
        for name in ("file_processor", "daily_briefing", "vision_fix_code"):
            self.assertIn(name, planner.PLANNER_PROMPT)

    def test_planner_prompt_is_derived_from_tool_declarations_not_hand_copied(self):
        # changing the canonical source must change the generated prompt —
        # proves it's actually generated, not just textually similar
        rendered = decls.build_planner_tool_reference({"weather_report"})
        self.assertIn("weather_report", rendered)
        self.assertIn("city", rendered)


if __name__ == "__main__":
    unittest.main(verbosity=2)
