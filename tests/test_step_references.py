"""
tests/test_step_references.py
================================
Covers the step-data-flow phase: ${step_N.output} reference resolution,
forward/self/duplicate-step-number validation at plan-creation time, and
the step_results replan-accumulation fix.

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

from agent.step_references import (  # noqa: E402
    find_references, find_all_references, resolve_references, UnresolvedReferenceError,
)
import agent.planner as planner    # noqa: E402
import agent.executor as executor  # noqa: E402


class FindReferencesTest(unittest.TestCase):
    def test_finds_single_reference(self):
        self.assertEqual(find_references("Cheapest: ${step_1.output}"), [1])

    def test_finds_multiple_references(self):
        self.assertEqual(find_references("${step_1.output} and ${step_2.output}"), [1, 2])

    def test_no_reference_returns_empty(self):
        self.assertEqual(find_references("plain text"), [])

    def test_recurses_into_list_values(self):
        self.assertEqual(find_references(["a", "${step_3.output}", "b"]), [3])

    def test_non_string_non_list_returns_empty(self):
        self.assertEqual(find_references(42), [])
        self.assertEqual(find_references(None), [])

    def test_find_all_references_scans_whole_params_dict(self):
        params = {"a": "${step_1.output}", "b": "no ref", "c": ["${step_2.output}"]}
        self.assertEqual(sorted(find_all_references(params)), [1, 2])


class ResolveReferencesTest(unittest.TestCase):
    def test_substitutes_whole_string(self):
        result = resolve_references({"message_text": "${step_1.output}"}, {1: "Flight data here"})
        self.assertEqual(result["message_text"], "Flight data here")

    def test_substitutes_embedded_within_larger_string(self):
        result = resolve_references(
            {"message_text": "Cheapest flight: ${step_1.output}. Book now."},
            {1: "$350 Delta"},
        )
        self.assertEqual(result["message_text"], "Cheapest flight: $350 Delta. Book now.")

    def test_leaves_non_reference_values_untouched(self):
        result = resolve_references({"receiver": "bob", "count": 5}, {1: "x"})
        self.assertEqual(result["receiver"], "bob")
        self.assertEqual(result["count"], 5)

    def test_substitutes_within_list_items(self):
        result = resolve_references({"items": ["${step_1.output}", "plain"]}, {1: "resolved"})
        self.assertEqual(result["items"], ["resolved", "plain"])

    def test_missing_step_raises_unresolved_reference_error(self):
        with self.assertRaises(UnresolvedReferenceError) as ctx:
            resolve_references({"message_text": "${step_2.output}"}, {1: "only step 1 ran"})
        self.assertEqual(ctx.exception.step_num, 2)

    def test_does_not_mutate_input_dict(self):
        original = {"message_text": "${step_1.output}"}
        resolve_references(original, {1: "resolved"})
        self.assertEqual(original["message_text"], "${step_1.output}")


class ShortCircuitReasonUnresolvedReferenceTest(unittest.TestCase):
    def test_unresolved_message_is_classified(self):
        msg = "Unresolved — step 2 references step 1's output, which is not available (skipped, failed, or never ran)."
        self.assertEqual(executor._short_circuit_reason(msg), "unresolved_reference")

    def test_ordinary_result_is_not_misclassified(self):
        self.assertIsNone(executor._short_circuit_reason("The weather is sunny."))


class PlannerValidationTest(unittest.TestCase):
    def test_duplicate_step_numbers_fall_back_to_single_step_plan(self):
        plan = {
            "goal": "test goal",
            "steps": [
                {"step": 1, "tool": "web_search", "description": "a", "parameters": {"query": "a"}, "critical": True},
                {"step": 1, "tool": "web_search", "description": "b", "parameters": {"query": "b"}, "critical": True},
            ],
        }
        fixed = planner._validate_and_fix_plan(plan, "test goal")
        self.assertEqual(len(fixed["steps"]), 1)
        self.assertEqual(fixed["steps"][0]["tool"], "web_search")

    def test_forward_reference_replaces_step_with_web_search(self):
        plan = {
            "goal": "test",
            "steps": [
                {"step": 1, "tool": "weather_report", "description": "d1",
                 "parameters": {"city": "${step_2.output}"}, "critical": True},
                {"step": 2, "tool": "weather_report", "description": "d2",
                 "parameters": {"city": "Paris"}, "critical": True},
            ],
        }
        fixed = planner._validate_and_fix_plan(plan, "test")
        step1 = next(s for s in fixed["steps"] if s["step"] == 1)
        self.assertEqual(step1["tool"], "web_search")
        self.assertNotIn("city", step1["parameters"])

    def test_self_reference_replaces_step_with_web_search(self):
        plan = {
            "goal": "test",
            "steps": [
                {"step": 1, "tool": "weather_report", "description": "d1",
                 "parameters": {"city": "${step_1.output}"}, "critical": True},
            ],
        }
        fixed = planner._validate_and_fix_plan(plan, "test")
        self.assertEqual(fixed["steps"][0]["tool"], "web_search")

    def test_reference_to_nonexistent_step_replaces_step_with_web_search(self):
        plan = {
            "goal": "test",
            "steps": [
                {"step": 1, "tool": "weather_report", "description": "d1",
                 "parameters": {"city": "${step_99.output}"}, "critical": True},
            ],
        }
        fixed = planner._validate_and_fix_plan(plan, "test")
        self.assertEqual(fixed["steps"][0]["tool"], "web_search")

    def test_valid_backward_reference_is_left_alone(self):
        plan = {
            "goal": "test",
            "steps": [
                {"step": 1, "tool": "weather_report", "description": "d1",
                 "parameters": {"city": "Paris"}, "critical": True},
                {"step": 2, "tool": "send_message", "description": "d2",
                 "parameters": {"receiver": "bob", "message_text": "${step_1.output}", "platform": "whatsapp"},
                 "critical": True},
            ],
        }
        fixed = planner._validate_and_fix_plan(plan, "test")
        step2 = next(s for s in fixed["steps"] if s["step"] == 2)
        self.assertEqual(step2["tool"], "send_message")
        self.assertEqual(step2["parameters"]["message_text"], "${step_1.output}")

    def test_planner_prompt_no_longer_forbids_references(self):
        self.assertNotIn("NEVER reference previous step results", planner.PLANNER_PROMPT)
        self.assertIn("${step_N.output}", planner.PLANNER_PROMPT)

    def test_planner_prompt_still_contains_generated_tool_reference(self):
        # proves the rules-text edit didn't disturb the generated section
        # from the tool-contracts phase
        for name in ("weather_report", "send_message", "flight_finder"):
            self.assertIn(name, planner.PLANNER_PROMPT)


class ReplanDoesNotLeakStaleStepResultsTest(unittest.TestCase):
    """The explicitly-required test: a stale step_results value from a
    discarded (pre-replan) attempt must not leak into the new plan's
    ${step_N.output} references."""

    def setUp(self):
        _use_temp_db()
        conn = db.get_conn()
        with conn:
            conn.execute(
                "INSERT INTO tasks (task_id, goal, priority, status, result, error, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("task-replan-leak", "test goal", 2, "running", None, "", time.time(), time.time()),
            )

        self._orig_create_plan  = executor.create_plan
        self._orig_replan       = executor.replan
        self._orig_dispatch     = executor.dispatch_tool
        self._orig_call_llm_text = executor.call_llm_text

        # A fully successful run reaches _summarize(), which otherwise
        # makes a real call_llm_text() call — no LLM server needed for
        # this test, and a failed connection attempt there falls back to
        # launching a real 'ollama serve' subprocess (core/llm_client.py's
        # ensure_ollama_running()), which must never happen from a test.
        executor.call_llm_text = lambda *a, **kw: ""

        self.calls = []

        def _fake_create_plan(goal, task_id=None):
            return {"steps": [
                {"step": 1, "tool": "weather_report", "description": "probe old",
                 "parameters": {"city": "origin_probe"}, "critical": True},
                {"step": 2, "tool": "file_controller", "description": "will be rejected",
                 "parameters": {"action": "delete"}, "critical": True},
            ]}

        def _fake_replan(goal, completed_steps, failed_step, error, task_id=None):
            return {"steps": [
                {"step": 1, "tool": "weather_report", "description": "probe new",
                 "parameters": {"city": "new_probe"}, "critical": True},
                {"step": 2, "tool": "send_message", "description": "send it",
                 "parameters": {
                     "receiver": "bob", "platform": "whatsapp",
                     "message_text": "Value: ${step_1.output}",
                 }, "critical": True},
            ]}

        def _fake_dispatch(tool, args, player, speak, task_id=None, submitted_interactively=True):
            self.calls.append((tool, dict(args)))
            if tool == "weather_report" and args.get("city") == "origin_probe":
                return "OLD_VALUE"
            if tool == "weather_report" and args.get("city") == "new_probe":
                return "NEW_VALUE"
            if tool == "file_controller":
                return "Rejected — simulated failure to force a replan"
            if tool == "send_message":
                return f"sent: {args.get('message_text')}"
            return "unhandled"

        executor.create_plan   = _fake_create_plan
        executor.replan        = _fake_replan
        executor.dispatch_tool = _fake_dispatch

    def tearDown(self):
        executor.create_plan   = self._orig_create_plan
        executor.replan        = self._orig_replan
        executor.dispatch_tool = self._orig_dispatch
        executor.call_llm_text = self._orig_call_llm_text

    def test_new_plans_step_1_output_is_used_not_the_discarded_plans(self):
        ex = executor.AgentExecutor()
        ex.execute(goal="test goal", task_id="task-replan-leak", submitted_interactively=False)

        send_message_calls = [args for tool, args in self.calls if tool == "send_message"]
        self.assertEqual(len(send_message_calls), 1)
        resolved_text = send_message_calls[0]["message_text"]

        self.assertEqual(resolved_text, "Value: NEW_VALUE")
        self.assertNotIn("OLD_VALUE", resolved_text)
        self.assertNotIn("${step_1.output}", resolved_text)


class FlightsToContactExampleTest(unittest.TestCase):
    """The concrete before/after example from the plan: search flights,
    then send the result to a contact via ${step_1.output}."""

    def setUp(self):
        _use_temp_db()
        conn = db.get_conn()
        with conn:
            conn.execute(
                "INSERT INTO tasks (task_id, goal, priority, status, result, error, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("task-flights-demo", "flights to contact", 2, "running", None, "", time.time(), time.time()),
            )

        self._orig_create_plan   = executor.create_plan
        self._orig_dispatch      = executor.dispatch_tool
        self._orig_call_llm_text = executor.call_llm_text
        executor.call_llm_text   = lambda *a, **kw: ""  # see ReplanDoesNotLeakStaleStepResultsTest.setUp
        self.calls = []

        def _fake_create_plan(goal, task_id=None):
            return {"steps": [
                {
                    "step": 1, "tool": "flight_finder",
                    "description": "Search flights from NYC to LAX",
                    "parameters": {"origin": "NYC", "destination": "LAX", "date": "2026-09-01"},
                    "critical": True,
                },
                {
                    "step": 2, "tool": "send_message",
                    "description": "Send the cheapest flight to the contact",
                    "parameters": {
                        "receiver": "Mom", "platform": "whatsapp",
                        "message_text": "Found this for you: ${step_1.output}",
                    },
                    "critical": True,
                },
            ]}

        def _fake_dispatch(tool, args, player, speak, task_id=None, submitted_interactively=True):
            self.calls.append((tool, dict(args)))
            if tool == "flight_finder":
                return "The cheapest option is Delta at $350 USD, departing 10:00, non-stop."
            if tool == "send_message":
                return f"Message sent to {args.get('receiver')}."
            return "unhandled"

        executor.create_plan   = _fake_create_plan
        executor.dispatch_tool = _fake_dispatch

    def tearDown(self):
        executor.create_plan   = self._orig_create_plan
        executor.dispatch_tool = self._orig_dispatch
        executor.call_llm_text = self._orig_call_llm_text

    def test_flight_result_is_embedded_in_the_message(self):
        ex = executor.AgentExecutor()
        ex.execute(goal="flights to contact", task_id="task-flights-demo", submitted_interactively=False)

        send_message_calls = [args for tool, args in self.calls if tool == "send_message"]
        self.assertEqual(len(send_message_calls), 1)
        self.assertEqual(
            send_message_calls[0]["message_text"],
            "Found this for you: The cheapest option is Delta at $350 USD, departing 10:00, non-stop.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
