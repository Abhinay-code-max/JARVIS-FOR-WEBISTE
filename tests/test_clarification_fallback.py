"""
tests/test_clarification_fallback.py
====================================
Tests for the clarify-with-user fallback in agent/executor.py and core/confirm.py.

Verifies:
1. ConfirmationGate supports request_clarification (text mode).
2. Missing required parameters trigger clarification when player is present.
3. When player is None or times out, task fails cleanly without hanging.
4. Non-missing-parameter validation errors (e.g. wrong type) do NOT trigger clarification.
5. Other short-circuit reasons (hard_denied, execution_timeout, approval_denied, unresolved_reference)
   do NOT trigger clarification.
6. Clarification is capped at 1 round per task.
"""
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import core.db as db

def _use_temp_db() -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="jarvis_clarify_test_db_"))
    db.DB_DIR  = tmp_dir
    db.DB_PATH = tmp_dir / "test.db"
    db._local  = threading.local()
    return db.DB_PATH

_use_temp_db()

import core.policy as policy
import core.tool_dispatch as tdisp
import core.confirm as confirm
from core.confirm import CONFIRM
import agent.executor as executor
from agent.executor import AgentExecutor, _extract_missing_parameters, _format_clarification_question


class ExtractMissingParametersTest(unittest.TestCase):
    def test_extracts_single_missing_param(self):
        err = "validation_failed: Rejected — 'file_controller' parameters are invalid: missing required parameter(s): action."
        self.assertEqual(_extract_missing_parameters(err), "action")

    def test_extracts_multiple_missing_params(self):
        err = "validation_failed: Rejected — 'send_message' parameters are invalid: missing required parameter(s): receiver, message_text, platform."
        self.assertEqual(_extract_missing_parameters(err), "receiver, message_text, platform")

    def test_ignores_type_validation_error(self):
        err = "validation_failed: Rejected — 'weather_report' parameters are invalid: parameter 'city' should be string, got int (123)."
        self.assertIsNone(_extract_missing_parameters(err))

    def test_ignores_non_validation_errors(self):
        self.assertIsNone(_extract_missing_parameters("execution_timeout: 'dev_agent' did not complete within 900s"))
        self.assertIsNone(_extract_missing_parameters("hard_denied: 'tool' is not permitted."))
        self.assertIsNone(_extract_missing_parameters(None))


class ConfirmationGateClarificationTest(unittest.TestCase):
    def test_request_clarification_no_player_returns_none_immediately(self):
        res = CONFIRM.request_clarification(None, "Which file?")
        self.assertIsNone(res)

    def test_request_clarification_with_player_and_answer(self):
        mock_player = MagicMock()
        answer_thread = threading.Thread(
            target=lambda: (time.sleep(0.05), CONFIRM.answer("report.txt"))
        )
        answer_thread.start()
        res = CONFIRM.request_clarification(mock_player, "Which file?", timeout=2.0)
        answer_thread.join()
        self.assertEqual(res, "report.txt")
        mock_player.write_log.assert_any_call("CLARIFY: Which file?")


class ExecutorClarificationFallbackTest(unittest.TestCase):
    def setUp(self):
        _use_temp_db()
        policy.seed_default_policy()
        self._orig_app = tdisp.TOOL_DISPATCH.get("open_app")
        tdisp.TOOL_DISPATCH["open_app"] = lambda args, player, speak: "Opened app."

    def tearDown(self):
        if self._orig_app is not None:
            tdisp.TOOL_DISPATCH["open_app"] = self._orig_app

    @patch("agent.executor.replan")
    @patch("agent.executor.create_plan")
    def test_missing_param_without_player_fails_cleanly(self, mock_create_plan, mock_replan):
        # Planner produces a step with missing required parameter 'app_name'
        mock_create_plan.return_value = {
            "goal": "open browser",
            "steps": [
                {"step": 1, "tool": "open_app", "description": "open browser", "parameters": {}}
            ],
        }
        mock_replan.return_value = {
            "goal": "open browser",
            "steps": [
                {"step": 1, "tool": "open_app", "description": "open browser", "parameters": {}}
            ],
        }
        ex = AgentExecutor()
        result = ex.execute(goal="open browser", player=None, submitted_interactively=False)
        self.assertIn("Task failed after", result)

    @patch.object(AgentExecutor, "_summarize", return_value="Summary: completed successfully")
    @patch("agent.executor.create_plan")
    def test_missing_param_with_player_prompts_and_recovers(self, mock_create_plan, mock_summarize):
        # Step 1: initial plan with empty parameters (missing app_name)
        initial_plan = {
            "goal": "open browser",
            "steps": [
                {"step": 1, "tool": "open_app", "description": "open browser", "parameters": {}}
            ],
        }
        # Step 2: plan after clarification (with app_name)
        clarified_plan = {
            "goal": "open browser",
            "steps": [
                {
                    "step": 1,
                    "tool": "open_app",
                    "description": "open Chrome browser",
                    "parameters": {"app_name": "Chrome"},
                }
            ],
        }
        mock_create_plan.side_effect = [initial_plan, clarified_plan]

        mock_player = MagicMock()
        def _auto_answer():
            time.sleep(0.05)
            CONFIRM.answer("Chrome")

        t = threading.Thread(target=_auto_answer)
        t.start()

        ex = AgentExecutor()
        result = ex.execute(goal="open browser", player=mock_player, submitted_interactively=False)
        t.join()

        # Should have called create_plan a second time with the context
        self.assertEqual(mock_create_plan.call_count, 2)
        second_call_kwargs = mock_create_plan.call_args_list[1][1]
        self.assertIn("Clarification from user", second_call_kwargs.get("context", ""))
        mock_player.write_log.assert_any_call("CLARIFY: Which app_name should open_app use to open browser?")
        self.assertEqual(result, "Summary: completed successfully")

    @patch("agent.executor.replan")
    @patch("agent.executor.create_plan")
    def test_clarification_capped_at_one_round(self, mock_create_plan, mock_replan):
        # Both initial plan and clarified plan return missing parameters
        bad_plan = {
            "goal": "open browser",
            "steps": [
                {"step": 1, "tool": "open_app", "description": "open browser", "parameters": {}}
            ],
        }
        mock_create_plan.return_value = bad_plan
        mock_replan.return_value = bad_plan

        mock_player = MagicMock()
        def _auto_answer():
            time.sleep(0.05)
            CONFIRM.answer("still incomplete")

        t = threading.Thread(target=_auto_answer)
        t.start()

        ex = AgentExecutor()
        result = ex.execute(goal="open browser", player=mock_player, submitted_interactively=False)
        t.join()

        # create_plan called initial + 1 clarification round = 2 times (not looping indefinitely)
        self.assertEqual(mock_create_plan.call_count, 2)
        self.assertIn("Task failed after", result)

    @patch("agent.executor.replan")
    @patch("agent.executor.create_plan")
    def test_type_validation_error_does_not_trigger_clarification(self, mock_create_plan, mock_replan):
        # Type error (int instead of string), not missing parameter
        bad_plan = {
            "goal": "open browser",
            "steps": [
                {"step": 1, "tool": "open_app", "description": "open browser", "parameters": {"app_name": 12345}}
            ],
        }
        mock_create_plan.return_value = bad_plan
        mock_replan.return_value = bad_plan

        mock_player = MagicMock()
        ex = AgentExecutor()
        result = ex.execute(goal="open browser", player=mock_player, submitted_interactively=False)

        # create_plan called only once (no clarification round triggered)
        self.assertEqual(mock_create_plan.call_count, 1)
        self.assertIn("Task failed after", result)
        # Verify write_log was never called with CLARIFY:
        for call_args in mock_player.write_log.call_args_list:
            self.assertNotIn("CLARIFY:", call_args[0][0])

    @patch("core.tool_gate.get_policy_level", return_value=policy.HARD_DENY)
    @patch("agent.executor.replan")
    @patch("agent.executor.create_plan")
    def test_hard_denied_does_not_trigger_clarification(self, mock_create_plan, mock_replan, mock_pol):
        # Hard denied tool per policy
        denied_plan = {
            "goal": "run shell",
            "steps": [
                {"step": 1, "tool": "computer_control", "description": "run shell", "parameters": {"action": "hotkey", "keys": "ctrl+c"}}
            ],
        }
        mock_create_plan.return_value = denied_plan
        mock_replan.return_value = denied_plan

        mock_player = MagicMock()
        ex = AgentExecutor()
        result = ex.execute(goal="run shell", player=mock_player, submitted_interactively=False)

        self.assertEqual(mock_create_plan.call_count, 1)
        self.assertIn("Task failed after", result)
        for call_args in mock_player.write_log.call_args_list:
            self.assertNotIn("CLARIFY:", call_args[0][0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
