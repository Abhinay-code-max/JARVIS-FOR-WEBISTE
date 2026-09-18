"""
tests/test_interactive_summarization.py
=======================================
Verifies that action-based interactive requests through AgentExecutor.execute()
and _summarize() speak the actual tool result content (weather data, app status,
folder creation, code generation, etc.) rather than a generic step count ("Completed 0 step(s)").
"""
import unittest
from unittest.mock import patch, MagicMock

import core.db as db
from agent.executor import AgentExecutor, _build_fallback_summary, _describe_outcome, _build_summary_prompt


class TestInteractiveSummarization(unittest.TestCase):

    def test_single_step_fallback_summary_returns_actual_detail(self):
        """When 1 step succeeds, fallback summary returns the tool's actual detail."""
        done = [{
            "step_num": 1,
            "tool": "weather_report",
            "description": "Get weather for Hyderabad",
            "status": "done",
            "detail": "Hyderabad: 26°C, Broken clouds. Wind 12 km/h, Humidity 65%.",
        }]
        summary = _build_fallback_summary("what's the weather in Hyderabad?", done, [], [])
        self.assertIn("Hyderabad: 26°C", summary)
        self.assertNotIn("Completed 0 step(s)", summary)
        self.assertNotIn("Completed 1 step(s)", summary)

    def test_describe_outcome_includes_tool_detail_for_done_steps(self):
        """_describe_outcome includes both step description and real detail data."""
        outcome = {
            "step_num": 1,
            "tool": "weather_report",
            "description": "Get weather for Hyderabad",
            "status": "done",
            "detail": "Hyderabad: 26°C, Broken clouds.",
        }
        desc = _describe_outcome(outcome)
        self.assertIn("Get weather for Hyderabad", desc)
        self.assertIn("Hyderabad: 26°C", desc)

    def test_summary_prompt_instructs_llm_to_state_actual_data(self):
        """_build_summary_prompt includes detail and instructs LLM to state real data."""
        done = [{
            "step_num": 1,
            "tool": "weather_report",
            "description": "Get weather for Hyderabad",
            "status": "done",
            "detail": "Hyderabad: 26°C, Broken clouds.",
        }]
        prompt = _build_summary_prompt("what's the weather?", done, [], [])
        self.assertIn("Hyderabad: 26°C", prompt)
        self.assertIn("Include the actual result", prompt)

    @patch("agent.executor.call_llm_text", return_value="")
    @patch("agent.executor.create_plan")
    @patch("agent.executor.dispatch_tool")
    def test_interactive_weather_execution_speaks_real_result(self, mock_dispatch, mock_plan, mock_llm):
        """Executing 'what is the weather?' speaks the actual weather result string."""
        mock_plan.return_value = {
            "steps": [{
                "step": 1,
                "tool": "weather_report",
                "description": "Check current weather in Hyderabad",
                "parameters": {"city": "Hyderabad"},
                "critical": True,
            }]
        }
        mock_dispatch.return_value = "Hyderabad: 26°C, Broken clouds. Wind 12 km/h, Humidity 65%."

        spoken = []
        ex = AgentExecutor()
        result = ex.execute(
            goal="what is the weather in Hyderabad?",
            speak=lambda s: spoken.append(s),
            submitted_interactively=True,
        )

        self.assertIn("26°C", result)
        self.assertTrue(any("26°C" in s for s in spoken))
        self.assertNotIn("Completed 0 step(s)", result)

    @patch("agent.executor.call_llm_text", return_value="")
    @patch("agent.executor.create_plan")
    @patch("agent.executor.dispatch_tool")
    def test_interactive_open_app_speaks_real_result(self, mock_dispatch, mock_plan, mock_llm):
        """Executing 'open whatsapp' speaks the real app opening confirmation."""
        mock_plan.return_value = {
            "steps": [{
                "step": 1,
                "tool": "open_app",
                "description": "Open WhatsApp",
                "parameters": {"app_name": "whatsapp"},
                "critical": True,
            }]
        }
        mock_dispatch.return_value = "Opened WhatsApp successfully."

        spoken = []
        ex = AgentExecutor()
        result = ex.execute(
            goal="open whatsapp",
            speak=lambda s: spoken.append(s),
            submitted_interactively=True,
        )

        self.assertIn("WhatsApp", result)
        self.assertTrue(any("WhatsApp" in s for s in spoken))
        self.assertNotIn("Completed 0 step(s)", result)

    @patch("agent.executor.call_llm_text", return_value="")
    @patch("agent.executor.create_plan")
    @patch("agent.executor.dispatch_tool")
    def test_interactive_folder_creation_speaks_real_result(self, mock_dispatch, mock_plan, mock_llm):
        """Executing folder creation speaks the folder creation confirmation."""
        mock_plan.return_value = {
            "steps": [{
                "step": 1,
                "tool": "file_controller",
                "description": "Create folder Projects on desktop",
                "parameters": {"action": "create_folder", "path": "desktop", "name": "Projects"},
                "critical": True,
            }]
        }
        mock_dispatch.return_value = "Created folder 'Projects' on desktop."

        spoken = []
        ex = AgentExecutor()
        result = ex.execute(
            goal="create a folder named Projects on desktop",
            speak=lambda s: spoken.append(s),
            submitted_interactively=True,
        )

        self.assertIn("Projects", result)
        self.assertTrue(any("Projects" in s for s in spoken))
        self.assertNotIn("Completed 0 step(s)", result)

    @patch("agent.executor.call_llm_text", return_value="")
    @patch("agent.executor.create_plan")
    @patch("agent.executor.dispatch_tool")
    def test_interactive_coding_task_speaks_real_result(self, mock_dispatch, mock_plan, mock_llm):
        """Executing coding task speaks the code helper result."""
        mock_plan.return_value = {
            "steps": [{
                "step": 1,
                "tool": "code_helper",
                "description": "Write a python script for fibonacci",
                "parameters": {"action": "write", "task": "fibonacci script"},
                "critical": True,
            }]
        }
        mock_dispatch.return_value = "Created fibonacci.py with 30 lines of code."

        spoken = []
        ex = AgentExecutor()
        result = ex.execute(
            goal="write a python script for fibonacci",
            speak=lambda s: spoken.append(s),
            submitted_interactively=True,
        )

        self.assertIn("fibonacci.py", result)
        self.assertTrue(any("fibonacci.py" in s for s in spoken))
        self.assertNotIn("Completed 0 step(s)", result)

    @patch("agent.executor.call_llm_text", return_value="Sir, the weather in Hyderabad is 26°C with broken clouds.")
    @patch("agent.executor.create_plan")
    @patch("agent.executor.dispatch_tool")
    def test_interactive_llm_summary_speaks_real_result(self, mock_dispatch, mock_plan, mock_llm):
        """When LLM summarization is used, it speaks the LLM's natural result sentence."""
        mock_plan.return_value = {
            "steps": [{
                "step": 1,
                "tool": "weather_report",
                "description": "Check current weather in Hyderabad",
                "parameters": {"city": "Hyderabad"},
                "critical": True,
            }]
        }
        mock_dispatch.return_value = "Hyderabad: 26°C, Broken clouds."

        spoken = []
        ex = AgentExecutor()
        result = ex.execute(
            goal="what is the weather in Hyderabad?",
            speak=lambda s: spoken.append(s),
            submitted_interactively=True,
        )

        self.assertEqual(result, "Sir, the weather in Hyderabad is 26°C with broken clouds.")
        self.assertEqual(spoken, ["Sir, the weather in Hyderabad is 26°C with broken clouds."])


if __name__ == "__main__":
    unittest.main()
