"""
tests/test_coding_agent.py
==========================
Unit and integration tests for actions/coding_agent.py:
  1. AntigravityProvider prompt assembly and availability checks
  2. Part C.1: Streaming real-time shell command approval (prompt contains exact command)
  3. Part C.2: Denied shell command handling (user rejects shell command)
  4. Part C.3: Shell approval cap enforcement (terminates cleanly when cap exceeded)
  5. Part C.4: Post-run filesystem boundary auditor (catches out-of-bounds file modifications)
  6. CodingAgent graceful fallback to dev_agent local-LLM loop
  7. TOOL_DISPATCH and dispatch_tool() registration and policy
"""
import json
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from actions.coding_agent import (
    AntigravityProvider,
    AntigravityError,
    AntigravityUnavailableError,
    AntigravityPermissionError,
    AntigravityTimeoutError,
    AntigravitySecurityCompromiseError,
    AntigravityApprovalCapExceededError,
    FilesystemAuditor,
    CodingAgent,
    coding_agent,
)
from core.tool_dispatch import TOOL_DISPATCH
from core.tool_contracts import get_contract
from core.policy import get_policy_level, ASK_AND_WAIT, DESKTOP


class MockPlayer:
    def __init__(self):
        self.logs = []

    def write_log(self, text: str):
        self.logs.append(text)


class TestAntigravityProviderStreaming(unittest.TestCase):

    def setUp(self):
        self.provider = AntigravityProvider(executable_path="fake_agy")
        self.player = MockPlayer()
        self.test_dir = Path.home() / "Desktop" / "JarvisProjects" / "_test_mock_project"

    def test_build_prompt_structure(self):
        prompt = self.provider.build_prompt(
            description="Build a CLI calculator",
            language="python",
            project_name="cli_calc",
            requirements=["support addition", "support subtraction"],
            working_dir=self.test_dir,
        )
        self.assertIn("CLI calculator", prompt)
        self.assertIn("cli_calc", prompt)
        self.assertIn("python", prompt)
        self.assertIn("support addition", prompt)
        self.assertIn("Target Working Directory", prompt)

    @patch("subprocess.run")
    def test_is_available(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        self.assertTrue(self.provider.is_available())

        mock_run.side_effect = FileNotFoundError()
        self.assertFalse(self.provider.is_available())

    # ── Part C.1: Normal coding task with approved benign shell command ───────
    @patch.object(AntigravityProvider, "is_available", return_value=True)
    @patch("core.confirm.CONFIRM.request", return_value=True)
    @patch("subprocess.Popen")
    def test_streaming_approval_benign_shell_command(self, mock_popen, mock_confirm, mock_avail):
        """User receives real-time prompt with exact command, approves it, task succeeds."""
        mock_proc = MagicMock()
        mock_proc.poll.side_effect = [None, None, 0]
        mock_proc.stdout.readline.side_effect = [
            json.dumps({"event": "tool_call", "id": "req-1", "tool": "run_command", "args": {"CommandLine": "python -m unittest"}}),
            json.dumps({"event": "result", "result": {"status": "SUCCESS", "response": "Tests passed and project built."}}),
            "",
        ]
        mock_popen.return_value = mock_proc

        res = self.provider.run_streaming(
            prompt="Build project",
            working_dir=self.test_dir,
            player=self.player,
        )

        self.assertTrue(mock_confirm.called)
        # Confirm prompt contains the actual command requested
        confirm_prompt_arg = mock_confirm.call_args[0][1]
        self.assertIn("python -m unittest", confirm_prompt_arg)

        # Confirm approval event was sent to agy stdin
        written_lines = [call[0][0] for call in mock_proc.stdin.write.call_args_list]
        self.assertTrue(any('"granted": true' in line for line in written_lines))
        self.assertEqual(res["status"], "success")
        self.assertIn("Tests passed", res["result"])

    # ── Part C.2: Denied shell command ────────────────────────────────────────
    @patch.object(AntigravityProvider, "is_available", return_value=True)
    @patch("core.confirm.CONFIRM.request", return_value=False)
    @patch("subprocess.Popen")
    def test_streaming_approval_denied_shell_command(self, mock_popen, mock_confirm, mock_avail):
        """When user denies shell execution, rejection event is relayed back over stdin."""
        mock_proc = MagicMock()
        mock_proc.poll.side_effect = [None, None, 0]
        mock_proc.stdout.readline.side_effect = [
            json.dumps({"event": "tool_call", "id": "req-2", "tool": "run_command", "args": {"CommandLine": "rm -rf /"}}),
            json.dumps({"event": "result", "result": {"status": "SUCCESS", "response": "Adapted without executing command."}}),
            "",
        ]
        mock_popen.return_value = mock_proc

        res = self.provider.run_streaming(
            prompt="Build project",
            working_dir=self.test_dir,
            player=self.player,
        )

        self.assertTrue(mock_confirm.called)
        written_lines = [call[0][0] for call in mock_proc.stdin.write.call_args_list]
        self.assertTrue(any('"granted": false' in line for line in written_lines))
        self.assertEqual(res["status"], "success")

    # ── Part C.3: Approval cap exceeded ───────────────────────────────────────
    @patch.object(AntigravityProvider, "is_available", return_value=True)
    @patch("core.confirm.CONFIRM.request", return_value=True)
    @patch("subprocess.Popen")
    def test_streaming_approval_cap_exceeded(self, mock_popen, mock_confirm, mock_avail):
        """When in-session shell executions exceed the max cap, terminates cleanly."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        # Emit 12 consecutive shell tool calls
        streamed_events = [
            json.dumps({"event": "tool_call", "id": f"req-{i}", "tool": "run_command", "args": {"CommandLine": f"echo step {i}"}})
            for i in range(12)
        ]
        mock_proc.stdout.readline.side_effect = streamed_events
        mock_popen.return_value = mock_proc

        with self.assertRaises(AntigravityApprovalCapExceededError):
            self.provider.run_streaming(
                prompt="Run runaway commands",
                working_dir=self.test_dir,
                player=self.player,
                max_shell_approvals=5,  # low cap for test
            )

        self.assertTrue(mock_proc.kill.called)

    # ── Part C.4: Post-Run Filesystem Boundary Auditor ─────────────────────────
    def test_post_run_filesystem_auditor_detects_out_of_bounds_write(self):
        """Auditor catches file modifications outside allowed project directory."""
        auditor = FilesystemAuditor(allowed_root=self.test_dir)
        
        # Mock baseline with files outside allowed_root in Desktop, AppData, and Documents
        desktop_file = Path.home() / "Desktop" / "_unauthorized_desktop.txt"
        appdata_file = Path.home() / "AppData" / "Local" / "_unauthorized_appdata.tmp"
        docs_file = Path.home() / "Documents" / "_unauthorized_doc.txt"

        auditor._baseline = {
            desktop_file: time.time() - 100.0,
            appdata_file: time.time() - 100.0,
            docs_file: time.time() - 100.0,
        }

        # Simulate file modifications
        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "stat") as mock_stat:
            mock_stat.return_value = MagicMock(st_mtime=time.time())
            violations = auditor.audit_modifications()

        self.assertIn(desktop_file, violations)
        self.assertIn(appdata_file, violations)
        self.assertIn(docs_file, violations)

    @patch.object(AntigravityProvider, "is_available", return_value=True)
    @patch("subprocess.Popen")
    @patch.object(FilesystemAuditor, "audit_modifications")
    def test_antigravity_raises_on_security_compromise(self, mock_audit, mock_popen, mock_avail):
        """When auditor detects out-of-bounds writes in AppData/Desktop, AntigravitySecurityCompromiseError is raised."""
        mock_audit.return_value = [Path.home() / "AppData" / "Local" / "tampered.tmp"]
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.stdout.readline.return_value = json.dumps({"event": "result", "result": {"status": "SUCCESS"}})
        mock_popen.return_value = mock_proc

        with self.assertRaises(AntigravitySecurityCompromiseError):
            self.provider.run_streaming(
                prompt="Malicious prompt",
                working_dir=self.test_dir,
                player=self.player,
            )


class TestCodingAgentFallback(unittest.TestCase):

    def setUp(self):
        self.mock_provider = MagicMock()
        self.agent = CodingAgent(provider=self.mock_provider)

    def test_coding_agent_uses_antigravity_when_available(self):
        self.mock_provider.is_available.return_value = True
        self.mock_provider.build_prompt.return_value = "built prompt"
        self.mock_provider.run.return_value = {
            "status": "success",
            "result": "Antigravity built the project successfully.",
            "duration": 5.0,
        }

        output = self.agent.build_project(
            description="Build a REST API",
            language="python",
            project_name="my_api",
        )

        self.assertTrue(self.mock_provider.run.called)
        self.assertIn("completed by Antigravity", output)
        self.assertIn("Antigravity built the project successfully", output)

    @patch("actions.dev_agent.dev_agent", return_value="Local dev_agent completed the build.")
    def test_coding_agent_falls_back_when_antigravity_unavailable(self, mock_dev_agent):
        self.mock_provider.is_available.return_value = False

        output = self.agent.build_project(
            description="Build a REST API",
            language="python",
            project_name="my_api",
        )

        self.assertFalse(self.mock_provider.run.called)
        self.assertTrue(mock_dev_agent.called)
        self.assertEqual(output, "Local dev_agent completed the build.")

    @patch("actions.dev_agent.dev_agent", return_value="Local dev_agent fallback success.")
    def test_coding_agent_falls_back_when_antigravity_errors(self, mock_dev_agent):
        self.mock_provider.is_available.return_value = True
        self.mock_provider.build_prompt.return_value = "built prompt"
        self.mock_provider.run.side_effect = AntigravityError("Subprocess failed")

        output = self.agent.build_project(
            description="Build a parser",
            language="python",
            project_name="parser_tool",
        )

        self.assertTrue(self.mock_provider.run.called)
        self.assertTrue(mock_dev_agent.called)
        self.assertEqual(output, "Local dev_agent fallback success.")


class TestToolDispatchRegistration(unittest.TestCase):

    def test_coding_agent_registered_in_tool_dispatch(self):
        self.assertIn("coding_agent", TOOL_DISPATCH)
        self.assertIn("dev_agent", TOOL_DISPATCH)

    def test_coding_agent_contract_registered(self):
        contract = get_contract("coding_agent")
        self.assertEqual(contract.tool_name, "coding_agent")
        self.assertEqual(contract.risk_level, "high")
        self.assertFalse(contract.retryable)

    def test_coding_agent_policy_level(self):
        level = get_policy_level("coding_agent", None, caller_class=DESKTOP)
        self.assertEqual(level, ASK_AND_WAIT)


if __name__ == "__main__":
    unittest.main()
