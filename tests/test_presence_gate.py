"""
tests/test_presence_gate.py
===========================
Comprehensive unit and integration tests for the Dual-Factor Biometric Presence Gate.

Tests:
  1. Both Face & Voice match Abhinay -> ALLOWED / APPROVED
  2. Face matches but Voice is an Impostor -> DENIED (Fail-closed)
  3. Voice matches but Face is an Impostor / Missing -> DENIED (Fail-closed)
  4. Neither Face nor Voice match -> DENIED (Fail-closed)
  5. Short / empty audio utterance -> DENIED (Fail-closed)
  6. End-to-end integration via dispatch_tool() with ASK_AND_WAIT
"""
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from core.presence import (
    verify_biometric_presence,
    TARGET_USER,
    VOICE_THRESHOLD,
    FACE_CONFIDENCE_THRESHOLD,
)
from core.tool_gate import dispatch_tool


class MockPlayer:
    def __init__(self):
        self.logs = []

    def write_log(self, text: str):
        self.logs.append(text)


class TestBiometricPresenceGate(unittest.TestCase):

    def setUp(self):
        self.player = MockPlayer()
        self.mock_face_id = MagicMock()
        self.mock_voice_id = MagicMock()

    def test_approved_when_both_face_and_voice_match(self):
        """When both Face and Voice match Abhinay with high confidence -> Success."""
        self.mock_face_id.identify.return_value = (TARGET_USER, 0.85)
        self.mock_voice_id.record_sample.return_value = np.zeros(16000 * 3, dtype=np.float32)
        self.mock_voice_id.identify.return_value = (TARGET_USER, 0.88)

        ok, reason, details = verify_biometric_presence(
            player=self.player,
            face_id=self.mock_face_id,
            voice_id=self.mock_voice_id,
        )

        self.assertTrue(ok)
        self.assertIn("verified", reason.lower())
        self.assertTrue(details["face"]["matched"])
        self.assertTrue(details["voice"]["matched"])
        self.assertTrue(any("Biometric presence verified" in log for log in self.player.logs))

    def test_denied_when_voice_impostor(self):
        """When Face matches Abhinay but Voice is an Impostor -> DENIED."""
        self.mock_face_id.identify.return_value = (TARGET_USER, 0.85)
        self.mock_voice_id.record_sample.return_value = np.zeros(16000 * 3, dtype=np.float32)
        # Impostor voice (different speaker or below threshold)
        self.mock_voice_id.identify.return_value = ("ImpostorSpeaker", 0.62)

        ok, reason, details = verify_biometric_presence(
            player=self.player,
            face_id=self.mock_face_id,
            voice_id=self.mock_voice_id,
        )

        self.assertFalse(ok)
        self.assertIn("Voice not recognized", reason)
        self.assertTrue(details["face"]["matched"])
        self.assertFalse(details["voice"]["matched"])
        self.assertTrue(any("Biometric presence failed" in log for log in self.player.logs))

    def test_denied_when_face_impostor(self):
        """When Voice matches Abhinay but Face is an Impostor -> DENIED."""
        # Impostor face
        self.mock_face_id.identify.return_value = ("UnknownPerson", 0.25)
        self.mock_voice_id.record_sample.return_value = np.zeros(16000 * 3, dtype=np.float32)
        self.mock_voice_id.identify.return_value = (TARGET_USER, 0.88)

        ok, reason, details = verify_biometric_presence(
            player=self.player,
            face_id=self.mock_face_id,
            voice_id=self.mock_voice_id,
        )

        self.assertFalse(ok)
        self.assertIn("Face not recognized", reason)
        self.assertFalse(details["face"]["matched"])
        self.assertFalse(details["voice"]["matched"])  # Voice not evaluated because face failed first
        self.assertTrue(any("Biometric presence failed" in log for log in self.player.logs))

    def test_denied_when_face_missing_camera_unavailable(self):
        """When camera is unavailable or no face detected -> DENIED."""
        self.mock_face_id.identify.return_value = (None, 0.0)

        ok, reason, details = verify_biometric_presence(
            player=self.player,
            face_id=self.mock_face_id,
            voice_id=self.mock_voice_id,
        )

        self.assertFalse(ok)
        self.assertIn("Face not recognized", reason)
        self.assertFalse(details["face"]["matched"])

    def test_denied_when_voice_audio_missing_or_silent(self):
        """When microphone returns empty audio or None -> DENIED."""
        self.mock_face_id.identify.return_value = (TARGET_USER, 0.85)
        self.mock_voice_id.record_sample.return_value = None  # Failed recording

        ok, reason, details = verify_biometric_presence(
            player=self.player,
            face_id=self.mock_face_id,
            voice_id=self.mock_voice_id,
        )

        self.assertFalse(ok)
        self.assertIn("Voice not recognized", reason)
        self.assertFalse(details["voice"]["matched"])

    def test_denied_when_neither_matches(self):
        """When neither Face nor Voice matches -> DENIED."""
        self.mock_face_id.identify.return_value = ("ImpostorFace", 0.20)
        self.mock_voice_id.record_sample.return_value = np.zeros(16000 * 3, dtype=np.float32)
        self.mock_voice_id.identify.return_value = ("ImpostorVoice", 0.50)

        ok, reason, details = verify_biometric_presence(
            player=self.player,
            face_id=self.mock_face_id,
            voice_id=self.mock_voice_id,
        )

        self.assertFalse(ok)
        self.assertFalse(details["face"]["matched"])
        self.assertFalse(details["voice"]["matched"])


class TestToolGatePresenceIntegration(unittest.TestCase):

    @patch("core.confirm.CONFIRM.request", return_value=True)
    @patch("core.presence.verify_biometric_presence")
    def test_dispatch_tool_blocks_sensitive_action_on_presence_failure(
        self, mock_presence, mock_confirm
    ):
        """ASK_AND_WAIT tool is confirmed by user but blocked when presence fails."""
        mock_presence.return_value = (False, "Voice mismatch", {})
        player = MockPlayer()

        result = dispatch_tool(
            tool="computer_settings",
            args={"action": "restart"},
            player=player,
            speak=None,
        )

        self.assertTrue(mock_confirm.called)
        self.assertTrue(mock_presence.called)
        self.assertIn("Denied — biometric presence verification failed", result)

    @patch("core.confirm.CONFIRM.request", return_value=True)
    @patch("core.presence.verify_biometric_presence")
    @patch("core.tool_gate._invoke_with_timeout", return_value="Restarted successfully")
    def test_dispatch_tool_allows_sensitive_action_on_presence_success(
        self, mock_invoke, mock_presence, mock_confirm
    ):
        """ASK_AND_WAIT tool is confirmed and allowed when presence passes."""
        mock_presence.return_value = (True, "Abhinay verified", {})
        player = MockPlayer()

        result = dispatch_tool(
            tool="computer_settings",
            args={"action": "restart"},
            player=player,
            speak=None,
        )
        self.assertTrue(mock_confirm.called)
        self.assertTrue(mock_presence.called)
        self.assertEqual(result, "Restarted successfully")

    def test_invalid_action_rejected_immediately_before_policy_and_execution(self):
        """When the wrong tool is requested with an invalid action (e.g. computer_control for create_folder),
        it is rejected immediately during input validation before policy evaluation or dispatch."""
        player = MockPlayer()
        result = dispatch_tool(
            tool="computer_control",
            args={"action": "create_folder", "path": "desktop"},
            player=player,
            speak=None,
        )
        self.assertTrue(result.startswith("Rejected — 'computer_control' parameters are invalid"))
        self.assertIn("'create_folder' is not a valid action for 'computer_control'", result)
        # Ensure no execution logs were produced
        self.assertFalse(any("[Computer]" in log for log in player.logs))
        self.assertFalse(any("NOTICE: ran" in log for log in player.logs))

    def test_policy_fallback_defaults_unrecognized_actions_to_ask_and_wait(self):
        """When an unrecognized action is checked against get_policy_level() for tools with
        ASK_AND_WAIT actions defined, it defaults safely to ASK_AND_WAIT instead of NOTIFY_ONLY."""
        from core.policy import get_policy_level, ASK_AND_WAIT, HARD_DENY

        for tool in ["computer_control", "file_controller", "browser_control", "computer_settings", "file_processor"]:
            level_desktop = get_policy_level(tool, "some_unrecognized_custom_action", caller_class="desktop")
            self.assertEqual(level_desktop, ASK_AND_WAIT, f"{tool} failed to default unrecognized action to ASK_AND_WAIT")

            level_service = get_policy_level(tool, "some_unrecognized_custom_action", caller_class="service:bugfix")
            self.assertEqual(level_service, HARD_DENY, f"{tool} failed to default unrecognized action to HARD_DENY for service")

    @patch("core.confirm.CONFIRM.request", return_value=True)
    @patch("core.presence.verify_biometric_presence")
    @patch("core.tool_gate._invoke_with_timeout")
    def test_dispatch_tool_blocks_computer_control_on_presence_failure(
        self, mock_invoke, mock_presence, mock_confirm
    ):
        """computer_control with sensitive action (e.g. type) is confirmed but blocked when presence fails."""
        mock_presence.return_value = (False, "Voice not recognized (detected: None, conf: 0.0%)", {})
        player = MockPlayer()

        result = dispatch_tool(
            tool="computer_control",
            args={"action": "type", "text": "sensitive input"},
            player=player,
            speak=None,
        )

        self.assertTrue(mock_confirm.called)
        self.assertTrue(mock_presence.called)
        mock_invoke.assert_not_called()
        self.assertIn("Denied — biometric presence verification failed", result)
        self.assertFalse(any("NOTICE: ran computer_control" in log for log in player.logs))
        self.assertFalse(any("[Computer]" in log for log in player.logs))

    @patch("core.confirm.CONFIRM.request", return_value=True)
    @patch("core.presence.verify_biometric_presence")
    @patch("core.tool_gate._invoke_with_timeout")
    def test_dispatch_tool_blocks_file_controller_create_folder_on_presence_failure(
        self, mock_invoke, mock_presence, mock_confirm
    ):
        """file_controller create_folder is confirmed but blocked when presence fails."""
        mock_presence.return_value = (False, "Voice not recognized (detected: None, conf: 0.0%)", {})
        player = MockPlayer()

        result = dispatch_tool(
            tool="file_controller",
            args={"action": "create_folder", "path": "desktop", "name": "NewFolder"},
            player=player,
            speak=None,
        )

        self.assertTrue(mock_confirm.called)
        self.assertTrue(mock_presence.called)
        mock_invoke.assert_not_called()
        self.assertIn("Denied — biometric presence verification failed", result)
        self.assertFalse(any("NOTICE: ran file_controller" in log for log in player.logs))

    @patch("core.confirm.CONFIRM.request", return_value=True)
    @patch("core.presence.verify_biometric_presence")
    @patch("core.tool_gate._apply_postcondition", side_effect=lambda tool, args, res: res)
    @patch("core.tool_gate._invoke_with_timeout", return_value="Created folder 'NewFolder' in desktop.")
    def test_dispatch_tool_allows_file_controller_create_folder_on_presence_success(
        self, mock_invoke, mock_postcond, mock_presence, mock_confirm
    ):
        """file_controller create_folder succeeds normally when presence check passes."""
        mock_presence.return_value = (True, "Abhinay verified", {})
        player = MockPlayer()

        result = dispatch_tool(
            tool="file_controller",
            args={"action": "create_folder", "path": "desktop", "name": "NewFolder"},
            player=player,
            speak=None,
        )

        self.assertTrue(mock_confirm.called)
        self.assertTrue(mock_presence.called)
        mock_invoke.assert_called_once()
        self.assertEqual(result, "Created folder 'NewFolder' in desktop.")

    @patch("core.confirm.CONFIRM.request", return_value=True)
    @patch("core.presence.verify_biometric_presence")
    @patch("core.tool_gate._invoke_with_timeout")
    def test_dispatch_tool_blocks_coding_agent_on_presence_failure(
        self, mock_invoke, mock_presence, mock_confirm
    ):
        """coding_agent is confirmed but blocked when presence fails."""
        mock_presence.return_value = (False, "Face not recognized", {})
        player = MockPlayer()

        result = dispatch_tool(
            tool="coding_agent",
            args={"description": "write a python script"},
            player=player,
            speak=None,
        )

        self.assertTrue(mock_confirm.called)
        self.assertTrue(mock_presence.called)
        mock_invoke.assert_not_called()
        self.assertIn("Denied — biometric presence verification failed", result)
        self.assertFalse(any("NOTICE: ran coding_agent" in log for log in player.logs))

    def test_executor_short_circuit_recognizes_presence_denial(self):
        """agent/executor.py's _short_circuit_reason correctly classifies biometric presence denial."""
        from agent.executor import _short_circuit_reason
        denial_result = "Denied — biometric presence verification failed: Voice not recognized (detected: None, conf: 0.0%)."
        reason = _short_circuit_reason(denial_result, tool="computer_control")
        self.assertEqual(reason, "approval_denied")


if __name__ == "__main__":
    unittest.main()
