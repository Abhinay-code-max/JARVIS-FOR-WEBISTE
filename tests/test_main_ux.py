"""
tests/test_main_ux.py
=====================
Tests for startup queuing feedback and wake-word rejection HUD indicators in main.py.
"""
import time
import threading
import unittest
from unittest.mock import MagicMock, patch

from main import JarvisXL


class MainUXFeedbackTest(unittest.TestCase):
    def setUp(self):
        self.mock_ui = MagicMock()
        self.jarvis = JarvisXL(self.mock_ui)

    def test_wake_gate_rejection_shows_hud_feedback(self):
        # Ensure awake window is closed
        self.jarvis._wake_until = 0.0

        # Speak text without wake word
        result = self.jarvis._wake_gate("what is the weather")

        # Must return None (dropped)
        self.assertIsNone(result)

        # Must write visible log to HUD
        self.mock_ui.write_log.assert_called_with('🔒 Heard: "what is the weather" — say \'Hey JARVIS\' to activate')

        # Must set HUD state to inform the user
        self.mock_ui.set_state.assert_called_with("LOCKED (SAY 'HEY JARVIS')")

    def test_wake_gate_accepts_wake_word(self):
        self.jarvis._wake_until = 0.0
        result = self.jarvis._wake_gate("Hey JARVIS, what is the weather")
        self.assertEqual(result, "what is the weather")
        self.assertGreater(self.jarvis._wake_until, time.time())

    def test_wake_gate_within_awake_window(self):
        # Awake window is open
        self.jarvis._wake_until = time.time() + 10.0
        result = self.jarvis._wake_gate("what time is it")
        self.assertEqual(result, "what time is it")
        # Should not set locked state
        self.mock_ui.set_state.assert_not_called()

    def test_wake_gate_barge_in_bypass(self):
        self.jarvis._wake_until = 0.0
        self.jarvis._barge_in_pending = True
        result = self.jarvis._wake_gate("stop talking and search for Python")
        self.assertEqual(result, "stop talking and search for Python")
        self.assertFalse(self.jarvis._barge_in_pending)
        self.assertGreater(self.jarvis._wake_until, time.time())

    def test_wake_gate_disabled_in_config(self):
        self.jarvis._config["wake_word_enabled"] = False
        self.jarvis._wake_until = 0.0
        result = self.jarvis._wake_gate("any random sentence")
        self.assertEqual(result, "any random sentence")

    def test_startup_window_queues_command_with_hud_notice(self):
        # System is not ready yet
        self.assertFalse(self.jarvis._is_ready.is_set())

        self.jarvis._enqueue_command("open WhatsApp")

        # Must write queuing feedback to HUD
        self.mock_ui.write_log.assert_called_with('⏳ Starting up — queued: "open WhatsApp"')

        # Must place in queue rather than silently dropping
        self.assertFalse(self.jarvis._text_queue.empty())
        queued_item = self.jarvis._text_queue.get_nowait()
        self.assertEqual(queued_item, "open WhatsApp")

    def test_startup_window_speech_intake_and_execution_on_ready(self):
        # System is initially not ready
        self.assertFalse(self.jarvis._is_ready.is_set())

        # Mock _process_message
        processed = []
        self.jarvis._process_message = lambda text: processed.append(text)

        # Simulate speech captured during startup with wake word
        self.jarvis._wake_until = 0.0
        gated = self.jarvis._wake_gate("Hey JARVIS open YouTube")
        self.assertEqual(gated, "open YouTube")

        # Enqueue the command while still starting up
        self.jarvis._enqueue_command(gated)
        self.mock_ui.write_log.assert_called_with('⏳ Starting up — queued: "open YouTube"')

        # Run text command loop iteration in background
        t = threading.Thread(target=self._run_one_command_iteration, daemon=True)
        t.start()

        # Give it a moment to block on _is_ready.wait()
        time.sleep(0.05)
        self.assertEqual(processed, [])  # Not processed yet

        # System finishes warmup / STT loading
        self.jarvis._is_ready.set()
        t.join(timeout=1.0)

        # Verified processed once ready
        self.assertEqual(processed, ["open YouTube"])

    def _run_one_command_iteration(self):
        text = self.jarvis._text_queue.get(timeout=1.0)
        if not self.jarvis._is_ready.is_set():
            self.jarvis._is_ready.wait()
        self.jarvis._process_message(text)

    def test_enqueue_duplicate_command_ignored(self):
        self.jarvis._is_ready.set()
        self.jarvis._enqueue_command("check weather")
        self.assertEqual(self.jarvis._text_queue.qsize(), 1)

        # Send exact duplicate immediately
        self.jarvis._enqueue_command("check weather")
        # Should not duplicate in queue
        self.assertEqual(self.jarvis._text_queue.qsize(), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
