"""
tests/test_subprocess_timeouts.py
===================================
Follow-up to the tool-contracts phase: computer_settings.py, desktop.py,
game_updater.py, and reminder.py had subprocess.run() calls with no
timeout= at all — a hang there was unbounded even though
core/tool_contracts.py's dispatch-level timeout wrapper exists, since that
wrapper only stops the *caller* from waiting, not the child process.

Two kinds of coverage:
  1. A static AST scan proving every real subprocess.run() call site in
     these four files now has an explicit timeout= — a durable regression
     guard, not just a point-in-time check. Deliberately does NOT flag
     reminder.py's 3 subprocess.run() calls that appear only as *string
     content* inside a generated notification script (they execute later,
     in a different, detached process reminder.py itself never waits on —
     out of scope, see the module docstring's reasoning) — and doesn't
     need to special-case them, since ast.walk only sees real Call nodes,
     never text inside a string literal.
  2. Behavioral tests, per file, that a subprocess.TimeoutExpired at one
     of the fixed call sites is actually caught and converted to the
     same descriptive-string convention every other tool failure uses —
     never propagates as an uncaught exception. subprocess.run itself is
     mocked to raise immediately; the real OS command (including
     restart/shutdown) is never invoked.
"""
from __future__ import annotations

import ast
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_ACTIONS_DIR = Path(__file__).resolve().parent.parent / "actions"
_FIXED_FILES = ["computer_settings.py", "desktop.py", "game_updater.py", "reminder.py"]


def _find_bare_subprocess_run_calls(path: Path) -> list[int]:
    """Returns line numbers of subprocess.run(...) Call nodes with no
    `timeout` keyword argument. Only sees real code — a subprocess.run(...)
    that appears as text inside a string literal (reminder.py's generated
    notification script) is not an ast.Call node at all, so it's never
    flagged."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bare = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_subprocess_run = (
            isinstance(func, ast.Attribute)
            and func.attr == "run"
            and isinstance(func.value, ast.Name)
            and func.value.id == "subprocess"
        )
        if not is_subprocess_run:
            continue
        has_timeout = any(kw.arg == "timeout" for kw in node.keywords)
        if not has_timeout:
            bare.append(node.lineno)
    return bare


class NoBareSubprocessRunTest(unittest.TestCase):
    def test_every_subprocess_run_call_has_a_timeout(self):
        for filename in _FIXED_FILES:
            path = _ACTIONS_DIR / filename
            bare = _find_bare_subprocess_run_calls(path)
            self.assertEqual(bare, [], f"{filename} has unbounded subprocess.run() at line(s): {bare}")


class ComputerSettingsTimeoutHandlingTest(unittest.TestCase):
    def test_restart_timeout_is_caught_and_returns_descriptive_string(self):
        import actions.computer_settings as cs
        if not cs._PYAUTOGUI:
            self.skipTest("pyautogui not available in this environment")

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="shutdown", timeout=5)):
            result = cs.computer_settings({"action": "restart"}, player=None)

        self.assertIsInstance(result, str)
        self.assertIn("restart", result.lower())
        self.assertIn("failed", result.lower())

    def test_shutdown_timeout_is_caught_and_returns_descriptive_string(self):
        import actions.computer_settings as cs
        if not cs._PYAUTOGUI:
            self.skipTest("pyautogui not available in this environment")

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="shutdown", timeout=5)):
            result = cs.computer_settings({"action": "shutdown"}, player=None)

        self.assertIsInstance(result, str)
        self.assertIn("shutdown", result.lower())
        self.assertIn("failed", result.lower())


class DesktopTimeoutHandlingTest(unittest.TestCase):
    """set_wallpaper()'s Windows branch doesn't call subprocess at all
    (uses ctypes directly) — the fixed calls are all Darwin/Linux. Force
    those branches on this Windows test machine by monkeypatching the
    module's _OS constant, matching how desktop.py itself branches."""

    def test_darwin_branch_timeout_is_caught_and_returns_descriptive_string(self):
        import tempfile
        import actions.desktop as desktop

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            fake_image = f.name

        orig_os = desktop._OS
        desktop._OS = "Darwin"
        try:
            with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="osascript", timeout=5)):
                result = desktop.set_wallpaper(fake_image)
        finally:
            desktop._OS = orig_os
            Path(fake_image).unlink(missing_ok=True)

        self.assertIsInstance(result, str)
        self.assertIn("could not set wallpaper", result.lower())

    def test_get_current_wallpaper_darwin_timeout_is_caught(self):
        import actions.desktop as desktop
        orig_os = desktop._OS
        desktop._OS = "Darwin"
        try:
            with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="osascript", timeout=5)):
                result = desktop.get_current_wallpaper()
        finally:
            desktop._OS = orig_os

        self.assertIsInstance(result, str)
        self.assertIn("could not get wallpaper", result.lower())


class GameUpdaterTimeoutHandlingTest(unittest.TestCase):
    def test_system_shutdown_timeout_is_swallowed_and_logged_not_raised(self):
        import actions.game_updater as gu
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="shutdown", timeout=5)):
            try:
                gu._system_shutdown()  # must not raise
            except subprocess.TimeoutExpired:
                self.fail("_system_shutdown() let TimeoutExpired propagate uncaught")

    def test_schedule_windows_timeout_returns_descriptive_string(self):
        import actions.game_updater as gu
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="schtasks", timeout=5)):
            result = gu._schedule_windows(hour=3, minute=0)
        self.assertIsInstance(result, str)
        self.assertTrue(result.startswith("Scheduling failed:"))

    def test_cancel_scheduled_update_timeout_returns_descriptive_string(self):
        import actions.game_updater as gu
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="schtasks", timeout=5)):
            result = gu._cancel_scheduled_update()
        self.assertIsInstance(result, str)
        self.assertTrue(result.startswith("Cancel failed:"))

    def test_get_schedule_status_timeout_is_caught_not_raised(self):
        import actions.game_updater as gu
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="schtasks", timeout=5)):
            result = gu._get_schedule_status()
        self.assertIsInstance(result, str)
        self.assertIn("no scheduled game update found", result.lower())


class ReminderTimeoutHandlingTest(unittest.TestCase):
    """reminder()'s existing outer try/except (unchanged by this fix)
    already converts any scheduling exception to a descriptive string —
    verified here by forcing _schedule_windows itself to raise, without
    touching the real ~/.jarvis/reminders directory _write_notify_script
    would otherwise create."""

    def test_scheduling_timeout_is_caught_by_existing_outer_handler(self):
        import actions.reminder as reminder
        from datetime import datetime, timedelta
        from pathlib import Path as _Path

        future = datetime.now() + timedelta(days=1)

        with patch.object(reminder, "_write_notify_script", return_value=_Path("fake_script.py")), \
             patch.object(reminder, "get_os", return_value="windows"), \
             patch.object(reminder, "_schedule_windows",
                           side_effect=subprocess.TimeoutExpired(cmd="schtasks", timeout=10)):
            result = reminder.reminder({
                "date": future.strftime("%Y-%m-%d"),
                "time": "09:00",
                "message": "test reminder",
            })

        self.assertEqual(result, "Something went wrong while scheduling the reminder.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
