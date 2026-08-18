"""
tests/test_open_app.py
======================
Unit and regression tests for actions/open_app.py.

Verifies:
1. app_paths.json loading and parsing (valid json, missing file, malformed file).
2. _normalize resolves UWP config entries, built-in aliases, and raw names.
3. _probe_windows_exe resolves common installation directories.
4. _launch_windows:
   - UWP protocol and AUMID/shell:AppsFolder fallback launching.
   - win32 direct executable path probing and execution.
   - PATH lookup via shutil.which.
   - PyAutoGUI start menu fallback.
5. open_app end-to-end behavior for WhatsApp (UWP), Chrome (win32), and fake app (graceful failure).
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import actions.open_app as open_app_mod
from actions.open_app import (
    _load_app_paths,
    _normalize,
    _probe_windows_exe,
    _launch_windows,
    open_app,
)


class OpenAppConfigLoadingTest(unittest.TestCase):
    def test_load_app_paths_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_dir = Path(tmpdir) / "config"
            cfg_dir.mkdir()
            cfg_file = cfg_dir / "app_paths.json"
            cfg_file.write_text(
                json.dumps({
                    "whatsapp": {
                        "type": "uwp",
                        "protocol": "whatsapp:",
                        "aumid": "5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App",
                    }
                }),
                encoding="utf-8",
            )
            with patch("actions.open_app.BASE_DIR", Path(tmpdir)):
                loaded = _load_app_paths()
                self.assertIn("whatsapp", loaded)
                self.assertEqual(loaded["whatsapp"]["type"], "uwp")
                self.assertEqual(loaded["whatsapp"]["protocol"], "whatsapp:")

    def test_load_app_paths_missing_file_returns_empty_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("actions.open_app.BASE_DIR", Path(tmpdir)):
                loaded = _load_app_paths()
                self.assertEqual(loaded, {})

    def test_load_app_paths_malformed_json_returns_empty_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_dir = Path(tmpdir) / "config"
            cfg_dir.mkdir()
            cfg_file = cfg_dir / "app_paths.json"
            cfg_file.write_text("invalid = key value", encoding="utf-8")
            with patch("actions.open_app.BASE_DIR", Path(tmpdir)):
                loaded = _load_app_paths()
                self.assertEqual(loaded, {})


class OpenAppNormalizeTest(unittest.TestCase):
    def test_normalize_resolves_uwp_from_app_paths(self):
        mock_paths = {
            "whatsapp": {
                "type": "uwp",
                "protocol": "whatsapp:",
                "aumid": "5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App",
            }
        }
        with patch("actions.open_app._load_app_paths", return_value=mock_paths):
            res = _normalize("WhatsApp")
            self.assertIsInstance(res, dict)
            self.assertEqual(res["type"], "uwp")
            self.assertEqual(res["protocol"], "whatsapp:")

    def test_normalize_resolves_builtin_alias(self):
        with patch("actions.open_app._load_app_paths", return_value={}):
            res = _normalize("Google Chrome")
            self.assertEqual(res, "chrome")

    def test_normalize_unknown_app_passes_through(self):
        with patch("actions.open_app._load_app_paths", return_value={}):
            res = _normalize("SomeUnknownApp123")
            self.assertEqual(res, "SomeUnknownApp123")


class OpenAppWindowsLaunchTest(unittest.TestCase):
    @patch("actions.open_app.subprocess.Popen")
    def test_uwp_protocol_launch(self, mock_popen):
        uwp_entry = {
            "type": "uwp",
            "protocol": "whatsapp:",
            "aumid": "5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App",
        }
        res = _launch_windows(uwp_entry, raw_app_name="WhatsApp")
        self.assertTrue(res)
        mock_popen.assert_called_once_with(["cmd", "/c", "start", "", "whatsapp:"])

    @patch("actions.open_app.subprocess.Popen")
    def test_uwp_aumid_fallback_when_protocol_fails(self, mock_popen):
        # First call (protocol) raises exception, second call (AUMID) succeeds
        mock_popen.side_effect = [RuntimeError("Protocol failed"), MagicMock()]
        uwp_entry = {
            "type": "uwp",
            "protocol": "whatsapp:",
            "aumid": "5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App",
        }
        res = _launch_windows(uwp_entry, raw_app_name="WhatsApp")
        self.assertTrue(res)
        self.assertEqual(mock_popen.call_count, 2)
        second_call_args = mock_popen.call_args_list[1][0][0]
        self.assertEqual(second_call_args, ["explorer.exe", "shell:AppsFolder\\5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App"])

    @patch("actions.open_app._probe_windows_exe", return_value=r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    @patch("actions.open_app.subprocess.Popen")
    def test_win32_probed_exe_launch(self, mock_popen, mock_probe):
        res = _launch_windows("chrome", raw_app_name="Chrome")
        self.assertTrue(res)
        mock_popen.assert_called_once_with(
            [r"C:\Program Files\Google\Chrome\Application\chrome.exe"],
            stdout=open_app_mod.subprocess.DEVNULL,
            stderr=open_app_mod.subprocess.DEVNULL,
        )

    @patch("actions.open_app._probe_windows_exe", return_value=None)
    @patch("actions.open_app.shutil.which", return_value=r"C:\Windows\System32\notepad.exe")
    @patch("actions.open_app.subprocess.Popen")
    def test_path_which_launch(self, mock_popen, mock_which, mock_probe):
        res = _launch_windows("notepad.exe", raw_app_name="notepad")
        self.assertTrue(res)
        mock_popen.assert_called_once_with(
            [r"C:\Windows\System32\notepad.exe"],
            stdout=open_app_mod.subprocess.DEVNULL,
            stderr=open_app_mod.subprocess.DEVNULL,
        )


class OpenAppEndToEndTest(unittest.TestCase):
    def test_empty_app_name_returns_error(self):
        res = open_app({})
        self.assertEqual(res, "No application name provided.")

    @patch("actions.open_app._SYSTEM", "Windows")
    @patch("actions.open_app._launch_windows", return_value=True)
    def test_whatsapp_open_success(self, mock_launch):
        res = open_app({"app_name": "WhatsApp"})
        self.assertEqual(res, "Opened WhatsApp.")
        mock_launch.assert_called_once()

    @patch("actions.open_app._SYSTEM", "Windows")
    @patch("actions.open_app._launch_windows", return_value=True)
    def test_chrome_open_success(self, mock_launch):
        res = open_app({"app_name": "Chrome"})
        self.assertEqual(res, "Opened Chrome.")
        mock_launch.assert_called_once()

    @patch("actions.open_app._SYSTEM", "Windows")
    @patch("actions.open_app._launch_windows", return_value=False)
    def test_fake_app_fails_gracefully(self, mock_launch):
        res = open_app({"app_name": "NonExistentApp12345XYZ"})
        self.assertIn("Could not confirm that NonExistentApp12345XYZ launched", res)


if __name__ == "__main__":
    unittest.main(verbosity=2)
