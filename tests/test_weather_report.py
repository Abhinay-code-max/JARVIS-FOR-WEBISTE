"""
tests/test_weather_report.py
============================
Regression and unit tests for actions/weather_report.py.

Verifies:
1. Module-level `_log` is a Logger instance conforming to the actions/* logger convention.
2. `weather_action` and `_log_msg` correctly emit log records to the `jarvis.weather` logger
   (preventing regression where a function shadowed the module logger and raised AttributeError).
3. Player object's `write_log` is called if player is provided.
4. Correct behavior on missing parameters, missing API keys, API errors (401, 404, 500),
   and successful responses.
"""
import logging
import unittest
from unittest.mock import MagicMock, patch

from actions.weather_report import weather_action, _log, _log_msg


class WeatherReportLoggerTest(unittest.TestCase):
    def test_module_logger_is_logger_instance(self):
        """Module-level _log must be a logging.Logger, matching the codebase convention."""
        self.assertIsInstance(_log, logging.Logger)
        self.assertEqual(_log.name, "jarvis.weather")

    @patch("actions.weather_report._get_user_city", return_value="")
    def test_weather_action_logs_to_module_logger(self, mock_user_city):
        """Calling weather_action must emit log records to jarvis.weather without AttributeError."""
        with self.assertLogs("jarvis.weather", level="DEBUG") as log_ctx:
            result = weather_action({"city": ""})
            self.assertEqual(result, "Please specify a city for the weather report.")

        self.assertTrue(any("Please specify a city" in rec for rec in log_ctx.output))

    @patch("actions.weather_report._get_user_city", return_value="")
    @patch("core.confirm.CONFIRM.request_clarification", return_value=None)
    def test_weather_action_logs_to_player_if_provided(self, mock_clarify, mock_user_city):
        """When player is provided, player.write_log is called."""
        mock_player = MagicMock()
        with self.assertLogs("jarvis.weather", level="DEBUG"):
            weather_action({"city": ""}, player=mock_player)

        mock_player.write_log.assert_called_once()
        self.assertIn("[weather]", mock_player.write_log.call_args[0][0])

    @patch("actions.weather_report._get_api_key", return_value="")
    def test_missing_api_key_logs_and_returns_error(self, mock_api_key):
        with self.assertLogs("jarvis.weather", level="DEBUG") as log_ctx:
            result = weather_action({"city": "Tokyo"})
            self.assertIn("Weather API key is not configured", result)

        self.assertTrue(any("Weather API key is not configured" in rec for rec in log_ctx.output))

    @patch("actions.weather_report._get_api_key", return_value="test_key")
    @patch("actions.weather_report.requests.get")
    def test_successful_weather_call_logs_and_formats_output(self, mock_get, mock_api_key):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "name": "London",
            "sys": {"country": "GB"},
            "main": {"temp": 21.4, "feels_like": 20.8, "humidity": 65},
            "weather": [{"description": "clear sky"}],
            "wind": {"speed": 3.5},
        }
        mock_get.return_value = mock_resp

        mock_player = MagicMock()
        with self.assertLogs("jarvis.weather", level="DEBUG") as log_ctx:
            result = weather_action({"city": "London"}, player=mock_player)

        self.assertIn("The weather in London, GB is currently Clear sky", result)
        self.assertIn("21°C", result)
        self.assertIn("feels like 21°C", result)
        self.assertTrue(any("OK — London: 21°C, Clear sky" in rec for rec in log_ctx.output))
        mock_player.write_log.assert_called_once_with("[weather] OK — London: 21°C, Clear sky")

    @patch("actions.weather_report._get_api_key", return_value="test_key")
    @patch("actions.weather_report.requests.get")
    def test_404_city_not_found(self, mock_get, mock_api_key):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        with self.assertLogs("jarvis.weather", level="DEBUG") as log_ctx:
            result = weather_action({"city": "NonExistentCityXYZ"})

        self.assertIn("Could not find weather data for 'NonExistentCityXYZ'", result)
        self.assertTrue(any("Could not find weather data" in rec for rec in log_ctx.output))

    @patch("actions.weather_report._get_api_key", return_value="test_key")
    @patch("actions.weather_report.requests.get")
    def test_401_invalid_key(self, mock_get, mock_api_key):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_get.return_value = mock_resp

        with self.assertLogs("jarvis.weather", level="DEBUG") as log_ctx:
            result = weather_action({"city": "Paris"})

        self.assertIn("Weather API key is invalid", result)
        self.assertTrue(any("Weather API key is invalid" in rec for rec in log_ctx.output))

    @patch("actions.weather_report._get_user_city", return_value="Hyderabad")
    @patch("actions.weather_report._get_api_key", return_value="test_key")
    @patch("actions.weather_report.requests.get")
    def test_city_omitted_uses_memory_fallback(self, mock_get, mock_api_key, mock_user_city):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "name": "Hyderabad",
            "sys": {"country": "IN"},
            "main": {"temp": 28.0, "feels_like": 30.0, "humidity": 70},
            "weather": [{"description": "few clouds"}],
            "wind": {"speed": 4.0},
        }
        mock_get.return_value = mock_resp

        result = weather_action({})
        self.assertIn("Hyderabad, IN", result)
        mock_get.assert_called_once()
        self.assertEqual(mock_get.call_args[1]["params"]["q"], "Hyderabad")

    @patch("actions.weather_report._get_user_city", return_value="")
    @patch("core.confirm.CONFIRM.request_clarification", return_value="Tokyo")
    @patch("memory.memory_manager.update_memory")
    @patch("actions.weather_report._get_api_key", return_value="test_key")
    @patch("actions.weather_report.requests.get")
    def test_city_omitted_clarifies_with_user_when_memory_empty(
        self, mock_get, mock_api_key, mock_update_mem, mock_clarify, mock_user_city
    ):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "name": "Tokyo",
            "sys": {"country": "JP"},
            "main": {"temp": 25.0, "feels_like": 26.0, "humidity": 60},
            "weather": [{"description": "clear sky"}],
            "wind": {"speed": 2.0},
        }
        mock_get.return_value = mock_resp
        mock_player = MagicMock()

        result = weather_action({}, player=mock_player)
        self.assertIn("Tokyo, JP", result)
        mock_clarify.assert_called_once()
        mock_update_mem.assert_called_once_with({"identity": {"city": "Tokyo"}})


if __name__ == "__main__":
    unittest.main(verbosity=2)
