# core/tool_dispatch.py
"""
Canonical table of shared action tools — the single source both
main.py's interactive _execute_tool() and agent/executor.py's autonomous
_call_tool() route through, instead of each maintaining its own
independently-drifting copy of "tool name -> action function" wiring.

Each entry normalizes its target action function's own signature (some
take `response=`, some take `speak=`, some take `session_memory=`, none
consistently) into a single (parameters, player, speak) -> str shape.

Deliberately excludes tools that are bound to live session/UI state
that only main.py's JarvisXL instance has (save_memory, identify_user,
register_user, shutdown_jarvis) and agent/executor.py's own
generated_code fallback, which isn't a real action module.
"""
from typing import Callable, Optional


def _open_app(parameters, player, speak):
    from actions.open_app import open_app
    return open_app(parameters=parameters, response=None, player=player) or \
        f"Opened {parameters.get('app_name')}."


def _weather_report(parameters, player, speak):
    from actions.weather_report import weather_action
    return weather_action(parameters=parameters, player=player) or "Weather delivered."


def _browser_control(parameters, player, speak):
    from actions.browser_control import browser_control
    return browser_control(parameters=parameters, player=player) or "Done."


def _file_controller(parameters, player, speak):
    from actions.file_controller import file_controller
    return file_controller(parameters=parameters, player=player) or "Done."


def _send_message(parameters, player, speak):
    from actions.send_message import send_message
    return send_message(parameters=parameters, response=None, player=player, session_memory=None) or \
        f"Message sent to {parameters.get('receiver')}."


def _reminder(parameters, player, speak):
    from actions.reminder import reminder
    return reminder(parameters=parameters, response=None, player=player) or "Reminder set."


def _youtube_video(parameters, player, speak):
    from actions.youtube_video import youtube_video
    return youtube_video(parameters=parameters, response=None, player=player) or "Done."


def _screen_process(parameters, player, speak):
    from actions.screen_processor import screen_process
    r = screen_process(parameters=parameters, response=None, player=player, session_memory=None)
    return r if isinstance(r, str) and r else "Screen analyzed."


def _computer_settings(parameters, player, speak):
    from actions.computer_settings import computer_settings
    return computer_settings(parameters=parameters, response=None, player=player) or "Done."


def _desktop_control(parameters, player, speak):
    from actions.desktop import desktop_control
    return desktop_control(parameters=parameters, player=player) or "Done."


def _code_helper(parameters, player, speak):
    from actions.code_helper import code_helper
    return code_helper(parameters=parameters, player=player, speak=speak) or "Done."


def _dev_agent(parameters, player, speak):
    from actions.dev_agent import dev_agent
    return dev_agent(parameters=parameters, player=player, speak=speak) or "Done."


def _web_search(parameters, player, speak):
    from actions.web_search import web_search
    return web_search(parameters=parameters, player=player) or "Done."


def _file_processor(parameters, player, speak):
    from actions.file_processor import file_processor
    if not parameters.get("file_path") and getattr(player, "current_file", None):
        parameters["file_path"] = player.current_file
    return file_processor(parameters=parameters, player=player, speak=speak) or "Done."


def _computer_control(parameters, player, speak):
    from actions.computer_control import computer_control
    return computer_control(parameters=parameters, player=player) or "Done."


def _game_updater(parameters, player, speak):
    from actions.game_updater import game_updater
    return game_updater(parameters=parameters, player=player, speak=speak) or "Done."


def _flight_finder(parameters, player, speak):
    from actions.flight_finder import flight_finder
    return flight_finder(parameters=parameters, player=player) or "Done."


TOOL_DISPATCH: dict[str, Callable[[dict, object, Optional[Callable]], str]] = {
    "open_app":          _open_app,
    "weather_report":    _weather_report,
    "browser_control":   _browser_control,
    "file_controller":   _file_controller,
    "send_message":      _send_message,
    "reminder":          _reminder,
    "youtube_video":     _youtube_video,
    "screen_process":    _screen_process,
    "computer_settings": _computer_settings,
    "desktop_control":   _desktop_control,
    "code_helper":       _code_helper,
    "dev_agent":         _dev_agent,
    "web_search":        _web_search,
    "file_processor":    _file_processor,
    "computer_control":  _computer_control,
    "game_updater":      _game_updater,
    "flight_finder":     _flight_finder,
}
