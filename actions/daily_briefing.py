import json
from datetime import datetime
from pathlib import Path

import requests

from config import load_config as _load_config, BASE_DIR
from memory.memory_manager import load_memory as _load_memory
import logging

_log = logging.getLogger("jarvis.daily_briefing")


def _extract_value(entry, _depth: int = 0) -> str:
    """
    Memory values can be doubly/triply nested due to bugs elsewhere:
      "Hyderabad"
      {"value": "Hyderabad"}
      {"value": "{'value': 'Hyderabad'}"}   <- double nested
      "{'value': 'Hyderabad'}"              <- string-of-dict
    Recursively unwraps until a plain clean string is found.
    """
    if _depth > 5:  # safety guard against infinite loops
        return str(entry).strip()

    if entry is None:
        return ""

    if isinstance(entry, dict):
        inner = entry.get("value", "")
        return _extract_value(inner, _depth + 1)

    if isinstance(entry, str):
        s = entry.strip()
        # Detect a string that looks like a dict literal: {'value': '...'}
        if s.startswith("{") and "value" in s:
            try:
                import ast
                parsed = ast.literal_eval(s)
                if isinstance(parsed, dict):
                    return _extract_value(parsed, _depth + 1)
            except Exception:
                pass
        return s

    return str(entry).strip()


def _get_user_city() -> str:
    """
    Pull the user's saved city from memory.
    Checks both top-level 'city' key and 'identity.city' / 'identity.location'.
    """
    mem = _load_memory()

    if "city" in mem:
        val = _extract_value(mem["city"])
        if val:
            return val

    identity = mem.get("identity", {})
    for key in ("city", "location"):
        if key in identity:
            val = _extract_value(identity[key])
            if val:
                return val

    return ""


def _get_weather_summary(city: str, api_key: str) -> str:
    if not city or not api_key:
        return ""
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": api_key, "units": "metric"}
        r = requests.get(url, params=params, timeout=8)
        if r.status_code != 200:
            _log.debug(f"Weather API returned {r.status_code} for city='{city}'")
            return ""
        data = r.json()
        temp = round(data["main"]["temp"])
        desc = data["weather"][0]["description"]
        return f"It's currently {temp}°C and {desc} in {city}."
    except Exception as e:
        _log.warning(f"Weather error: {e}")
        return ""


def _get_pending_reminders() -> list[str]:
    try:
        rem_path = BASE_DIR / "memory" / "reminders.json"
        if not rem_path.exists():
            return []
        reminders = json.loads(rem_path.read_text(encoding="utf-8"))
        today = datetime.now().strftime("%Y-%m-%d")
        todays = [
            r.get("message", "")
            for r in reminders
            if r.get("date") == today and r.get("message")
        ]
        return todays
    except Exception:
        return []


def _get_top_news_headline() -> str:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.news("top world news today", max_results=3))
        if results:
            titles = [r.get("title", "") for r in results[:2] if r.get("title")]
            if titles:
                return " Also in the news: " + "; ".join(titles) + "."
    except Exception as e:
        _log.warning(f"News error: {e}")
    return ""


def daily_briefing(parameters: dict = None, player=None, speak=None) -> str:
    parameters = parameters or {}
    cfg     = _load_config()
    api_key = cfg.get("openweather_api_key", "")

    city = parameters.get("city") or _get_user_city()
    _log.debug(f"Resolved city: '{city}'")

    now  = datetime.now()
    hour = now.hour
    if hour < 12:
        greeting = "Good morning"
    elif hour < 17:
        greeting = "Good afternoon"
    else:
        greeting = "Good evening"

    date_str = now.strftime("%A, %B %d")
    parts = [f"{greeting}. Today is {date_str}."]

    if city:
        weather_line = _get_weather_summary(city, api_key)
        if weather_line:
            parts.append(weather_line)
        else:
            parts.append(f"I couldn't retrieve the weather for {city} right now.")
    else:
        parts.append("I don't know your city yet — tell me where you're located so I can include weather.")

    reminders = _get_pending_reminders()
    if reminders:
        if len(reminders) == 1:
            parts.append(f"You have one reminder today: {reminders[0]}.")
        else:
            joined = "; ".join(reminders)
            parts.append(f"You have {len(reminders)} reminders today: {joined}.")
    else:
        parts.append("You have no reminders scheduled for today.")

    news_line = _get_top_news_headline()
    if news_line:
        parts.append(news_line)

    briefing = " ".join(parts)

    if player:
        try:
            player.write_log("SYS: Daily briefing delivered.")
        except Exception:
            pass

    return briefing
