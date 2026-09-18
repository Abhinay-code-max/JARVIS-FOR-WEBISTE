import re
import requests

from config import load_config
import logging

_log = logging.getLogger("jarvis.weather")


def _get_api_key() -> str:
    return load_config().get("openweather_api_key", "").strip()


def _log_msg(msg: str, player=None) -> None:
    _log.debug(f"{msg}")
    if player:
        try:
            player.write_log(f"[weather] {msg}")
        except Exception:
            pass


def _extract_city_from_text(text: str) -> str:
    """
    Extract an explicitly named city or location from natural language text/description.
    e.g. 'weather in Tokyo', 'forecast for San Francisco', 'weather report for Paris'.
    Returns empty string if no specific city is named.
    """
    if not text or not isinstance(text, str):
        return ""
    generic = {
        "here", "there", "my area", "the area", "this area",
        "my city", "the city", "this city", "my town", "the town",
        "today", "tomorrow", "tonight", "now", "currently",
        "this morning", "this evening", "this afternoon", "the morning",
        "the evening", "the afternoon", "the weekend", "this week",
        "me", "us", "him", "her", "them", "someone", "anyone",
    }
    patterns = [
        r"\b(?:weather|forecast|temperature|rain|conditions?)\s+(?:in|for|at|of)\s+([A-Za-z\s\-]+?)(?:\s+(?:today|tomorrow|now|currently|this week|please|right now)|[?.!,;]|$)",
        r"\b(?:in|for|at|of)\s+([A-Za-z\s\-]+?)(?:\s+(?:today|tomorrow|now|currently|this week|please|right now)|[?.!,;]|$)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            cand = m.group(1).strip()
            cand = re.sub(r"\s+(?:today|tomorrow|now|currently|this week|please|right now)$", "", cand, flags=re.IGNORECASE).strip()
            if cand.lower() not in generic and len(cand) > 1:
                return cand.title()
    return ""


def _get_user_city() -> str:
    """
    Pull the user's saved city from memory.
    Checks top-level 'city' key and 'identity.city' / 'identity.location'.
    """
    try:
        from actions.daily_briefing import _get_user_city as _briefing_city
        city = _briefing_city()
        if city:
            return city
    except Exception:
        pass

    try:
        from memory.memory_manager import load_memory
        mem = load_memory()
        if "city" in mem:
            val = mem["city"]
            if isinstance(val, dict):
                val = val.get("value", "")
            if isinstance(val, str) and val.strip():
                return val.strip()
        identity = mem.get("identity", {})
        for key in ("city", "location"):
            if key in identity:
                val = identity[key]
                if isinstance(val, dict):
                    val = val.get("value", "")
                if isinstance(val, str) and val.strip():
                    return val.strip()
    except Exception:
        pass
    return ""


def weather_action(
    parameters: dict = None,
    player=None,
    speak=None,
    session_memory=None,
) -> str:
    params = parameters or {}
    city = params.get("city") or params.get("location")
    when = params.get("time", "today")

    # If city not explicitly passed in parameters, check text fields before falling back to memory
    if not city or not isinstance(city, str) or not city.strip():
        for field in ("query", "description", "goal", "prompt", "text"):
            val = params.get(field)
            if val and isinstance(val, str):
                extracted = _extract_city_from_text(val)
                if extracted:
                    city = extracted
                    break

    # If still no city, Fallback 1: check saved memory
    if not city or not isinstance(city, str) or not city.strip():
        city = _get_user_city()

    if not city or not isinstance(city, str) or not city.strip():
        # Fallback 2: clarify with user if live player is present
        if player is not None:
            try:
                from core.confirm import CONFIRM
                prompt = "Which city would you like the weather report for, sir?"
                answer = CONFIRM.request_clarification(player, prompt, speak=speak)
                if answer and isinstance(answer, str) and answer.strip():
                    city = answer.strip()
            except Exception as e:
                _log.warning("Clarification fallback error: %s", e)

    if not city or not isinstance(city, str) or not city.strip():
        msg = "Please specify a city for the weather report."
        _log_msg(msg, player)
        return msg

    city = city.strip()
    api_key = _get_api_key()

    if not api_key:
        msg = "Weather API key is not configured. Please add openweather_api_key to config/api_keys.json."
        _log_msg(msg, player)
        return msg

    try:
        # Use OpenWeatherMap Current Weather API
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": api_key,
            "units": "metric",  # Celsius
        }

        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 401:
            msg = "Weather API key is invalid or not yet activated (can take up to 10 minutes after signup)."
            _log_msg(msg, player)
            return msg

        if response.status_code == 404:
            msg = f"Could not find weather data for '{city}'. Please check the city name."
            _log_msg(msg, player)
            return msg

        if response.status_code != 200:
            msg = f"Weather service returned an error (status {response.status_code})."
            _log_msg(msg, player)
            return msg

        data = response.json()

        temp        = round(data["main"]["temp"])
        feels_like  = round(data["main"]["feels_like"])
        humidity    = data["main"]["humidity"]
        description = data["weather"][0]["description"].capitalize()
        wind_speed  = data["wind"]["speed"]
        city_name   = data["name"]
        country     = data.get("sys", {}).get("country", "")

        result = (
            f"The weather in {city_name}, {country} is currently {description}, "
            f"with a temperature of {temp}°C (feels like {feels_like}°C). "
            f"Humidity is at {humidity}%, and wind speed is {wind_speed} meters per second."
        )

        _log_msg(f"OK — {city_name}: {temp}°C, {description}", player)
        return result

    except requests.exceptions.Timeout:
        msg = "Weather service timed out. Please try again."
        _log_msg(msg, player)
        return msg
    except requests.exceptions.ConnectionError:
        msg = "Could not connect to the weather service. Check your internet connection."
        _log_msg(msg, player)
        return msg
    except Exception as e:
        msg = f"Could not retrieve weather data: {e}"
        _log_msg(msg, player)
        return msg
