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
    city = params.get("city")
    when = params.get("time", "today")

    if not city or not isinstance(city, str) or not city.strip():
        # Fallback 1: check memory
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
                    try:
                        from memory.memory_manager import update_memory
                        update_memory({"identity": {"city": city}})
                    except Exception as e:
                        _log.warning("Could not save city to memory: %s", e)
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
