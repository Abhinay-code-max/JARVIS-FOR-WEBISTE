"""
core/calendar_auth.py
======================
Google Calendar OAuth2 (desktop-app loopback flow) for
core/proactive.py's calendar_event_upcoming trigger. Read-only —
calendar.events.readonly scope only. Nothing in this module, or anything
that calls it, ever writes to the calendar.

One-time external setup (do this before anything here will work; also
see config's calendar_enabled flag, which core/installer.py gates the
google-auth/google-auth-oauthlib/google-api-python-client installs on —
add "calendar_enabled": true to config/api_keys.json to opt in):

  1. https://console.cloud.google.com/ — create a project (or reuse one),
     then APIs & Services -> Library -> enable "Google Calendar API".

  2. APIs & Services -> OAuth consent screen:
       - User Type: External (required for a personal Gmail account).
       - Scopes: add .../auth/calendar.events.readonly.
       - Publish the app to **Production**. This is the step most worth
         getting right: an app left in "Testing" status gets refresh
         tokens that silently expire after 7 days, which for an
         always-on assistant means calendar access quietly dying every
         week. Personal-use apps (a single developer reading only their
         own data) are exempt from Google's manual verification review,
         so Production status here doesn't require anything beyond
         publishing — you'll still see a one-time "unverified app"
         click-through warning during step 4 below, that's expected.

  3. APIs & Services -> Credentials -> Create Credentials -> OAuth
     client ID -> Application type "Desktop app" -> download the JSON
     -> save it as CLIENT_SECRET_PATH below (~/.jarvis/google_client_secret.json).

  4. Run `python -m core.calendar_auth` once from the repo root. It
     opens your browser, you sign in and approve, and the resulting
     token is cached at TOKEN_PATH (~/.jarvis/google_calendar_token.json).
     Nothing else to do after that — get_calendar_service() refreshes
     the access token silently using the stored refresh token. If the
     refresh token itself is ever revoked, re-run this step.
"""
from __future__ import annotations

import logging
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

_log = logging.getLogger("jarvis.calendar_auth")

SCOPES = ["https://www.googleapis.com/auth/calendar.events.readonly"]

_JARVIS_DIR         = Path.home() / ".jarvis"
CLIENT_SECRET_PATH  = _JARVIS_DIR / "google_client_secret.json"
TOKEN_PATH          = _JARVIS_DIR / "google_calendar_token.json"


def _load_credentials() -> Credentials | None:
    if not TOKEN_PATH.exists():
        return None
    try:
        return Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    except Exception:
        _log.warning("Could not read stored calendar token at %s", TOKEN_PATH, exc_info=True)
        return None


def _save_credentials(creds: Credentials) -> None:
    _JARVIS_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")
    try:
        TOKEN_PATH.chmod(0o600)   # owner read/write only — same convention as reminder.py's scripts
    except Exception:
        pass


def get_calendar_service():
    """Returns an authorized Google Calendar API client, or None if
    calendar auth hasn't been completed yet (see module docstring) or
    the stored token can't be refreshed (e.g. access was revoked —
    needs re-authorization via run_local_auth_flow() the same way).
    Never launches a browser itself — that's run_local_auth_flow() only,
    meant to be run explicitly and interactively, never from inside
    core/proactive.py's poll loop."""
    creds = _load_credentials()
    if creds is None:
        return None

    if not creds.valid:
        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                _save_credentials(creds)
            except Exception:
                _log.warning(
                    "Calendar token refresh failed — re-run `python -m core.calendar_auth`",
                    exc_info=True,
                )
                return None
        else:
            return None

    return build("calendar", "v3", credentials=creds, cache_discovery=False)


def run_local_auth_flow() -> None:
    """One-time interactive setup — opens a browser for the user to sign
    in and grant calendar.events.readonly access, then caches the
    resulting token at TOKEN_PATH. Run manually
    (`python -m core.calendar_auth`); never called automatically."""
    if not CLIENT_SECRET_PATH.exists():
        raise FileNotFoundError(
            f"No client secret found at {CLIENT_SECRET_PATH}. Download it "
            "from Google Cloud Console (see this module's docstring, step 3) "
            "and save it there first."
        )
    flow  = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET_PATH), SCOPES)
    creds = flow.run_local_server(port=0)
    _save_credentials(creds)
    print(f"Calendar authorized. Token saved to {TOKEN_PATH}.")


if __name__ == "__main__":
    run_local_auth_flow()
