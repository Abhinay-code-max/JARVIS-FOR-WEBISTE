"""
core/hermes_auth.py
=====================
Bearer-token storage and verification for api/hermes_app.py's network
boundary (headless-extraction phase, Step 1.5) — one token per
core/policy.py caller_class.

Different shape than core/calendar_auth.py's and core/github_ci_auth.py's
tokens: those are issued by an *external* service and pasted in by the
user (calendar_auth.py via a real OAuth round-trip, github_ci_auth.py via
getpass — see that module for why paste-in makes sense there). There is
nothing to paste here: Hermes is the service *issuing* these tokens, not
consuming someone else's. The plan's own phrasing ("stored the same way
the GitHub PAT is stored... via getpass") is read here as "same local-
file-0600 storage convention, entered/shown non-interactively-echoed" —
getpass()'s specific hidden-*input* mechanism doesn't apply to a mint
flow that *generates and displays* a brand-new secret rather than
accepting a pasted one. Flagged explicitly rather than silently forcing
a getpass() call that wouldn't fit what's actually happening.

Storage: one file per caller_class under ~/.jarvis/hermes_tokens/ (same
~/.jarvis/ convention as the calendar/GitHub credentials), 0600, named by
caller_class with ':' replaced by '_' (not a valid Windows filename
character) — e.g. service:bugfix -> service_bugfix.token.

get_token() re-reads from disk on every call, deliberately never caching
in a module-level variable — the plan's explicit requirement, so a
revoked/re-minted token takes effect on the very next request rather
than needing a process restart.

One-time setup per caller_class you actually need issued (only desktop
and service:support are minted in this phase — see Step 1.5/1.8; the
other three get policy rows but no token yet):

  python -m core.hermes_auth mint desktop
  python -m core.hermes_auth mint service:support

Each prints the new token exactly once — copy it somewhere safe
immediately. Minting again for the same caller_class overwrites the old
token with no grace period (a real revoke-and-reissue, not additive) —
the previous token stops working the instant the new one is saved.
"""
from __future__ import annotations

import logging
import secrets
import sys
from pathlib import Path

from core.policy import CALLER_CLASSES

_log = logging.getLogger("jarvis.hermes_auth")

_JARVIS_DIR = Path.home() / ".jarvis"
_TOKENS_DIR = _JARVIS_DIR / "hermes_tokens"

_TOKEN_BYTES = 32  # secrets.token_urlsafe(32) -> 256 bits of entropy


def _token_path(caller_class: str) -> Path:
    return _TOKENS_DIR / f"{caller_class.replace(':', '_')}.token"


def get_token(caller_class: str) -> str | None:
    """Returns the stored token for `caller_class`, or None if it hasn't
    been minted yet. Re-reads from disk every call — never cached — so a
    freshly re-minted or removed token takes effect immediately, without
    a process restart."""
    path = _token_path(caller_class)
    if not path.exists():
        return None
    try:
        token = path.read_text(encoding="utf-8").strip()
        return token or None
    except Exception:
        _log.warning("Could not read stored Hermes token for %s", caller_class, exc_info=True)
        return None


def mint_token(caller_class: str) -> str:
    """Generates a new cryptographically random token for `caller_class`,
    overwriting any previously stored one (revoke-and-reissue, no
    dual-token grace period in this phase). Returns the new token so a
    caller (run_local_mint_flow(), or a future admin path) can display it
    exactly once — never logged, never stored anywhere but this one file."""
    if caller_class not in CALLER_CLASSES:
        raise ValueError(f"unknown caller_class {caller_class!r} — must be one of {sorted(CALLER_CLASSES)}")
    token = secrets.token_urlsafe(_TOKEN_BYTES)
    _TOKENS_DIR.mkdir(parents=True, exist_ok=True)
    path = _token_path(caller_class)
    path.write_text(token, encoding="utf-8")
    try:
        path.chmod(0o600)   # owner read/write only — same convention as calendar_auth.py/github_ci_auth.py
    except Exception:
        pass
    return token


def run_local_mint_flow(caller_class: str) -> None:
    """One-time interactive setup — mints a token for `caller_class` and
    prints it exactly once. Run manually
    (`python -m core.hermes_auth mint <caller_class>`); never called
    automatically."""
    token = mint_token(caller_class)
    print(f"Minted a new Hermes token for caller_class={caller_class!r}.")
    print(f"Token (copy this now — it will not be shown again): {token}")
    print(f"Stored locally at {_token_path(caller_class)}.")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "mint":
        print("Usage: python -m core.hermes_auth mint <caller_class>")
        print(f"  caller_class must be one of: {sorted(CALLER_CLASSES)}")
        sys.exit(1)
    run_local_mint_flow(sys.argv[2])
