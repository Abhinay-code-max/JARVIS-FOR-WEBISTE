"""
core/github_ci_auth.py
========================
GitHub personal-access-token storage for core/proactive.py's
ci_run_completed trigger. Read-only — Actions:read permission only.
Nothing in this module, or anything that calls it, ever writes to GitHub
(no run re-triggering, no repo writes, no other API calls at all).

Simpler setup than core/calendar_auth.py's OAuth loopback flow: GitHub
personal access tokens don't involve a browser consent round-trip or a
refresh cycle — you create one on github.com, paste it in once, and (for
a personal-account-owned repo) it never expires.

One-time external setup (do this before anything here will work; also
see config's github_ci_enabled flag — this trigger needs no new package,
just `requests`, already a core dependency):

  1. https://github.com/settings/personal-access-tokens/new — create a
     new **fine-grained** personal access token.
       - Resource owner: your personal account. Both repos
         core/proactive.py's CI_REPOS watches must be personal-account-
         owned (not org-owned) for step 2 below to be available — an
         org-owned repo forces a max 366-day expiration instead, which
         this module doesn't rotate automatically.
       - Expiration: **No expiration**.
       - Repository access: "Only select repositories" -> pick exactly
         the repos in core/proactive.py's CI_REPOS. Don't grant "All
         repositories" — this trigger has no use for anything beyond
         those two, and a broader grant is a bigger blast radius for no
         benefit.
       - Permissions: Repository permissions -> Actions -> **Read-only**.
         Nothing else — listing workflow runs needs no other scope.
  2. Run `python -m core.github_ci_auth` once from the repo root. It
     prompts for the token (paste it; input is not echoed to the
     terminal) and saves it to TOKEN_PATH (~/.jarvis/github_pat.txt),
     matching core/calendar_auth.py's ~/.jarvis/ storage convention.
     Nothing else to do after that — get_github_token() just reads the
     stored file; there's no refresh step because a non-expiring
     fine-grained PAT never needs one. If the token is ever revoked or
     regenerated, re-run this step with the new value.
"""
from __future__ import annotations

import getpass
import logging
from pathlib import Path

_log = logging.getLogger("jarvis.github_ci_auth")

_JARVIS_DIR = Path.home() / ".jarvis"
TOKEN_PATH  = _JARVIS_DIR / "github_pat.txt"


def get_github_token() -> str | None:
    """Returns the stored fine-grained PAT, or None if setup hasn't been
    completed yet (see module docstring). Never prompts — that's
    run_local_auth_flow() only, meant to be run explicitly and
    interactively, never from inside core/proactive.py's poll loop."""
    if not TOKEN_PATH.exists():
        return None
    try:
        token = TOKEN_PATH.read_text(encoding="utf-8").strip()
        return token or None
    except Exception:
        _log.warning("Could not read stored GitHub token at %s", TOKEN_PATH, exc_info=True)
        return None


def run_local_auth_flow() -> None:
    """One-time interactive setup — prompts for a pasted fine-grained PAT
    and caches it at TOKEN_PATH. Run manually
    (`python -m core.github_ci_auth`); never called automatically."""
    token = getpass.getpass("Paste your fine-grained GitHub PAT (input hidden): ").strip()
    if not token:
        raise ValueError("No token entered.")
    _JARVIS_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(token, encoding="utf-8")
    try:
        TOKEN_PATH.chmod(0o600)   # owner read/write only — same convention as calendar_auth.py
    except Exception:
        pass
    print(f"GitHub token saved to {TOKEN_PATH}.")


if __name__ == "__main__":
    run_local_auth_flow()
