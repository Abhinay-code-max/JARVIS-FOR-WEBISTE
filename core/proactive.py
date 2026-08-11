"""
core/proactive.py
==================
Phase-1 proactive nudges: JARVIS surfacing things unprompted, sourced
entirely from data it already generates about itself (approvals, tasks).
No external event source (calendar, messages, CI, ...) — that's a
separate, unbuilt future phase.

Three trigger types:
  - a background approval that's been pending too long (core/task_approval.py)
  - a background task that failed with an unhandled exception (agent/task_queue.py's
    _run_task except-block, which logs but never speaks — the one failure
    path in the app that today produces zero spoken feedback, ever, even
    when the task was submitted with a live speak callback)
  - a Google Calendar event starting soon (core/calendar_auth.py) — the
    one trigger sourced from outside JARVIS's own data, read-only,
    opt-in via config's calendar_enabled flag

Single daemon poll loop, not a job-scheduler library — this app has one
process, one user, and a handful of trigger types; a sleep loop is the
right size (see agent/task_queue.py / core/task_approval.py for the same
reasoning applied to their own machinery). The calendar trigger runs on
its own, slower cadence (CALENDAR_SYNC_INTERVAL_SEC) *inside* this same
loop rather than a second thread — see _sync_calendar()'s docstring for
why a live network call on every 60s tick would be a regression the
other two triggers don't have today.

Delivery reuses the existing TTS queue (JarvisXL.speak() / main.py) — the
same pathway the agent_task tool already uses to speak background-task
results unprompted. Nothing new there; what's new is deciding *when* to
call it, gated by:
  - not already speaking, not muted (JarvisXL's own live state)
  - quiet_hours (config/api_keys.json, optional; absent = disabled)
  - fire-once dedup per trigger (see proactive_nudges schema comment in
    core/db.py for why triggers need different dedup strategies)
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Callable

from config import load_config
from core.db import (
    get_cached_calendar_events,
    get_conn,
    get_step_outcomes,
    nudge_already_sent,
    record_nudge,
    replace_calendar_cache,
)
from core.task_approval import TASK_APPROVAL

_log = logging.getLogger("jarvis.proactive")

POLL_INTERVAL_SEC = 60

# 15 min: half of core/task_approval.py's DEFAULT_TIMEOUT (30 min), so a
# nudge always lands with real runway left before the approval
# auto-times-out — not so short that a user who stepped away for a few
# minutes gets nudged over an ordinary delay.
STALE_APPROVAL_SEC = 900

TRIGGER_TASK_FAILED = "task_failed"

# Calendar: sync (network) runs every 5 min; the nudge condition (checked
# every 60s tick, local cache only) fires once per event, 10 min before
# its start — 10 min is the exact lead time used as the canonical §16
# example. The sync window (30 min) is generous relative to both, so no
# event can arrive inside the 10-min lead-time boundary between one sync
# and the next without already being in the cache.
CALENDAR_SYNC_INTERVAL_SEC = 300
CALENDAR_LEAD_TIME_SEC     = 600
CALENDAR_SYNC_WINDOW_SEC   = 1800

TRIGGER_CALENDAR_EVENT = "calendar_event_upcoming"


def _parse_calendar_event(item: dict) -> dict | None:
    """Converts one raw Events.list() item into the flat dict
    replace_calendar_cache() expects. Returns None for an item with no
    usable start time (defensive — Google always sends one, but a
    malformed/partial item shouldn't crash the whole sync).

    all_day: Google represents an all-day event with start.date (a bare
    "YYYY-MM-DD") instead of start.dateTime — that's the one and only
    signal used here, matching the API's own representation exactly.

    start: computed from dateTime, which always carries an explicit
    UTC offset (Google never omits it), so .timestamp() on the parsed
    aware datetime yields a correct absolute epoch regardless of the
    calendar's configured timezone or this machine's local timezone. For
    an all-day event's bare date (no time/offset at all) an exact epoch
    is meaningless anyway — all_day events are always filtered out
    before the start value is ever used, so midnight-UTC is fine as a
    placeholder.

    rsvp_status: the signed-in user's own attendee entry
    (attendees[].self == true), or None if the event has no attendees
    list at all (e.g. a personal block the user created for themselves)
    — deliberately not treated as "declined" or filtered out."""
    from datetime import timezone as _tz

    start_info = item.get("start", {})
    all_day    = "date" in start_info and "dateTime" not in start_info

    if all_day:
        if "date" not in start_info:
            return None
        start_ts = datetime.fromisoformat(start_info["date"]).replace(tzinfo=_tz.utc).timestamp()
    else:
        if "dateTime" not in start_info:
            return None
        start_ts = datetime.fromisoformat(start_info["dateTime"]).timestamp()

    rsvp_status = None
    for attendee in item.get("attendees", []):
        if attendee.get("self"):
            rsvp_status = attendee.get("responseStatus")
            break

    return {
        "event_id":    item["id"],
        "summary":     item.get("summary", "(no title)"),
        "start":       start_ts,
        "all_day":     all_day,
        "status":      item.get("status", "confirmed"),
        "rsvp_status": rsvp_status,
    }


def _quiet_hours_active(cfg: dict, now: datetime | None = None) -> bool:
    """True if `now` (default: current time) falls inside the configured
    quiet_hours window. Config shape: {"quiet_hours": {"start": "HH:MM",
    "end": "HH:MM"}} in config/api_keys.json — absent or malformed means
    quiet hours are simply off, not an error."""
    qh = cfg.get("quiet_hours")
    if not isinstance(qh, dict) or not qh.get("start") or not qh.get("end"):
        return False
    try:
        start = datetime.strptime(qh["start"], "%H:%M").time()
        end   = datetime.strptime(qh["end"], "%H:%M").time()
    except (ValueError, TypeError):
        return False

    now_t = (now or datetime.now()).time()
    if start <= end:
        return start <= now_t < end
    return now_t >= start or now_t < end  # window wraps past midnight


class ProactiveLoop:
    def __init__(
        self,
        speak:       Callable[[str], None],
        is_speaking: Callable[[], bool],
        is_muted:    Callable[[], bool],
        now:         Callable[[], datetime] = datetime.now,
    ):
        self._speak       = speak
        self._is_speaking = is_speaking
        self._is_muted    = is_muted
        self._now         = now
        # In-memory only, deliberately — see core/db.py's proactive_nudges
        # schema comment. A restart wipes this set, but by then
        # _reconcile_orphaned_approvals() has already flipped every
        # leftover 'pending' row to 'expired', so nothing stale can
        # resurface through TASK_APPROVAL.list_pending() anyway.
        self._nudged_approvals: set[int] = set()
        # Epoch time of the last calendar sync, via self._now() so tests
        # can control cadence deterministically — 0.0 forces an immediate
        # first sync on the loop's first tick rather than waiting a full
        # CALENDAR_SYNC_INTERVAL_SEC after startup.
        self._last_calendar_sync: float = 0.0
        self._running = False

    def _gated_speak(self, text: str) -> bool:
        """Returns True if the nudge was actually spoken (so callers only
        record dedup state on a real delivery, not a suppressed one)."""
        if self._is_speaking() or self._is_muted():
            return False
        if _quiet_hours_active(load_config(), now=self._now()):
            return False
        self._speak(text)
        return True

    # -- trigger: stale pending approval ------------------------------------

    def _check_stale_approvals(self) -> None:
        now = time.time()
        for row in TASK_APPROVAL.list_pending():
            approval_id = row["approval_id"]
            if approval_id in self._nudged_approvals:
                continue
            age = now - row["requested_at"]
            if age < STALE_APPROVAL_SEC:
                continue
            minutes = int(age // 60)
            msg = (
                f"Sir, approval request #{approval_id} has been waiting "
                f"{minutes} minutes: {row['prompt'][:120]}"
            )
            if self._gated_speak(msg):
                self._nudged_approvals.add(approval_id)

    # -- trigger: background task failed -------------------------------------

    def _check_failed_tasks(self) -> None:
        conn = get_conn()
        rows = conn.execute(
            "SELECT task_id, goal, error FROM tasks WHERE status = 'failed'"
        ).fetchall()

        for row in rows:
            task_id = row["task_id"]
            if nudge_already_sent(TRIGGER_TASK_FAILED, task_id):
                continue

            outcomes  = get_step_outcomes(task_id)
            last_step = outcomes[-1] if outcomes else None
            step_ctx  = f" while on \"{last_step['description']}\"" if last_step else ""
            error     = (row["error"] or "no further detail").strip()

            msg = (
                f"Sir, the background task \"{row['goal'][:60]}\" failed"
                f"{step_ctx}: {error[:150]}"
            )
            # If _gated_speak() suppressed it (quiet hours / already
            # speaking / muted), leave it un-dedup'd so the next tick
            # retries — tasks.status stays 'failed' forever, so there's no
            # narrow window to miss the way there is for approvals.
            if self._gated_speak(msg):
                record_nudge(TRIGGER_TASK_FAILED, task_id)

    # -- trigger: calendar event starting soon -------------------------------

    def _sync_calendar(self) -> None:
        """Refreshes calendar_events_cache from the Google Calendar API.
        Called from _tick() at most once every CALENDAR_SYNC_INTERVAL_SEC
        — never on every 60s tick. A live API call on every tick would
        block the other two (purely local, near-instant) checks behind
        network latency or a hung connection, something neither of them
        risks today; decoupling the cadences inside this same loop avoids
        that without needing a second thread.

        On any failure (calendar packages not installed, not yet
        authorized, network error, revoked token, rate limit) this logs
        and returns — _check_calendar_events() then just keeps reading
        whatever was cached by the last successful sync. A calendar
        outage must never take down the other two triggers or the loop
        itself."""
        if not load_config().get("calendar_enabled"):
            return

        try:
            from core import calendar_auth
        except ImportError:
            _log.warning(
                "calendar_enabled is set but the google-* packages aren't "
                "installed — see core/calendar_auth.py for setup."
            )
            return

        try:
            service = calendar_auth.get_calendar_service()
        except Exception:
            _log.warning("Calendar auth failed — skipping this sync cycle", exc_info=True)
            return

        if service is None:
            _log.debug(
                "Calendar not authorized yet — run `python -m core.calendar_auth` "
                "(see that module's docstring)."
            )
            return

        from datetime import timedelta, timezone

        # .astimezone(timezone.utc) correctly converts both a naive
        # datetime (Python treats it as system local time since 3.6) and
        # an already-aware one — no separate branch needed.
        now_utc  = self._now().astimezone(timezone.utc)
        time_min = now_utc.isoformat()
        time_max = (now_utc + timedelta(seconds=CALENDAR_SYNC_WINDOW_SEC)).isoformat()

        try:
            result = service.events().list(
                calendarId  = "primary",
                timeMin     = time_min,
                timeMax     = time_max,
                singleEvents = True,
                orderBy     = "startTime",
            ).execute()
        except Exception:
            _log.warning("Calendar sync failed — serving last-known cache", exc_info=True)
            return

        events = [
            parsed for parsed in (
                _parse_calendar_event(item) for item in result.get("items", [])
            ) if parsed is not None
        ]
        replace_calendar_cache(events)

    def _check_calendar_events(self) -> None:
        now = self._now().timestamp()
        for row in get_cached_calendar_events():
            if row["all_day"] or row["status"] == "cancelled" or row["rsvp_status"] == "declined":
                continue

            remaining = row["start"] - now
            if not (0 < remaining <= CALENDAR_LEAD_TIME_SEC):
                continue

            event_id = row["event_id"]
            if nudge_already_sent(TRIGGER_CALENDAR_EVENT, event_id):
                continue

            minutes = max(1, round(remaining / 60))
            unit    = "minute" if minutes == 1 else "minutes"
            msg     = f"You have '{row['summary']}' starting in {minutes} {unit}, sir."

            if self._gated_speak(msg):
                record_nudge(TRIGGER_CALENDAR_EVENT, event_id)

    def _tick(self) -> None:
        try:
            self._check_stale_approvals()
        except Exception:
            _log.warning("Stale-approval check failed", exc_info=True)
        try:
            self._check_failed_tasks()
        except Exception:
            _log.warning("Failed-task check failed", exc_info=True)

        now = self._now().timestamp()
        if now - self._last_calendar_sync >= CALENDAR_SYNC_INTERVAL_SEC:
            self._last_calendar_sync = now
            try:
                self._sync_calendar()
            except Exception:
                _log.warning("Calendar sync failed", exc_info=True)
        try:
            self._check_calendar_events()
        except Exception:
            _log.warning("Calendar event check failed", exc_info=True)

    def run(self) -> None:
        # Forces TaskQueue.start() (and its startup reconciliation pass —
        # see agent/task_queue.py) if nothing has submitted a background
        # task yet this run. Idempotent: a no-op if the queue is already
        # started.
        from agent.task_queue import get_queue
        get_queue()

        self._running = True
        while self._running:
            time.sleep(POLL_INTERVAL_SEC)
            self._tick()

    def stop(self) -> None:
        self._running = False
