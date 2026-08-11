"""
tests/test_proactive.py
========================
Phase-1 proactive nudges (core/proactive.py): the four trigger types, the
speaking/muted/quiet-hours gate, and the dedup story for each trigger
(durable table for failed tasks, calendar events, and CI runs;
in-memory-only for stale approvals — see core/db.py's proactive_nudges
schema comment for why they differ).

The calendar trigger's tests never import the real google-* packages
(not installed in this environment, matching most real installs that
haven't opted into calendar_enabled) — core.calendar_auth is stood in
for via sys.modules patching, since core/proactive.py's `from core import
calendar_auth` is a lazy, function-local import specifically so a
missing/failing calendar_auth can never break the other triggers. The CI
trigger's tests use the same sys.modules-patching approach for
core.github_ci_auth (also a lazy, function-local import), but `requests`
itself IS a real installed dependency here, so its tests patch
`requests.get` directly rather than needing a whole fake module.

Redirects core.db at a fresh temp sqlite file, same pattern as the other
test modules in this package, so nothing here touches data/jarvis.db.
"""
from __future__ import annotations

import sys
import tempfile
import threading
import time
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.db as db


def _use_temp_db() -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="jarvis_test_db_"))
    db.DB_DIR  = tmp_dir
    db.DB_PATH = tmp_dir / "test.db"
    db._local  = threading.local()
    return db.DB_PATH


_use_temp_db()

import core.proactive as proactive          # noqa: E402
from core.task_approval import TASK_APPROVAL, _PendingApproval  # noqa: E402


def _insert_task(task_id: str, status: str, error: str = "", goal: str = "test goal"):
    conn = db.get_conn()
    with conn:
        conn.execute(
            "INSERT INTO tasks (task_id, goal, priority, status, result, error, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (task_id, goal, 2, status, None, error, time.time(), time.time()),
        )


def _insert_pending_approval(approval_id: int, prompt: str, requested_at: float, task_id: str = "t1"):
    """Mirrors what TASK_APPROVAL.wait_for_approval() does, but with a
    caller-chosen approval_id/requested_at so tests can backdate it past
    the staleness threshold without a real 15-minute sleep."""
    conn = db.get_conn()
    with conn:
        conn.execute(
            "INSERT INTO approvals (approval_id, prompt, requested_at, answered_at, outcome, task_id) "
            "VALUES (?, ?, ?, NULL, 'pending', ?)",
            (approval_id, prompt, requested_at, task_id),
        )
    TASK_APPROVAL._pending[approval_id] = _PendingApproval()


def _insert_cached_event(
    event_id: str, start_ts: float, summary: str = "Standup",
    all_day: bool = False, status: str = "confirmed", rsvp_status: str | None = None,
):
    """Writes directly into calendar_events_cache — bypasses
    _sync_calendar() entirely, so trigger-logic tests (filters, lead
    time, dedup) don't need a fake Google API client at all."""
    conn = db.get_conn()
    with conn:
        conn.execute(
            "INSERT INTO calendar_events_cache "
            "(event_id, summary, start, all_day, status, rsvp_status, synced_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (event_id, summary, start_ts, 1 if all_day else 0, status, rsvp_status, time.time()),
        )


def _google_event(
    event_id: str, start_dt: datetime, summary: str = "Standup",
    status: str = "confirmed", all_day: bool = False, rsvp_status: str | None = None,
) -> dict:
    """One raw Events.list() item, in Google's own shape — for
    _sync_calendar()/_parse_calendar_event() tests only."""
    item: dict = {"id": event_id, "summary": summary, "status": status}
    item["start"] = (
        {"date": start_dt.strftime("%Y-%m-%d")} if all_day
        else {"dateTime": start_dt.isoformat()}
    )
    if rsvp_status is not None:
        item["attendees"] = [{"self": True, "responseStatus": rsvp_status}]
    return item


class _FakeEventsCall:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error    = error

    def execute(self):
        if self._error:
            raise self._error
        return self._response


class _FakeEvents:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error    = error

    def list(self, **kwargs):
        return _FakeEventsCall(self._response, self._error)


class _FakeService:
    """Mimics googleapiclient's service.events().list().execute() chain
    just enough for _sync_calendar()'s tests."""

    def __init__(self, response=None, error=None):
        self._response = response
        self._error    = error

    def events(self):
        return _FakeEvents(self._response, self._error)


def _fake_calendar_auth_module(service) -> types.ModuleType:
    """A stand-in for core.calendar_auth, registered into sys.modules so
    core/proactive.py's lazy `from core import calendar_auth` picks it up
    instead of the real (google-* dependent) module."""
    fake = types.ModuleType("core.calendar_auth")
    fake.get_calendar_service = lambda: service
    return fake


def _insert_cached_ci_run(
    run_id: int, repo: str | None = None, workflow: str = "CI",
    title: str = "Fix login bug", branch: str = "main", sha: str = "abc123",
    status: str = "completed", conclusion: str | None = "success",
):
    """Writes directly into ci_runs_cache — bypasses _sync_ci() entirely,
    so trigger-logic tests (filters, dedup, grouping) don't need a fake
    GitHub API response at all."""
    conn = db.get_conn()
    with conn:
        conn.execute(
            "INSERT INTO ci_runs_cache "
            "(run_id, repo, workflow, title, branch, sha, status, conclusion, html_url, synced_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id, repo or proactive.CI_REPOS[0], workflow, title, branch, sha,
                status, conclusion, f"https://github.com/{repo or proactive.CI_REPOS[0]}/actions/runs/{run_id}",
                time.time(),
            ),
        )


def _github_run(
    run_id: int, workflow: str = "CI", title: str = "Fix login bug",
    branch: str = "main", sha: str = "abc123",
    status: str = "completed", conclusion: str | None = "success",
) -> dict:
    """One raw 'list workflow runs' item, in GitHub's own response shape
    — for _sync_ci()/_parse_ci_run() tests only."""
    return {
        "id": run_id, "name": workflow, "display_title": title,
        "head_branch": branch, "head_sha": sha,
        "status": status, "conclusion": conclusion,
        "html_url": f"https://github.com/x/y/actions/runs/{run_id}",
    }


class _FakeGithubResponse:
    """Mimics requests.Response just enough for _sync_ci()'s tests —
    .raise_for_status()/.json(), no real HTTP involved."""

    def __init__(self, json_data=None, error=None):
        self._json_data = json_data or {}
        self._error     = error

    def raise_for_status(self):
        if self._error:
            raise self._error

    def json(self):
        return self._json_data


def _fake_github_ci_auth_module(token) -> types.ModuleType:
    """A stand-in for core.github_ci_auth, registered into sys.modules so
    core/proactive.py's lazy `from core import github_ci_auth` picks it
    up instead of the real module."""
    fake = types.ModuleType("core.github_ci_auth")
    fake.get_github_token = lambda: token
    return fake


class _RecordingLoop(proactive.ProactiveLoop):
    """Same gating logic as the real loop, but records what actually got
    spoken instead of needing a live JarvisXL/TTS stack."""

    def __init__(self, is_speaking=lambda: False, is_muted=lambda: False,
                 now=datetime.now):
        self.spoken: list[str] = []
        super().__init__(speak=self.spoken.append, is_speaking=is_speaking,
                          is_muted=is_muted, now=now)


class ProactiveTestCase(unittest.TestCase):
    def setUp(self):
        _use_temp_db()
        TASK_APPROVAL._pending.clear()


class StaleApprovalTriggerTest(ProactiveTestCase):
    def test_not_yet_stale_does_not_nudge(self):
        _insert_pending_approval(1, "delete these files", time.time() - 60)
        loop = _RecordingLoop()
        loop._check_stale_approvals()
        self.assertEqual(loop.spoken, [])

    def test_past_threshold_nudges_once(self):
        _insert_pending_approval(
            1, "delete these files",
            time.time() - proactive.STALE_APPROVAL_SEC - 1,
        )
        loop = _RecordingLoop()
        loop._check_stale_approvals()
        self.assertEqual(len(loop.spoken), 1)
        self.assertIn("delete these files", loop.spoken[0])

    def test_second_tick_does_not_renudge_the_same_approval(self):
        _insert_pending_approval(
            1, "delete these files",
            time.time() - proactive.STALE_APPROVAL_SEC - 1,
        )
        loop = _RecordingLoop()
        loop._check_stale_approvals()
        loop._check_stale_approvals()
        self.assertEqual(len(loop.spoken), 1)

    def test_resolved_approval_stops_matching_without_any_dedup_state(self):
        """The natural-state-exit claim from the plan: once a human
        resolves an approval via the real wait_for_approval()/answer()
        pair (the waiting thread pops itself from the in-memory pending
        registry on wake, same as production), list_pending() stops
        returning it — no durable table needed for this trigger."""
        def _wait():
            TASK_APPROVAL.wait_for_approval("t1", "delete these files", timeout=5)

        t = threading.Thread(target=_wait, daemon=True)
        t.start()
        deadline = time.time() + 2
        while time.time() < deadline and not TASK_APPROVAL.list_pending():
            time.sleep(0.01)
        pending = TASK_APPROVAL.list_pending()
        self.assertEqual(len(pending), 1)

        TASK_APPROVAL.answer(pending[0]["approval_id"], approve=True)
        t.join(timeout=2)

        loop = _RecordingLoop()
        loop._check_stale_approvals()
        self.assertEqual(loop.spoken, [])


class FailedTaskTriggerTest(ProactiveTestCase):
    def test_failed_task_nudges_with_error_detail(self):
        _insert_task("t1", "failed", error="network timeout", goal="book a flight")
        loop = _RecordingLoop()
        loop._check_failed_tasks()
        self.assertEqual(len(loop.spoken), 1)
        self.assertIn("book a flight", loop.spoken[0])
        self.assertIn("network timeout", loop.spoken[0])

    def test_completed_task_does_not_nudge(self):
        _insert_task("t1", "completed")
        loop = _RecordingLoop()
        loop._check_failed_tasks()
        self.assertEqual(loop.spoken, [])

    def test_restart_does_not_resurface_a_failed_task_already_nudged(self):
        """The fix the plan required: dedup for this trigger must survive
        a process restart (a fresh ProactiveLoop / in-memory state), since
        a failed task row has no natural state-exit like approvals do."""
        _insert_task("t1", "failed", error="network timeout", goal="book a flight")
        first_run = _RecordingLoop()
        first_run._check_failed_tasks()
        self.assertEqual(len(first_run.spoken), 1)

        # Simulate a restart: brand-new loop instance, empty in-memory
        # state, same on-disk db.
        second_run = _RecordingLoop()
        second_run._check_failed_tasks()
        self.assertEqual(second_run.spoken, [])

    def test_suppressed_nudge_is_not_dedup_recorded_and_retries_next_tick(self):
        _insert_task("t1", "failed", error="network timeout", goal="book a flight")
        loop = _RecordingLoop(is_muted=lambda: True)
        loop._check_failed_tasks()
        self.assertEqual(loop.spoken, [])
        self.assertFalse(db.nudge_already_sent(proactive.TRIGGER_TASK_FAILED, "t1"))


class GatingTest(ProactiveTestCase):
    def test_speaking_suppresses_nudge(self):
        _insert_task("t1", "failed", error="x", goal="y")
        loop = _RecordingLoop(is_speaking=lambda: True)
        loop._check_failed_tasks()
        self.assertEqual(loop.spoken, [])

    def test_muted_suppresses_nudge(self):
        _insert_task("t1", "failed", error="x", goal="y")
        loop = _RecordingLoop(is_muted=lambda: True)
        loop._check_failed_tasks()
        self.assertEqual(loop.spoken, [])

    def test_quiet_hours_suppresses_a_nudge_that_would_otherwise_fire(self):
        """Concrete proof the config-level quiet_hours gate actually
        suppresses delivery, not just that the key exists unused."""
        cfg = {"quiet_hours": {"start": "22:00", "end": "07:00"}}
        original_load_config = proactive.load_config
        proactive.load_config = lambda: cfg
        try:
            _insert_task("t1", "failed", error="x", goal="y")
            during_quiet_hours = datetime(2026, 8, 10, 23, 0)
            loop = _RecordingLoop(now=lambda: during_quiet_hours)
            self.assertTrue(proactive._quiet_hours_active(cfg, now=during_quiet_hours))
            loop._check_failed_tasks()
            self.assertEqual(loop.spoken, [])
        finally:
            proactive.load_config = original_load_config

    def test_outside_quiet_hours_window_nudge_still_fires(self):
        cfg = {"quiet_hours": {"start": "22:00", "end": "07:00"}}
        original_load_config = proactive.load_config
        proactive.load_config = lambda: cfg
        try:
            _insert_task("t1", "failed", error="x", goal="y")
            midday = datetime(2026, 8, 10, 12, 0)
            loop = _RecordingLoop(now=lambda: midday)
            self.assertFalse(proactive._quiet_hours_active(cfg, now=midday))
            loop._check_failed_tasks()
            self.assertEqual(len(loop.spoken), 1)
        finally:
            proactive.load_config = original_load_config

    def test_no_quiet_hours_configured_never_suppresses(self):
        self.assertFalse(proactive._quiet_hours_active({}))
        self.assertFalse(proactive._quiet_hours_active({"quiet_hours": {}}))
        self.assertFalse(proactive._quiet_hours_active({"quiet_hours": {"start": "bad"}}))


class ParseCalendarEventTest(unittest.TestCase):
    def test_timed_event_parses_absolute_start_and_is_not_all_day(self):
        start  = datetime.now(timezone.utc) + timedelta(minutes=5)
        parsed = proactive._parse_calendar_event(_google_event("e1", start))
        self.assertIsNotNone(parsed)
        self.assertFalse(parsed["all_day"])
        self.assertAlmostEqual(parsed["start"], start.timestamp(), delta=1)

    def test_all_day_event_is_flagged(self):
        start  = datetime.now(timezone.utc)
        parsed = proactive._parse_calendar_event(_google_event("e2", start, all_day=True))
        self.assertTrue(parsed["all_day"])

    def test_self_attendee_rsvp_is_extracted(self):
        start  = datetime.now(timezone.utc) + timedelta(minutes=5)
        parsed = proactive._parse_calendar_event(_google_event("e3", start, rsvp_status="declined"))
        self.assertEqual(parsed["rsvp_status"], "declined")

    def test_no_attendees_means_no_rsvp_status(self):
        start  = datetime.now(timezone.utc) + timedelta(minutes=5)
        parsed = proactive._parse_calendar_event(_google_event("e4", start))
        self.assertIsNone(parsed["rsvp_status"])

    def test_malformed_item_with_no_start_returns_none(self):
        item = {"id": "e5", "summary": "broken", "status": "confirmed", "start": {}}
        self.assertIsNone(proactive._parse_calendar_event(item))


class SyncCalendarTest(ProactiveTestCase):
    def test_disabled_by_default_never_touches_calendar_auth(self):
        """calendar_enabled defaults to unset/False — _sync_calendar()
        must return before even importing core.calendar_auth. Forces
        load_config() explicitly rather than trusting the ambient real
        config (which could have calendar_enabled set for real once the
        feature is actually in use — see the CI trigger's equivalent test
        for a concrete case where that assumption broke). Proven by NOT
        registering any fake module: if it tried the real import, this
        would raise (google-* packages aren't installed here)."""
        with patch.object(proactive, "load_config", lambda: {}):
            loop = _RecordingLoop()
            loop._sync_calendar()
        self.assertEqual(db.get_cached_calendar_events(), [])

    def test_enabled_populates_cache_from_api_response(self):
        start    = datetime.now(timezone.utc) + timedelta(minutes=5)
        response = {"items": [_google_event("e1", start, summary="Standup")]}
        fake_auth = _fake_calendar_auth_module(_FakeService(response=response))
        cfg       = {"calendar_enabled": True}

        with patch.dict(sys.modules, {"core.calendar_auth": fake_auth}), \
             patch.object(proactive, "load_config", lambda: cfg):
            _RecordingLoop()._sync_calendar()

        cached = db.get_cached_calendar_events()
        self.assertEqual(len(cached), 1)
        self.assertEqual(cached[0]["event_id"], "e1")
        self.assertEqual(cached[0]["summary"], "Standup")

    def test_not_yet_authorized_service_is_none_leaves_cache_empty(self):
        cfg       = {"calendar_enabled": True}
        fake_auth = _fake_calendar_auth_module(None)

        with patch.dict(sys.modules, {"core.calendar_auth": fake_auth}), \
             patch.object(proactive, "load_config", lambda: cfg):
            _RecordingLoop()._sync_calendar()  # must not raise

        self.assertEqual(db.get_cached_calendar_events(), [])

    def test_api_failure_keeps_serving_last_known_cache(self):
        """The resilience guarantee: a broken sync cycle must not wipe or
        corrupt whatever the previous successful sync already cached."""
        cfg   = {"calendar_enabled": True}
        start = datetime.now(timezone.utc) + timedelta(minutes=5)
        good  = {"items": [_google_event("e1", start)]}

        with patch.object(proactive, "load_config", lambda: cfg):
            with patch.dict(sys.modules, {"core.calendar_auth": _fake_calendar_auth_module(_FakeService(response=good))}):
                _RecordingLoop()._sync_calendar()
            self.assertEqual(len(db.get_cached_calendar_events()), 1)

            failing = _fake_calendar_auth_module(_FakeService(error=ConnectionError("network down")))
            with patch.dict(sys.modules, {"core.calendar_auth": failing}):
                _RecordingLoop()._sync_calendar()  # must not raise

        cached = db.get_cached_calendar_events()
        self.assertEqual(len(cached), 1)
        self.assertEqual(cached[0]["event_id"], "e1")

    def test_get_calendar_service_raising_does_not_crash_sync(self):
        cfg  = {"calendar_enabled": True}
        fake = types.ModuleType("core.calendar_auth")

        def _raise():
            raise RuntimeError("token file corrupted")

        fake.get_calendar_service = _raise

        with patch.dict(sys.modules, {"core.calendar_auth": fake}), \
             patch.object(proactive, "load_config", lambda: cfg):
            _RecordingLoop()._sync_calendar()  # must not raise

        self.assertEqual(db.get_cached_calendar_events(), [])

    def test_tick_survives_an_unexpected_exception_inside_sync_calendar(self):
        """Defense in depth beyond _sync_calendar()'s own try/excepts:
        even if it somehow raised anyway, _tick()'s own wrapper must
        keep the other two triggers running unaffected."""
        cfg = {"calendar_enabled": True}
        _insert_task("t1", "failed", error="x", goal="y")

        class ExplodingLoop(_RecordingLoop):
            def _sync_calendar(self):
                raise RuntimeError("boom")

        with patch.object(proactive, "load_config", lambda: cfg):
            loop = ExplodingLoop()
            loop._tick()  # must not raise

        self.assertEqual(len(loop.spoken), 1)
        self.assertIn("y", loop.spoken[0])


class CalendarTriggerTest(ProactiveTestCase):
    def test_event_within_lead_time_nudges_with_summary_and_minutes(self):
        _insert_cached_event("e1", time.time() + 300, summary="Team Standup")
        loop = _RecordingLoop()
        loop._check_calendar_events()
        self.assertEqual(len(loop.spoken), 1)
        self.assertIn("Team Standup", loop.spoken[0])
        self.assertIn("5 minute", loop.spoken[0])

    def test_event_outside_lead_time_does_not_nudge(self):
        _insert_cached_event("e1", time.time() + proactive.CALENDAR_LEAD_TIME_SEC + 60)
        loop = _RecordingLoop()
        loop._check_calendar_events()
        self.assertEqual(loop.spoken, [])

    def test_event_already_started_does_not_nudge(self):
        _insert_cached_event("e1", time.time() - 30)
        loop = _RecordingLoop()
        loop._check_calendar_events()
        self.assertEqual(loop.spoken, [])

    def test_all_day_event_is_skipped(self):
        _insert_cached_event("e1", time.time() + 300, all_day=True)
        loop = _RecordingLoop()
        loop._check_calendar_events()
        self.assertEqual(loop.spoken, [])

    def test_cancelled_event_is_skipped(self):
        _insert_cached_event("e1", time.time() + 300, status="cancelled")
        loop = _RecordingLoop()
        loop._check_calendar_events()
        self.assertEqual(loop.spoken, [])

    def test_declined_rsvp_is_skipped(self):
        _insert_cached_event("e1", time.time() + 300, rsvp_status="declined")
        loop = _RecordingLoop()
        loop._check_calendar_events()
        self.assertEqual(loop.spoken, [])

    def test_tentative_rsvp_still_nudges(self):
        _insert_cached_event("e1", time.time() + 300, rsvp_status="tentative")
        loop = _RecordingLoop()
        loop._check_calendar_events()
        self.assertEqual(len(loop.spoken), 1)

    def test_needs_action_rsvp_still_nudges(self):
        _insert_cached_event("e1", time.time() + 300, rsvp_status="needsAction")
        loop = _RecordingLoop()
        loop._check_calendar_events()
        self.assertEqual(len(loop.spoken), 1)

    def test_no_attendees_personal_block_still_nudges(self):
        _insert_cached_event("e1", time.time() + 300, rsvp_status=None)
        loop = _RecordingLoop()
        loop._check_calendar_events()
        self.assertEqual(len(loop.spoken), 1)

    def test_second_check_does_not_renudge_the_same_event(self):
        _insert_cached_event("e1", time.time() + 300)
        loop = _RecordingLoop()
        loop._check_calendar_events()
        loop._check_calendar_events()
        self.assertEqual(len(loop.spoken), 1)

    def test_dedup_is_durable_across_a_simulated_restart(self):
        _insert_cached_event("e1", time.time() + 300)
        first_run = _RecordingLoop()
        first_run._check_calendar_events()
        self.assertEqual(len(first_run.spoken), 1)

        second_run = _RecordingLoop()   # fresh in-memory state, same db
        second_run._check_calendar_events()
        self.assertEqual(second_run.spoken, [])

    def test_recurring_event_instances_dedup_independently(self):
        """Each occurrence's Google-assigned id is already globally
        unique (confirmed in the plan) — no recurrence-aware logic
        needed here, just proof two occurrences of 'the same meeting'
        nudge independently rather than colliding on one dedup row."""
        now = time.time()
        _insert_cached_event("instance_20260810T090000Z", now + 300, summary="Daily Sync")
        _insert_cached_event("instance_20260811T090000Z", now + 300, summary="Daily Sync")
        loop = _RecordingLoop()
        loop._check_calendar_events()
        self.assertEqual(len(loop.spoken), 2)

    def test_gating_suppresses_calendar_nudge_and_leaves_it_undedup(self):
        _insert_cached_event("e1", time.time() + 300)
        loop = _RecordingLoop(is_muted=lambda: True)
        loop._check_calendar_events()
        self.assertEqual(loop.spoken, [])
        self.assertFalse(db.nudge_already_sent(proactive.TRIGGER_CALENDAR_EVENT, "e1"))


class ParseCiRunTest(unittest.TestCase):
    def test_parses_all_expected_fields(self):
        item   = _github_run(101, workflow="Build", title="Fix flaky test", branch="main", sha="deadbeef")
        parsed = proactive._parse_ci_run(item)
        self.assertEqual(parsed["run_id"], 101)
        self.assertEqual(parsed["workflow"], "Build")
        self.assertEqual(parsed["title"], "Fix flaky test")
        self.assertEqual(parsed["branch"], "main")
        self.assertEqual(parsed["sha"], "deadbeef")
        self.assertEqual(parsed["status"], "completed")
        self.assertEqual(parsed["conclusion"], "success")

    def test_missing_name_falls_back_to_placeholder(self):
        item = _github_run(102)
        del item["name"]
        parsed = proactive._parse_ci_run(item)
        self.assertEqual(parsed["workflow"], "(unnamed workflow)")

    def test_in_progress_run_has_null_conclusion(self):
        item   = _github_run(103, status="in_progress", conclusion=None)
        parsed = proactive._parse_ci_run(item)
        self.assertEqual(parsed["status"], "in_progress")
        self.assertIsNone(parsed["conclusion"])


class SyncCiTest(ProactiveTestCase):
    def test_disabled_by_default_never_touches_requests_or_github_ci_auth(self):
        """github_ci_enabled defaults to unset/False — _sync_ci() must
        return before even calling requests.get. Forces load_config()
        explicitly rather than trusting the ambient real config: this
        machine now has github_ci_enabled genuinely set (the feature is
        live), so without this the test would silently exercise the real
        network path with the real stored PAT — a real bug this exact
        test caught, see git history. Call-counted rather than relying on
        an exception surviving _sync_ci()'s own broad except-Exception
        (which would otherwise swallow a raised AssertionError here
        without this test ever noticing)."""
        calls = {"n": 0}

        def _counting_get(*a, **kw):
            calls["n"] += 1
            raise AssertionError("must not be called")

        with patch.object(proactive, "load_config", lambda: {}), \
             patch("requests.get", side_effect=_counting_get):
            loop = _RecordingLoop()
            loop._sync_ci()

        self.assertEqual(calls["n"], 0)
        self.assertEqual(db.get_cached_ci_runs(), [])

    def test_enabled_populates_cache_from_api_response_for_both_repos(self):
        cfg       = {"github_ci_enabled": True}
        fake_auth = _fake_github_ci_auth_module("fake-token")
        repo_a, repo_b = proactive.CI_REPOS

        def _fake_get(url, headers=None, params=None, timeout=None):
            if repo_a in url:
                return _FakeGithubResponse({"workflow_runs": [_github_run(1)]})
            if repo_b in url:
                return _FakeGithubResponse({"workflow_runs": [_github_run(2)]})
            raise AssertionError(f"unexpected URL: {url}")

        with patch("core.github_ci_auth", fake_auth, create=True), \
             patch.object(proactive, "load_config", lambda: cfg), \
             patch("requests.get", side_effect=_fake_get):
            _RecordingLoop()._sync_ci()

        cached = {r["run_id"]: r["repo"] for r in db.get_cached_ci_runs()}
        self.assertEqual(cached, {1: repo_a, 2: repo_b})

    def test_request_headers_carry_bearer_token_and_api_version(self):
        cfg       = {"github_ci_enabled": True}
        fake_auth = _fake_github_ci_auth_module("secret-token-123")
        seen_headers = []

        def _fake_get(url, headers=None, params=None, timeout=None):
            seen_headers.append(headers)
            return _FakeGithubResponse({"workflow_runs": []})

        with patch("core.github_ci_auth", fake_auth, create=True), \
             patch.object(proactive, "load_config", lambda: cfg), \
             patch("requests.get", side_effect=_fake_get):
            _RecordingLoop()._sync_ci()

        self.assertEqual(len(seen_headers), 2)  # one call per watched repo
        for headers in seen_headers:
            self.assertEqual(headers["Authorization"], "Bearer secret-token-123")
            self.assertEqual(headers["X-GitHub-Api-Version"], proactive._GITHUB_API_VERSION)

    def test_not_yet_authorized_leaves_cache_empty(self):
        cfg       = {"github_ci_enabled": True}
        fake_auth = _fake_github_ci_auth_module(None)

        with patch("core.github_ci_auth", fake_auth, create=True), \
             patch.object(proactive, "load_config", lambda: cfg), \
             patch("requests.get", side_effect=AssertionError("must not be called")):
            _RecordingLoop()._sync_ci()  # must not raise

        self.assertEqual(db.get_cached_ci_runs(), [])

    def test_get_github_token_raising_does_not_crash_sync(self):
        cfg  = {"github_ci_enabled": True}
        fake = types.ModuleType("core.github_ci_auth")

        def _raise():
            raise RuntimeError("token file corrupted")

        fake.get_github_token = _raise

        with patch("core.github_ci_auth", fake, create=True), \
             patch.object(proactive, "load_config", lambda: cfg):
            _RecordingLoop()._sync_ci()  # must not raise

        self.assertEqual(db.get_cached_ci_runs(), [])

    def test_one_repo_failing_does_not_wipe_the_other_repos_cache(self):
        """The resilience guarantee, specific to CI's per-repo scoping:
        repo A's sync failing must not touch repo B's already-cached
        rows — unlike calendar's single-source full-table replace, this
        table holds two independent sources, and replace_ci_cache() only
        ever deletes the one repo it's given."""
        cfg       = {"github_ci_enabled": True}
        fake_auth = _fake_github_ci_auth_module("fake-token")
        repo_a, repo_b = proactive.CI_REPOS

        def _good_get(url, headers=None, params=None, timeout=None):
            if repo_a in url:
                return _FakeGithubResponse({"workflow_runs": [_github_run(1)]})
            return _FakeGithubResponse({"workflow_runs": [_github_run(2)]})

        with patch("core.github_ci_auth", fake_auth, create=True), \
             patch.object(proactive, "load_config", lambda: cfg), \
             patch("requests.get", side_effect=_good_get):
            _RecordingLoop()._sync_ci()

        cached = {r["run_id"]: r["repo"] for r in db.get_cached_ci_runs()}
        self.assertEqual(cached, {1: repo_a, 2: repo_b})

        def _mixed_get(url, headers=None, params=None, timeout=None):
            if repo_a in url:
                raise ConnectionError("network down")
            return _FakeGithubResponse({"workflow_runs": [_github_run(3)]})

        with patch("core.github_ci_auth", fake_auth, create=True), \
             patch.object(proactive, "load_config", lambda: cfg), \
             patch("requests.get", side_effect=_mixed_get):
            _RecordingLoop()._sync_ci()  # must not raise

        cached = {r["run_id"]: r["repo"] for r in db.get_cached_ci_runs()}
        # repo_a's row from the first (successful) sync must survive
        # untouched — this cycle's failure for repo_a didn't delete it.
        self.assertEqual(cached.get(1), repo_a)
        # repo_b's cache DID get replaced (run 2 -> run 3), proving
        # repo_a's failure didn't somehow also block repo_b's own sync.
        self.assertNotIn(2, cached)
        self.assertEqual(cached.get(3), repo_b)

    def test_tick_survives_an_unexpected_exception_inside_sync_ci(self):
        """Defense in depth beyond _sync_ci()'s own try/excepts: even if
        it somehow raised anyway, _tick()'s own wrapper must keep the
        other three triggers running unaffected."""
        cfg = {"github_ci_enabled": True}
        _insert_task("t1", "failed", error="x", goal="y")

        class ExplodingLoop(_RecordingLoop):
            def _sync_ci(self):
                raise RuntimeError("boom")

        with patch.object(proactive, "load_config", lambda: cfg):
            loop = ExplodingLoop()
            loop._tick()  # must not raise

        self.assertEqual(len(loop.spoken), 1)
        self.assertIn("y", loop.spoken[0])


class CiRunTriggerTest(ProactiveTestCase):
    def test_completed_failure_nudges_with_workflow_title_and_repo(self):
        _insert_cached_ci_run(1, workflow="Tests", title="Fix login bug", conclusion="failure")
        loop = _RecordingLoop()
        loop._check_ci_runs()
        self.assertEqual(len(loop.spoken), 1)
        msg = loop.spoken[0]
        self.assertIn("Tests", msg)
        self.assertIn("failed", msg)
        self.assertIn("Fix login bug", msg)
        self.assertIn(proactive.CI_REPOS[0], msg)

    def test_completed_success_nudges_with_flat_non_celebratory_register(self):
        _insert_cached_ci_run(1, workflow="Tests", title="Fix login bug", conclusion="success")
        loop = _RecordingLoop()
        loop._check_ci_runs()
        self.assertEqual(len(loop.spoken), 1)
        msg = loop.spoken[0]
        self.assertIn("Tests", msg)
        self.assertIn("passed", msg)
        self.assertNotIn("!", msg)  # no celebratory framing, matches every other trigger's tone

    def test_in_progress_run_does_not_nudge(self):
        _insert_cached_ci_run(1, status="in_progress", conclusion=None)
        loop = _RecordingLoop()
        loop._check_ci_runs()
        self.assertEqual(loop.spoken, [])

    def test_queued_run_does_not_nudge(self):
        _insert_cached_ci_run(1, status="queued", conclusion=None)
        loop = _RecordingLoop()
        loop._check_ci_runs()
        self.assertEqual(loop.spoken, [])

    def test_cancelled_conclusion_does_not_nudge(self):
        """Only success/failure are nudge-worthy conclusions, per the
        plan — cancelled/skipped/timed_out/etc. are cached but silent."""
        _insert_cached_ci_run(1, status="completed", conclusion="cancelled")
        loop = _RecordingLoop()
        loop._check_ci_runs()
        self.assertEqual(loop.spoken, [])

    def test_second_check_does_not_renudge_the_same_run_same_conclusion(self):
        _insert_cached_ci_run(1, conclusion="failure")
        loop = _RecordingLoop()
        loop._check_ci_runs()
        loop._check_ci_runs()
        self.assertEqual(len(loop.spoken), 1)

    def test_dedup_is_durable_across_a_simulated_restart(self):
        _insert_cached_ci_run(1, conclusion="failure")
        first_run = _RecordingLoop()
        first_run._check_ci_runs()
        self.assertEqual(len(first_run.spoken), 1)

        second_run = _RecordingLoop()   # fresh in-memory state, same db
        second_run._check_ci_runs()
        self.assertEqual(second_run.spoken, [])

    def test_failure_then_later_success_on_the_same_run_id_both_nudge(self):
        """The exact scenario named in the plan: GitHub's 're-run failed
        jobs' reuses the same run_id and just updates its conclusion in
        place. Dedup keyed on (run_id, conclusion) — not run_id alone —
        must let the follow-up success through even though this run_id
        was already nudged once, as a failure."""
        _insert_cached_ci_run(1, workflow="Tests", conclusion="failure")
        loop = _RecordingLoop()
        loop._check_ci_runs()
        self.assertEqual(len(loop.spoken), 1)
        self.assertIn("failed", loop.spoken[0])

        # Simulate the re-run's effect: same run_id, cache now reflects
        # the new (successful) conclusion — exactly what a real
        # _sync_ci() would produce once GitHub's own conclusion flips.
        conn = db.get_conn()
        with conn:
            conn.execute("UPDATE ci_runs_cache SET conclusion = 'success' WHERE run_id = 1")

        loop._check_ci_runs()
        self.assertEqual(len(loop.spoken), 2, "the corrected success must still nudge, not be silently swallowed")
        self.assertIn("passed", loop.spoken[1])

        # And both dedup rows exist independently.
        self.assertTrue(db.nudge_already_sent(proactive.TRIGGER_CI_RUN, "1:failure"))
        self.assertTrue(db.nudge_already_sent(proactive.TRIGGER_CI_RUN, "1:success"))

    def test_repeated_check_without_a_real_conclusion_change_does_not_renudge(self):
        """Contrast case for the above: dedup must still hold when the
        conclusion genuinely hasn't changed — this isn't 'always re-nudge
        on re-check', only a real conclusion transition re-nudges."""
        _insert_cached_ci_run(1, conclusion="failure")
        loop = _RecordingLoop()
        loop._check_ci_runs()
        loop._check_ci_runs()  # conclusion unchanged — must not re-fire
        self.assertEqual(len(loop.spoken), 1)

    def test_two_runs_same_repo_and_sha_in_one_tick_are_grouped_into_one_nudge(self):
        repo = proactive.CI_REPOS[0]
        _insert_cached_ci_run(1, repo=repo, workflow="Lint", sha="abc123", conclusion="success")
        _insert_cached_ci_run(2, repo=repo, workflow="Tests", sha="abc123", conclusion="failure")
        loop = _RecordingLoop()
        loop._check_ci_runs()
        self.assertEqual(len(loop.spoken), 1, "expected one combined nudge, not one per run")
        msg = loop.spoken[0]
        self.assertIn("Lint passed", msg)
        self.assertIn("Tests failed", msg)
        self.assertIn(repo, msg)
        # Both runs' dedup rows are recorded even though only one
        # _gated_speak() call happened.
        self.assertTrue(db.nudge_already_sent(proactive.TRIGGER_CI_RUN, "1:success"))
        self.assertTrue(db.nudge_already_sent(proactive.TRIGGER_CI_RUN, "2:failure"))

    def test_three_runs_same_repo_and_sha_are_grouped_together(self):
        repo = proactive.CI_REPOS[0]
        _insert_cached_ci_run(1, repo=repo, workflow="Lint", sha="abc123", conclusion="success")
        _insert_cached_ci_run(2, repo=repo, workflow="Tests", sha="abc123", conclusion="failure")
        _insert_cached_ci_run(3, repo=repo, workflow="Deploy", sha="abc123", conclusion="success")
        loop = _RecordingLoop()
        loop._check_ci_runs()
        self.assertEqual(len(loop.spoken), 1)
        msg = loop.spoken[0]
        self.assertIn("Lint passed", msg)
        self.assertIn("Tests failed", msg)
        self.assertIn("Deploy passed", msg)

    def test_runs_different_sha_same_repo_are_not_grouped(self):
        repo = proactive.CI_REPOS[0]
        _insert_cached_ci_run(1, repo=repo, workflow="Lint", sha="sha-one", conclusion="success")
        _insert_cached_ci_run(2, repo=repo, workflow="Tests", sha="sha-two", conclusion="failure")
        loop = _RecordingLoop()
        loop._check_ci_runs()
        self.assertEqual(len(loop.spoken), 2, "different commits must nudge separately, not be combined")

    def test_runs_different_repo_same_sha_are_not_grouped(self):
        repo_a, repo_b = proactive.CI_REPOS
        _insert_cached_ci_run(1, repo=repo_a, workflow="Lint", sha="same-sha", conclusion="success")
        _insert_cached_ci_run(2, repo=repo_b, workflow="Tests", sha="same-sha", conclusion="failure")
        loop = _RecordingLoop()
        loop._check_ci_runs()
        self.assertEqual(len(loop.spoken), 2)

    def test_a_run_already_nudged_in_a_prior_tick_is_not_pulled_into_a_new_groups_message(self):
        repo = proactive.CI_REPOS[0]
        _insert_cached_ci_run(1, repo=repo, workflow="Lint", sha="abc123", conclusion="success")
        loop = _RecordingLoop()
        loop._check_ci_runs()
        self.assertEqual(len(loop.spoken), 1)
        self.assertNotIn("Tests", loop.spoken[0])

        # A second, slower workflow for the same push completes on a
        # later tick — must nudge on its own, not silently merge into
        # (or get suppressed by) the already-nudged first run.
        _insert_cached_ci_run(2, repo=repo, workflow="Tests", sha="abc123", conclusion="failure")
        loop._check_ci_runs()
        self.assertEqual(len(loop.spoken), 2)
        self.assertIn("Tests", loop.spoken[1])
        self.assertNotIn("Lint", loop.spoken[1])

    def test_gating_suppresses_single_ci_nudge_and_leaves_it_undedup(self):
        _insert_cached_ci_run(1, conclusion="failure")
        loop = _RecordingLoop(is_muted=lambda: True)
        loop._check_ci_runs()
        self.assertEqual(loop.spoken, [])
        self.assertFalse(db.nudge_already_sent(proactive.TRIGGER_CI_RUN, "1:failure"))

    def test_gating_suppresses_grouped_nudge_and_leaves_all_members_undedup(self):
        repo = proactive.CI_REPOS[0]
        _insert_cached_ci_run(1, repo=repo, workflow="Lint", sha="abc123", conclusion="success")
        _insert_cached_ci_run(2, repo=repo, workflow="Tests", sha="abc123", conclusion="failure")
        loop = _RecordingLoop(is_muted=lambda: True)
        loop._check_ci_runs()
        self.assertEqual(loop.spoken, [])
        self.assertFalse(db.nudge_already_sent(proactive.TRIGGER_CI_RUN, "1:success"))
        self.assertFalse(db.nudge_already_sent(proactive.TRIGGER_CI_RUN, "2:failure"))


class TickCadenceTest(ProactiveTestCase):
    def test_sync_runs_on_first_tick_then_not_again_before_five_minutes(self):
        cfg = {"calendar_enabled": True}
        sync_calls: list[datetime] = []

        class CountingLoop(_RecordingLoop):
            def _sync_calendar(self):
                sync_calls.append(self._now())

        clock = {"t": datetime(2026, 8, 10, 9, 0, 0)}
        loop  = CountingLoop(now=lambda: clock["t"])

        with patch.object(proactive, "load_config", lambda: cfg):
            loop._tick()                                           # t=0:00 -> due (first ever)
            clock["t"] += timedelta(seconds=proactive.POLL_INTERVAL_SEC)
            loop._tick()                                           # t=1:00 -> not due
            clock["t"] += timedelta(seconds=proactive.POLL_INTERVAL_SEC)
            loop._tick()                                           # t=2:00 -> not due
            clock["t"] += timedelta(seconds=proactive.CALENDAR_SYNC_INTERVAL_SEC)
            loop._tick()                                           # t=7:00 -> due again

        self.assertEqual(len(sync_calls), 2)

    def test_non_sync_ticks_never_touch_calendar_auth_at_all(self):
        """A tick that isn't due for a sync must not call into the
        calendar client in any way — proves its behavior (and therefore
        its timing) can't depend on calendar latency, without needing a
        real network call to demonstrate it."""
        cfg     = {"calendar_enabled": True}
        touched = {"count": 0}

        class _CountingEvents:
            def list(self, **kwargs):
                touched["count"] += 1
                return _FakeEventsCall({"items": []})

        class _CountingService:
            def events(self):
                return _CountingEvents()

        fake_auth = _fake_calendar_auth_module(_CountingService())
        clock     = {"t": datetime(2026, 8, 10, 9, 0, 0)}
        loop      = _RecordingLoop(now=lambda: clock["t"])

        with patch.object(proactive, "load_config", lambda: cfg), \
             patch.dict(sys.modules, {"core.calendar_auth": fake_auth}):
            loop._tick()                                            # due
            clock["t"] += timedelta(seconds=proactive.POLL_INTERVAL_SEC)
            loop._tick()                                            # not due
            clock["t"] += timedelta(seconds=proactive.POLL_INTERVAL_SEC)
            loop._tick()                                            # not due

        self.assertEqual(touched["count"], 1)

    def test_sync_latency_is_isolated_to_the_tick_that_is_due(self):
        """Times a due tick against a simulated slow calendar API call
        against a not-due tick immediately after, to show the extra
        latency is confined to the sync tick — a non-sync tick's timing
        is structurally independent of calendar latency."""
        cfg = {"calendar_enabled": True}
        SIMULATED_NETWORK_DELAY_SEC = 0.15

        class _SlowEventsCall:
            def execute(self):
                time.sleep(SIMULATED_NETWORK_DELAY_SEC)
                return {"items": []}

        class _SlowEvents:
            def list(self, **kwargs):
                return _SlowEventsCall()

        class _SlowService:
            def events(self):
                return _SlowEvents()

        fake_auth = _fake_calendar_auth_module(_SlowService())
        clock     = {"t": datetime(2026, 8, 10, 9, 0, 0)}
        loop      = _RecordingLoop(now=lambda: clock["t"])

        with patch.object(proactive, "load_config", lambda: cfg), \
             patch.dict(sys.modules, {"core.calendar_auth": fake_auth}):
            due_start    = time.monotonic()
            loop._tick()                                            # due -> pays the delay
            due_duration = time.monotonic() - due_start

            clock["t"] += timedelta(seconds=proactive.POLL_INTERVAL_SEC)
            not_due_start    = time.monotonic()
            loop._tick()                                            # not due -> must not pay it again
            not_due_duration = time.monotonic() - not_due_start

        print(
            f"\n[timing] sync tick: {due_duration:.3f}s, "
            f"non-sync tick: {not_due_duration:.3f}s "
            f"(simulated network delay: {SIMULATED_NETWORK_DELAY_SEC}s)"
        )
        self.assertGreaterEqual(due_duration, SIMULATED_NETWORK_DELAY_SEC)
        self.assertLess(not_due_duration, SIMULATED_NETWORK_DELAY_SEC / 2)

    def test_ci_sync_runs_on_first_tick_then_not_again_before_five_minutes(self):
        cfg = {"github_ci_enabled": True}
        sync_calls: list[datetime] = []

        class CountingLoop(_RecordingLoop):
            def _sync_ci(self):
                sync_calls.append(self._now())

        clock = {"t": datetime(2026, 8, 10, 9, 0, 0)}
        loop  = CountingLoop(now=lambda: clock["t"])

        with patch.object(proactive, "load_config", lambda: cfg):
            loop._tick()                                           # t=0:00 -> due (first ever)
            clock["t"] += timedelta(seconds=proactive.POLL_INTERVAL_SEC)
            loop._tick()                                           # t=1:00 -> not due
            clock["t"] += timedelta(seconds=proactive.POLL_INTERVAL_SEC)
            loop._tick()                                           # t=2:00 -> not due
            clock["t"] += timedelta(seconds=proactive.CI_SYNC_INTERVAL_SEC)
            loop._tick()                                           # t=7:00 -> due again

        self.assertEqual(len(sync_calls), 2)

    def test_non_sync_ticks_never_touch_requests_get(self):
        """A tick that isn't due for a CI sync must not call into
        requests.get at all — proves its timing can't depend on GitHub
        API latency, without needing a real network call to demonstrate it."""
        cfg = {"github_ci_enabled": True}
        fake_auth = _fake_github_ci_auth_module("fake-token")
        touched   = {"count": 0}

        def _counting_get(url, headers=None, params=None, timeout=None):
            touched["count"] += 1
            return _FakeGithubResponse({"workflow_runs": []})

        clock = {"t": datetime(2026, 8, 10, 9, 0, 0)}
        loop  = _RecordingLoop(now=lambda: clock["t"])

        with patch.object(proactive, "load_config", lambda: cfg), \
             patch("core.github_ci_auth", fake_auth, create=True), \
             patch("requests.get", side_effect=_counting_get):
            loop._tick()                                            # due
            clock["t"] += timedelta(seconds=proactive.POLL_INTERVAL_SEC)
            loop._tick()                                            # not due
            clock["t"] += timedelta(seconds=proactive.POLL_INTERVAL_SEC)
            loop._tick()                                            # not due

        # One tick due -> one requests.get call per watched repo.
        self.assertEqual(touched["count"], len(proactive.CI_REPOS))

    def test_calendar_and_ci_sync_cadences_are_independent(self):
        """Both decoupled timers live in the same _tick() now — confirm
        they don't interfere with or accidentally share state: CI syncing
        on its own schedule must not perturb calendar's, and vice versa."""
        cfg = {"calendar_enabled": True, "github_ci_enabled": True}
        calendar_calls: list[datetime] = []
        ci_calls:       list[datetime] = []

        class CountingLoop(_RecordingLoop):
            def _sync_calendar(self):
                calendar_calls.append(self._now())

            def _sync_ci(self):
                ci_calls.append(self._now())

        clock = {"t": datetime(2026, 8, 10, 9, 0, 0)}
        loop  = CountingLoop(now=lambda: clock["t"])

        with patch.object(proactive, "load_config", lambda: cfg):
            loop._tick()                                            # t=0:00 -> both due (first ever)
            clock["t"] += timedelta(seconds=proactive.CALENDAR_SYNC_INTERVAL_SEC)
            loop._tick()                                            # t=5:00 -> both due again (same interval)

        self.assertEqual(len(calendar_calls), 2)
        self.assertEqual(len(ci_calls), 2)


if __name__ == "__main__":
    unittest.main()
