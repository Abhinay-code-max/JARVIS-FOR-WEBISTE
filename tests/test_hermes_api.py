"""
tests/test_hermes_api.py
==========================
Headless-extraction phase, Step 1.5: api/hermes_app.py's FastAPI +
WebSocket network boundary. Uses FastAPI's TestClient throughout — an
in-process ASGI transport that binds no socket at all, strictly safer
than even 127.0.0.1 — so nothing here comes anywhere near the still-open
deployment-reachability question (Step 1.5.x). Real dispatch_tool(),
real core.policy gating, real rate limiter, real hmac comparison; only
TOOL_DISPATCH entries are stubbed where a step calls a real tool, same
convention as every other integration test in this package.

Redirects core.db at a fresh temp sqlite file and core.hermes_auth's
token storage at a fresh temp directory per test, so nothing here
touches data/jarvis.db or ~/.jarvis/hermes_tokens/.
"""
from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.db as db


def _use_temp_db() -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="jarvis_test_db_"))
    db.DB_DIR  = tmp_dir
    db.DB_PATH = tmp_dir / "test.db"
    db._local  = threading.local()
    return db.DB_PATH


_use_temp_db()

import core.policy          as policy        # noqa: E402
import core.hermes_auth     as hermes_auth   # noqa: E402
import core.tool_dispatch   as tdisp         # noqa: E402
import core.task_approval   as ta_mod        # noqa: E402
import agent.executor       as executor      # noqa: E402
import api.hermes_app       as hermes_app    # noqa: E402
from fastapi.testclient import TestClient    # noqa: E402


def _use_temp_tokens() -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="jarvis_test_hermes_tokens_"))
    hermes_auth._TOKENS_DIR = tmp_dir
    return tmp_dir


def _reset_rate_limiters(auth_max=120, route_max=60):
    hermes_app._AUTH_RATE_LIMITER  = hermes_app._RateLimiter(max_calls=auth_max,  window_sec=60.0)
    hermes_app._ROUTE_RATE_LIMITER = hermes_app._RateLimiter(max_calls=route_max, window_sec=60.0)


class HermesApiTestCase(unittest.TestCase):
    def setUp(self):
        _use_temp_db()
        # Called directly (idempotent) rather than relying on
        # get_policy_level()'s lazy _ensure_seeded() cache — see
        # core/policy.py's _seeded_for_path comment for the real,
        # DB-path-keyed fix this test file's own hang originally
        # surfaced (a bare bool cache going stale across test files that
        # each redirect DB_PATH to their own temp file).
        policy.seed_default_policy()
        _use_temp_tokens()
        _reset_rate_limiters()
        ta_mod.TASK_APPROVAL._pending.clear()
        self.app    = hermes_app.create_app()
        self.client = TestClient(self.app)

        # POST /tasks submits through the REAL, global background
        # TaskQueue (a module-level singleton — once started, its worker
        # thread keeps running for the rest of this test process). Left
        # unstubbed, AgentExecutor.execute() would call the real
        # create_plan(), a real Ollama call, in the background for every
        # task these tests submit — slow, network-dependent, and (per
        # the weather_report lesson in WebSocketTest below) a real risk
        # of leaving abandoned background work that interferes with
        # later tests. Stubbed to end each submitted task immediately
        # with no real LLM/tool call — these tests are about the
        # submission/gating boundary itself, already covered against the
        # real AgentExecutor step loop by tests/test_caller_attribution.py.
        self._orig_create_plan = executor.create_plan
        executor.create_plan   = lambda goal, task_id=None: {"steps": []}

    def tearDown(self):
        executor.create_plan = self._orig_create_plan

    def _mint(self, caller_class: str) -> str:
        return hermes_auth.mint_token(caller_class)

    def _wait_for_task(self, task_id: str, timeout: float = 5.0) -> dict:
        """Every test that submits through POST /tasks must wait for the
        real, global background TaskQueue to actually finish it before
        the test method returns. Found the hard way: the queue is a
        module-level singleton whose worker thread keeps running for the
        rest of the *process* once started — a task left in-flight when
        a test returns can get processed later, on the SAME shared
        agent.executor.create_plan/dispatch_tool/... module attributes a
        completely unrelated, later test file (tests/test_verification.py,
        specifically) monkey-patches for its own purposes, silently
        inflating that other test's own call-counting assertions. Not
        reproducible running this file in isolation — only surfaced
        running the full suite, alphabetically right before
        test_verification.py."""
        from agent.task_queue import get_queue
        deadline = time.time() + timeout
        status = None
        while time.time() < deadline:
            status = get_queue().get_status(task_id)
            if status and status["status"] in ("completed", "failed", "cancelled"):
                return status
            time.sleep(0.02)
        self.fail(f"task {task_id} never reached a terminal status within {timeout}s: {status}")


class AuthRejectionTest(HermesApiTestCase):
    def test_no_header_is_rejected(self):
        r = self.client.get("/approvals")
        self.assertEqual(r.status_code, 401)

    def test_malformed_header_is_rejected(self):
        r = self.client.get("/approvals", headers={"Authorization": "NotBearer xyz"})
        self.assertEqual(r.status_code, 401)

    def test_empty_bearer_token_is_rejected(self):
        r = self.client.get("/approvals", headers={"Authorization": "Bearer "})
        self.assertEqual(r.status_code, 401)

    def test_wrong_token_is_rejected_even_when_a_real_one_exists(self):
        self._mint(policy.DESKTOP)
        r = self.client.get("/approvals", headers={"Authorization": "Bearer completely-wrong"})
        self.assertEqual(r.status_code, 401)

    def test_correct_token_is_accepted(self):
        token = self._mint(policy.DESKTOP)
        r = self.client.get("/approvals", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(r.status_code, 200)

    def test_every_caller_class_can_authenticate_with_its_own_token(self):
        for caller_class in policy.CALLER_CLASSES:
            token = hermes_auth.mint_token(caller_class)
            r = self.client.get("/approvals", headers={"Authorization": f"Bearer {token}"})
            self.assertEqual(r.status_code, 200, f"{caller_class} failed to authenticate with its own token")

    def test_one_class_token_does_not_authenticate_as_another(self):
        desktop_token = self._mint(policy.DESKTOP)
        self._mint(policy.SERVICE_SUPPORT)
        # desktop's token must resolve to caller_class='desktop', not
        # accidentally match anything else — verified indirectly via
        # POST /tasks, which only 'desktop' is allowed to use (Step 1.2b).
        r = self.client.post("/tasks", json={"goal": "x"}, headers={"Authorization": f"Bearer {desktop_token}"})
        self.assertNotEqual(r.status_code, 403)
        self._wait_for_task(r.json()["task_id"])

    def test_no_route_is_reachable_without_auth(self):
        """Structural claim from the module docstring, checked directly:
        every REST route registered on rest_router is behind
        _authenticate — not just the one or two this file happens to
        probe individually elsewhere."""
        for route in hermes_app.rest_router.routes:
            for method in route.methods - {"HEAD"}:
                r = self.client.request(method, route.path.replace("{task_id}", "x").replace("{approval_id}", "1"))
                self.assertEqual(
                    r.status_code, 401,
                    f"{method} {route.path} was reachable without authentication",
                )


class TokenReReadTest(HermesApiTestCase):
    def test_revoked_token_stops_working_on_the_very_next_request_no_restart(self):
        """The plan's explicit requirement: never cache a token in a
        module-level constant. Proven directly: mint, use successfully,
        re-mint (revoking the old value), confirm the OLD token is
        rejected on the very next request with no process restart, and
        the NEW token works."""
        old_token = self._mint(policy.DESKTOP)
        r1 = self.client.get("/approvals", headers={"Authorization": f"Bearer {old_token}"})
        self.assertEqual(r1.status_code, 200)

        new_token = self._mint(policy.DESKTOP)  # revoke-and-reissue
        self.assertNotEqual(old_token, new_token)

        r2 = self.client.get("/approvals", headers={"Authorization": f"Bearer {old_token}"})
        self.assertEqual(r2.status_code, 401, "old token still worked after re-minting — token was cached")

        r3 = self.client.get("/approvals", headers={"Authorization": f"Bearer {new_token}"})
        self.assertEqual(r3.status_code, 200)


class HmacComparisonTest(unittest.TestCase):
    def test_source_uses_hmac_compare_digest_not_equality(self):
        """Static check that the actual comparison operator used is
        hmac.compare_digest — a functional wrong-token-rejected test
        (covered elsewhere) can't distinguish constant-time comparison
        from a plain ==, so this checks the real source directly."""
        src = (Path(__file__).resolve().parent.parent / "api" / "hermes_app.py").read_text(encoding="utf-8")
        self.assertIn("hmac.compare_digest(stored, presented)", src)
        # And confirm the token comparison itself never uses a plain ==
        # anywhere nearby (defense against a second, forgotten comparison
        # path being added later).
        self.assertNotIn("stored == presented", src)
        self.assertNotIn("presented == stored", src)


class RateLimitTest(HermesApiTestCase):
    def test_route_rate_limit_blocks_after_the_configured_max(self):
        _reset_rate_limiters(route_max=3)
        token = self._mint(policy.DESKTOP)
        statuses = [
            self.client.get("/approvals", headers={"Authorization": f"Bearer {token}"}).status_code
            for _ in range(5)
        ]
        self.assertEqual(statuses, [200, 200, 200, 429, 429])

    def test_route_rate_limit_is_independent_per_caller_class(self):
        _reset_rate_limiters(route_max=2)
        desktop_token = self._mint(policy.DESKTOP)
        support_token = self._mint(policy.SERVICE_SUPPORT)

        for _ in range(2):
            self.assertEqual(
                self.client.get("/approvals", headers={"Authorization": f"Bearer {desktop_token}"}).status_code, 200,
            )
        # desktop is now at its limit — support must be unaffected.
        self.assertEqual(
            self.client.get("/approvals", headers={"Authorization": f"Bearer {desktop_token}"}).status_code, 429,
        )
        self.assertEqual(
            self.client.get("/approvals", headers={"Authorization": f"Bearer {support_token}"}).status_code, 200,
        )

    def test_route_rate_limit_only_counts_requests_that_passed_auth(self):
        """The plan's specific claim: the per-route limit counts only
        post-auth requests. Flood failed-auth attempts first, then
        confirm a real, correctly-authenticated request still gets
        through — a failed attempt must not consume the ROUTE budget
        (only the separate auth-gate budget, tested below)."""
        _reset_rate_limiters(route_max=3)
        token = self._mint(policy.DESKTOP)

        for _ in range(10):
            r = self.client.get("/approvals", headers={"Authorization": "Bearer wrong"})
            self.assertEqual(r.status_code, 401)

        # The real token's route budget must be untouched by those 10
        # failures — still 3 successes available.
        statuses = [
            self.client.get("/approvals", headers={"Authorization": f"Bearer {token}"}).status_code
            for _ in range(3)
        ]
        self.assertEqual(statuses, [200, 200, 200])

    def test_auth_gate_rate_limit_blocks_a_flood_of_wrong_token_attempts(self):
        _reset_rate_limiters(auth_max=5, route_max=1000)
        statuses = [
            self.client.get("/approvals", headers={"Authorization": "Bearer wrong"}).status_code
            for _ in range(8)
        ]
        self.assertEqual(statuses[:5], [401] * 5)
        # The remaining attempts are blocked by the AUTH gate itself —
        # still a 401-shaped rejection from the caller's perspective (no
        # route was ever reached), but distinctly logged as a rate-limit
        # hit server-side rather than a bad-credential attempt.
        self.assertTrue(all(s == 401 for s in statuses[5:]))


class AuditLoggingTest(HermesApiTestCase):
    """jarvis.hermes_api's logger has no handler attached at all unless
    core/logging_setup.py's init_logging() has been called (real
    deployments call it once at startup; this test process never does).
    Attaches core/logging_setup.py's own SQLiteHandler directly and
    synchronously — same reasoning as
    tests/test_caller_attribution.py's LogEventsAttributionTest: the
    full QueueHandler/QueueListener pipeline is a long-lived background
    thread behind shared module-level state, not worth the flakiness
    risk for what this is actually testing (the extra={} -> SQL mapping,
    already covered directly in Step 1.3's own tests) — AuditedRoute's
    own _log.info(..., extra={...}) call is exactly the same either way."""

    def setUp(self):
        super().setUp()
        import logging
        import core.logging_setup as logging_setup
        self._logger  = logging.getLogger("jarvis.hermes_api")
        self._handler = logging_setup.SQLiteHandler()
        self._logger.addHandler(self._handler)
        self._logger.setLevel(logging.INFO)

    def tearDown(self):
        self._logger.removeHandler(self._handler)
        super().tearDown()

    def test_successful_request_is_logged_with_caller_attribution(self):
        token = self._mint(policy.SERVICE_SUPPORT)
        self.client.get("/approvals", headers={"Authorization": f"Bearer {token}"})

        row = db.get_conn().execute(
            "SELECT source, message, caller_class, triggered_by FROM log_events "
            "WHERE source = 'jarvis.hermes_api' ORDER BY event_id DESC LIMIT 1"
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertIn("GET /approvals", row["message"])
        self.assertIn("200", row["message"])
        self.assertEqual(row["caller_class"], policy.SERVICE_SUPPORT)
        self.assertEqual(row["triggered_by"], policy.SERVICE_SUPPORT)

    def test_rejected_request_is_still_logged(self):
        self.client.get("/approvals", headers={"Authorization": "Bearer wrong"})
        row = db.get_conn().execute(
            "SELECT message FROM log_events WHERE source = 'jarvis.hermes_api' ORDER BY event_id DESC LIMIT 1"
        ).fetchone()
        self.assertIsNotNone(row, "a rejected request produced no audit trail at all")
        self.assertIn("401", row["message"])


class AgentTaskGatingOverApiTest(HermesApiTestCase):
    def test_desktop_can_submit_a_task(self):
        token = self._mint(policy.DESKTOP)
        r = self.client.post("/tasks", json={"goal": "check weather"}, headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIn("task_id", r.json())
        self._wait_for_task(r.json()["task_id"])

    def test_service_bugfix_cannot_submit_a_task(self):
        """Reuses core.policy.is_agent_task_allowed() (Step 1.2b) over
        the real API boundary — the same hard-deny, no exceptions."""
        token = self._mint(policy.SERVICE_BUGFIX)
        r = self.client.post("/tasks", json={"goal": "fix a bug"}, headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(r.status_code, 403)

    def test_every_service_class_is_denied_task_submission(self):
        for caller_class in policy.SERVICE_CALLER_CLASSES:
            token = hermes_auth.mint_token(caller_class)
            r = self.client.post("/tasks", json={"goal": "x"}, headers={"Authorization": f"Bearer {token}"})
            self.assertEqual(r.status_code, 403, f"{caller_class} was allowed to submit a task")


class TaskAndApprovalEndpointsTest(HermesApiTestCase):
    def test_task_status_round_trip(self):
        token = self._mint(policy.DESKTOP)
        submit = self.client.post("/tasks", json={"goal": "check weather"}, headers={"Authorization": f"Bearer {token}"})
        task_id = submit.json()["task_id"]

        r = self.client.get(f"/tasks/{task_id}", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["task_id"], task_id)
        self._wait_for_task(task_id)

    def test_unknown_task_id_is_404(self):
        token = self._mint(policy.DESKTOP)
        r = self.client.get("/tasks/does-not-exist", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(r.status_code, 404)

    def test_approval_resolution_round_trip(self):
        token = self._mint(policy.DESKTOP)

        def _wait():
            ta_mod.TASK_APPROVAL.wait_for_approval("t1", "delete something", timeout=5)

        t = threading.Thread(target=_wait, daemon=True)
        t.start()
        deadline = time.time() + 2
        pending = []
        while time.time() < deadline and not pending:
            pending = ta_mod.TASK_APPROVAL.list_pending()
            time.sleep(0.01)
        self.assertEqual(len(pending), 1)
        approval_id = pending[0]["approval_id"]

        r = self.client.get("/approvals", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(len(r.json()), 1)

        r = self.client.post(f"/approvals/{approval_id}", json={"approve": True}, headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(r.status_code, 200)
        t.join(timeout=2)

    def test_resolving_an_unknown_approval_is_404(self):
        token = self._mint(policy.DESKTOP)
        r = self.client.post("/approvals/999999", json={"approve": True}, headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(r.status_code, 404)

    def test_memory_endpoint_returns_a_snapshot(self):
        token = self._mint(policy.DESKTOP)
        r = self.client.get("/memory", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(r.status_code, 200)
        self.assertIsInstance(r.json(), dict)


class WebSocketTest(HermesApiTestCase):
    def test_wrong_token_never_establishes_the_connection(self):
        with self.assertRaises(Exception):
            with self.client.websocket_connect("/ws", headers={"Authorization": "Bearer wrong"}):
                pass

    def test_missing_header_never_establishes_the_connection(self):
        with self.assertRaises(Exception):
            with self.client.websocket_connect("/ws"):
                pass

    def test_real_tool_call_relay_round_trip(self):
        token = self._mint(policy.DESKTOP)
        orig = tdisp.TOOL_DISPATCH.get("weather_report")
        tdisp.TOOL_DISPATCH["weather_report"] = lambda args, player, speak: "Sunny, 20C."
        try:
            with self.client.websocket_connect("/ws", headers={"Authorization": f"Bearer {token}"}) as ws:
                ws.send_json({"tool": "weather_report", "args": {"city": "Paris"}})
                data = ws.receive_json()
        finally:
            if orig is not None:
                tdisp.TOOL_DISPATCH["weather_report"] = orig
        self.assertEqual(data, {"result": "Sunny, 20C."})

    def test_service_caller_class_flows_through_to_dispatch_tool_over_ws(self):
        """A service:* caller hitting an unlisted tool over WS must still
        get the real hard-deny — not the desktop default — proving
        caller_class genuinely reaches dispatch_tool() from this path,
        not just the REST one."""
        token = hermes_auth.mint_token(policy.SERVICE_PROMOTIONS)  # empty allow-list
        with self.client.websocket_connect("/ws", headers={"Authorization": f"Bearer {token}"}) as ws:
            # A real, valid city — otherwise validate_input() rejects the
            # call before the policy gate is even reached, which would
            # prove nothing about the hard-deny path this test is for.
            ws.send_json({"tool": "weather_report", "args": {"city": "Paris"}})
            data = ws.receive_json()
        self.assertEqual(data, {"result": "'weather_report' is not permitted."})

    def test_malformed_message_does_not_crash_the_connection(self):
        token = self._mint(policy.DESKTOP)
        orig = tdisp.TOOL_DISPATCH.get("weather_report")
        # Stubbed for the same reason every other test in this file
        # stubs it — the real tool makes a genuine network call
        # (actions/weather_report.py's requests.get(..., timeout=10)).
        # Found the hard way: an earlier version of this test left it
        # unstubbed, and dispatch_tool()'s timeout wrapper abandons (but
        # does not kill) the underlying thread on timeout — see
        # core/tool_gate.py's _invoke_with_timeout docstring — so the
        # real, still-running network call outlived this test and
        # deadlocked the *next* WebSocket test in the same process.
        tdisp.TOOL_DISPATCH["weather_report"] = lambda args, player, speak: "Sunny, 20C."
        try:
            with self.client.websocket_connect("/ws", headers={"Authorization": f"Bearer {token}"}) as ws:
                ws.send_text("not json")
                data = ws.receive_json()
                self.assertIn("error", data)
                # Connection must still be alive after a bad message.
                ws.send_json({"tool": "weather_report", "args": {}})
                data2 = ws.receive_json()
                self.assertIn("result", data2)
        finally:
            if orig is not None:
                tdisp.TOOL_DISPATCH["weather_report"] = orig

    def test_ws_route_count_matches_the_documented_single_bypass_caveat(self):
        """Static check backing the module docstring's honesty about
        scope: exactly one websocket route exists today, so the "a
        second ws route would need its own auth call" caveat is
        accurate, not stale."""
        ws_routes = [r for r in hermes_app.create_app().routes if getattr(r, "path", None) == "/ws"]
        self.assertEqual(len(ws_routes), 1)


class LocalOnlyBindingTest(unittest.TestCase):
    def test_run_entrypoint_is_hardcoded_to_localhost(self):
        src = (Path(__file__).resolve().parent.parent / "api" / "hermes_app.py").read_text(encoding="utf-8")
        # Scoped to the actual __main__ block specifically — the module's
        # own module docstring intentionally *mentions* "0.0.0.0" as
        # cautionary prose (warning against ever passing it), so a bare
        # whole-file substring check would trip on its own warning text.
        main_block_start = src.find('if __name__ == "__main__":')
        self.assertNotEqual(main_block_start, -1)
        main_block = src[main_block_start:]
        self.assertIn('host="127.0.0.1"', main_block)
        self.assertNotIn("0.0.0.0", main_block)


if __name__ == "__main__":
    unittest.main(verbosity=2)
