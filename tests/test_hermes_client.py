"""
tests/test_hermes_client.py
=============================
Headless-extraction phase, Step 1.7: core/hermes_client.py, the desktop
app's own HTTP client of its local Hermes server (used by main.py's
JarvisXL._execute_tool for the agent_task and approval branches — see
that module for the Hermes-first, in-process-fallback shape this client
exists to support).

Unlike tests/test_hermes_api.py (which uses FastAPI's TestClient — an
in-process ASGI transport binding no real socket at all), this module's
`requests`-based client genuinely needs a real socket to exercise, since
there's no requests-to-ASGI adapter in this repo's dependencies. Runs a
real uvicorn server in a background thread, bound to 127.0.0.1 on an
ephemeral port (never 8765, the production port) — still fully inside
the reachability constraint (see api/hermes_app.py's module docstring):
this is 127.0.0.1 either way, just over a real socket instead of
TestClient's in-process transport.

Redirects core.db and core.hermes_auth's token storage at fresh temp
locations per test, same convention as every other test module in this
package.
"""
from __future__ import annotations

import socket
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

import uvicorn                               # noqa: E402
import core.policy         as policy         # noqa: E402
import core.hermes_auth    as hermes_auth    # noqa: E402
import core.hermes_client  as hermes_client  # noqa: E402
import core.task_approval  as ta_mod         # noqa: E402
import agent.executor      as executor       # noqa: E402
import api.hermes_app      as hermes_app     # noqa: E402


def _use_temp_tokens() -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="jarvis_test_hermes_tokens_"))
    hermes_auth._TOKENS_DIR = tmp_dir
    return tmp_dir


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _LiveHermesServer:
    """A real uvicorn server on a real 127.0.0.1 socket — the one thing
    FastAPI's TestClient (used everywhere else in this phase) can't
    stand in for, and the exact shape main.py's own
    _start_hermes_server() uses in production."""

    def __init__(self):
        self.port    = _free_port()
        self.config  = uvicorn.Config(hermes_app.create_app(), host="127.0.0.1", port=self.port, log_level="error")
        self.server  = uvicorn.Server(self.config)
        self._thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self):
        self._thread.start()
        deadline = time.time() + 5.0
        while not self.server.started and time.time() < deadline:
            time.sleep(0.02)
        if not self.server.started:
            raise RuntimeError("test Hermes server did not start in time")

    def stop(self):
        self.server.should_exit = True
        self._thread.join(timeout=5.0)


class HermesClientTest(unittest.TestCase):
    def setUp(self):
        _use_temp_db()
        policy.seed_default_policy()
        _use_temp_tokens()
        hermes_app._AUTH_RATE_LIMITER  = hermes_app._RateLimiter(max_calls=120, window_sec=60.0)
        hermes_app._ROUTE_RATE_LIMITER = hermes_app._RateLimiter(max_calls=60,  window_sec=60.0)
        ta_mod.TASK_APPROVAL._pending.clear()

        self._orig_create_plan = executor.create_plan
        executor.create_plan   = lambda goal, task_id=None: {"steps": []}

        self.server = _LiveHermesServer()
        self.server.start()
        self._orig_base_url    = hermes_client.BASE_URL
        hermes_client.BASE_URL = f"http://127.0.0.1:{self.server.port}"

    def tearDown(self):
        self.server.stop()
        hermes_client.BASE_URL = self._orig_base_url
        executor.create_plan   = self._orig_create_plan

    def test_no_token_minted_raises_hermes_unavailable(self):
        with self.assertRaises(hermes_client.HermesUnavailable):
            hermes_client.submit_task("do a thing")

    def test_headers_raises_directly_without_a_token(self):
        """Unlike the end-to-end test above (where a missing token would
        still surface as HermesUnavailable via the server's own 401, even
        if this local guard were deleted — see this file's Step 1.7
        mutation-testing note), this calls _headers() directly so the
        guard itself, not just the overall contract, is under test."""
        with self.assertRaises(hermes_client.HermesUnavailable):
            hermes_client._headers()

    def test_submit_and_poll_a_real_task(self):
        hermes_auth.mint_token(policy.DESKTOP)
        task_id = hermes_client.submit_task("do a thing", priority="normal")
        self.assertTrue(task_id)

        deadline = time.time() + 5.0
        status = None
        while time.time() < deadline:
            status = hermes_client.get_task_status(task_id)
            if status and status["status"] in ("completed", "failed", "cancelled"):
                break
            time.sleep(0.05)
        self.assertIsNotNone(status)
        self.assertEqual(status["status"], "completed", status)

    def test_get_task_status_unknown_id_returns_none(self):
        hermes_auth.mint_token(policy.DESKTOP)
        self.assertIsNone(hermes_client.get_task_status("no-such-task"))

    def test_list_and_resolve_a_real_approval(self):
        hermes_auth.mint_token(policy.DESKTOP)

        result_holder = {}
        def _run():
            result_holder["result"] = ta_mod.TASK_APPROVAL.wait_for_approval(
                "t-1", "Approve this?", timeout=10.0
            )
        t = threading.Thread(target=_run, daemon=True)
        t.start()

        deadline = time.time() + 5.0
        pending  = []
        while time.time() < deadline:
            pending = hermes_client.list_pending_approvals()
            if pending:
                break
            time.sleep(0.05)
        self.assertTrue(pending, "approval never appeared via the real HTTP client")
        approval_id = pending[0]["approval_id"]

        ok = hermes_client.resolve_approval(approval_id, approve=True)
        self.assertTrue(ok)
        t.join(timeout=5.0)
        self.assertEqual(result_holder["result"][0], True)

    def test_resolve_unknown_approval_returns_false(self):
        hermes_auth.mint_token(policy.DESKTOP)
        self.assertFalse(hermes_client.resolve_approval(999999, approve=True))

    def test_server_unreachable_raises_hermes_unavailable(self):
        hermes_auth.mint_token(policy.DESKTOP)
        hermes_client.BASE_URL = "http://127.0.0.1:1"  # nothing listens here
        with self.assertRaises(hermes_client.HermesUnavailable):
            hermes_client.submit_task("do a thing")


if __name__ == "__main__":
    unittest.main(verbosity=2)
