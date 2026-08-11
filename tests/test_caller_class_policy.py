"""
tests/test_caller_class_policy.py
===================================
The headless-extraction phase's Step 1.2: permission_policy's new
caller_class dimension. Covers:
  - desktop's existing rows/behavior are unchanged (default-preserving
    caller_class parameter, real full-suite regression already covers
    this broadly — this file adds a few direct, explicit checks)
  - each service:* class's explicit allow-list, seeded correctly
  - the default-deny fallback for service:* classes (materially
    different from desktop's default-ask-and-wait fallback)
  - file_controller's additional path-prefix scoping for service:bugfix
  - the DELEGATED_TOOLS caveat: a service:* allow-list row for
    dev_agent/vision_fix_code is a structural no-op at dispatch_tool()'s
    policy check (DELEGATED_TOOLS is checked first) — the real gate for
    any player=None caller is CONFIRM.request()'s own fail-closed
    behavior, regression-tested here directly
  - the real ALTER TABLE migration path for a pre-existing install whose
    permission_policy table predates caller_class

Redirects core.db at a fresh temp sqlite file per test, same pattern as
every other module in this package, so nothing here touches
data/jarvis.db.
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
import threading
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

import core.policy        as policy   # noqa: E402
import core.tool_dispatch as tdisp    # noqa: E402
import core.tool_gate     as gate     # noqa: E402
import core.confirm       as confirm  # noqa: E402


class _PolicyTestCase(unittest.TestCase):
    def setUp(self):
        _use_temp_db()
        # seed_default_policy() is idempotent (per-row existence checks)
        # and called directly here rather than relying on
        # get_policy_level()'s lazy _ensure_seeded() cache, so this
        # test's own fresh temp DB is seeded synchronously regardless of
        # whether some earlier test already triggered seeding against a
        # different DB_PATH in this same process (core/policy.py's
        # _ensure_seeded() cache is itself keyed on DB_PATH now, so it
        # would get this right lazily too — this is just belt-and-suspenders).
        policy.seed_default_policy()


class DesktopUnchangedTest(_PolicyTestCase):
    def test_desktop_default_param_matches_explicit_desktop(self):
        implicit = policy.get_policy_level("file_controller", "delete")
        explicit = policy.get_policy_level("file_controller", "delete", policy.DESKTOP)
        self.assertEqual(implicit, explicit)
        self.assertEqual(implicit, policy.ASK_AND_WAIT)

    def test_desktop_unlisted_tool_still_falls_back_to_ask_and_wait(self):
        self.assertEqual(
            policy.get_policy_level("_totally_unknown_tool", None, policy.DESKTOP),
            policy.ASK_AND_WAIT,
        )

    def test_desktop_auto_allow_rows_unaffected(self):
        self.assertEqual(policy.get_policy_level("weather_report", None), policy.AUTO_ALLOW)
        self.assertEqual(policy.get_policy_level("web_search", None), policy.AUTO_ALLOW)


class ServiceAllowListTest(_PolicyTestCase):
    def test_bugfix_allow_list(self):
        for tool in ("code_helper", "dev_agent", "vision_fix_code", "file_processor", "file_controller"):
            self.assertEqual(
                policy.get_policy_level(tool, None, policy.SERVICE_BUGFIX),
                policy.ASK_AND_WAIT,
                f"{tool} should be ask-and-wait for service:bugfix",
            )

    def test_bugfix_unlisted_tool_is_hard_denied(self):
        # Expected: default-deny, explicit allow-list up — NOT desktop's
        # own ask-and-wait fallback, even though weather_report is
        # auto-allow for desktop.
        self.assertEqual(
            policy.get_policy_level("weather_report", None, policy.SERVICE_BUGFIX),
            policy.HARD_DENY,
        )
        self.assertEqual(
            policy.get_policy_level("send_message", None, policy.SERVICE_BUGFIX),
            policy.HARD_DENY,
        )

    def test_support_allow_list_is_empty_everything_hard_denied(self):
        for tool in tdisp.TOOL_DISPATCH:
            self.assertEqual(
                policy.get_policy_level(tool, None, policy.SERVICE_SUPPORT),
                policy.HARD_DENY,
                f"{tool} should be hard-denied for service:support (empty allow-list)",
            )

    def test_promotions_allow_list_is_empty_everything_hard_denied(self):
        for tool in tdisp.TOOL_DISPATCH:
            self.assertEqual(
                policy.get_policy_level(tool, None, policy.SERVICE_PROMOTIONS),
                policy.HARD_DENY,
                f"{tool} should be hard-denied for service:promotions (empty allow-list)",
            )

    def test_personal_allow_list(self):
        self.assertEqual(policy.get_policy_level("reminder", None, policy.SERVICE_PERSONAL), policy.ASK_AND_WAIT)
        self.assertEqual(policy.get_policy_level("file_processor", None, policy.SERVICE_PERSONAL), policy.ASK_AND_WAIT)

    def test_personal_open_questions_default_to_hard_deny_not_silently_allowed(self):
        # web_search/flight_finder were explicitly left as open questions,
        # not decided either way — the safe default (hard-deny) must hold
        # until someone actually decides, not silently resolve to allowed.
        self.assertEqual(policy.get_policy_level("web_search", None, policy.SERVICE_PERSONAL), policy.HARD_DENY)
        self.assertEqual(policy.get_policy_level("flight_finder", None, policy.SERVICE_PERSONAL), policy.HARD_DENY)

    def test_caller_classes_are_fully_isolated_from_each_other(self):
        # service:bugfix's allow-list must not leak into service:personal
        # or vice versa.
        self.assertEqual(policy.get_policy_level("code_helper", None, policy.SERVICE_PERSONAL), policy.HARD_DENY)
        self.assertEqual(policy.get_policy_level("reminder", None, policy.SERVICE_BUGFIX), policy.HARD_DENY)


class SeedIdempotencyTest(_PolicyTestCase):
    def test_seeding_twice_does_not_duplicate_rows(self):
        conn = db.get_conn()
        before = conn.execute("SELECT COUNT(*) FROM permission_policy").fetchone()[0]
        policy.seed_default_policy()
        after = conn.execute("SELECT COUNT(*) FROM permission_policy").fetchone()[0]
        self.assertEqual(before, after)

    def test_every_default_and_service_row_present_exactly_once(self):
        conn = db.get_conn()
        for tool_name, action, level in policy.DEFAULT_POLICY:
            row = policy._row_exists(conn, tool_name, action, policy.DESKTOP)
            self.assertTrue(row, f"missing desktop row for {tool_name}/{action}")
        for caller_class, rows in policy.SERVICE_POLICY.items():
            for tool_name, action, level in rows:
                self.assertTrue(
                    policy._row_exists(conn, tool_name, action, caller_class),
                    f"missing {caller_class} row for {tool_name}/{action}",
                )


class SeedCacheSurvivesDbSwitchTest(unittest.TestCase):
    """Regression test for a real cross-test-pollution bug found while
    building tests/test_hermes_api.py: _ensure_seeded()'s cache used to
    be a bare bool, set True the first time ANY test seeded ANY DB in
    this process — so a later test that switched core.db.DB_PATH to its
    own fresh temp DB would see the stale "already seeded" flag and skip
    seeding its own, genuinely-empty DB. Concretely broke
    tests/test_verification.py's PostconditionFailureRetryWiringTest —
    not because that file did anything wrong, but because
    tests/test_hermes_api.py happens to sort alphabetically right before
    it under `python -m unittest discover`. Fixed by keying the cache on
    the DB path actually seeded, not a bare bool — this test proves the
    fix directly, switching DB_PATH twice within one process, matching
    exactly what a full `discover` run does across files."""

    def test_ensure_seeded_reseeds_after_switching_to_a_different_db_path(self):
        first_db  = _use_temp_db()
        policy._ensure_seeded()
        count_first = db.get_conn().execute("SELECT COUNT(*) FROM permission_policy").fetchone()[0]
        self.assertGreater(count_first, 0)

        second_db = _use_temp_db()
        self.assertNotEqual(first_db, second_db)
        # Before the fix: _ensure_seeded() would see the stale
        # "already seeded" bool and skip seeding this brand-new,
        # genuinely-empty DB entirely.
        policy._ensure_seeded()
        count_second = db.get_conn().execute("SELECT COUNT(*) FROM permission_policy").fetchone()[0]
        self.assertGreater(count_second, 0, "second DB was never seeded — the stale-cache bug is back")
        self.assertEqual(count_first, count_second)

    def test_get_policy_level_gives_real_answers_after_a_db_switch_mid_process(self):
        """The actual, concrete symptom: an unlisted-tool fallback
        answer for the WRONG reason (missing rows) rather than the real
        one (a genuine policy decision) — this is what turned into
        AgentExecutor blocking for real on TASK_APPROVAL.wait_for_approval()
        with nothing able to ever answer it."""
        _use_temp_db()
        self.assertEqual(policy.get_policy_level("weather_report", None, policy.DESKTOP), policy.AUTO_ALLOW)

        _use_temp_db()  # switch to a second, fresh, unseeded-until-now DB
        # Must still resolve to the real seeded answer, not silently fall
        # through to the generic "unlisted tool" fallback just because
        # seeding was skipped for this DB.
        self.assertEqual(policy.get_policy_level("weather_report", None, policy.DESKTOP), policy.AUTO_ALLOW)


class DispatchToolCallerClassIntegrationTest(_PolicyTestCase):
    """Real dispatch_tool() calls, not just get_policy_level() in
    isolation — confirms the hard-deny fallback for an unlisted
    service:* tool actually refuses at the real chokepoint, mirroring
    tests/test_adversarial_gate.py's A.1 mutation-tested convention."""

    def setUp(self):
        super().setUp()
        self.calls = []
        self._orig_tool = tdisp.TOOL_DISPATCH.get("weather_report")
        tdisp.TOOL_DISPATCH["weather_report"] = self._fake_tool

    def tearDown(self):
        if self._orig_tool is not None:
            tdisp.TOOL_DISPATCH["weather_report"] = self._orig_tool

    def _fake_tool(self, args, player, speak):
        self.calls.append(args)
        return "Sunny."

    def test_service_support_hits_real_hard_deny_for_an_unlisted_tool(self):
        result = gate.dispatch_tool(
            "weather_report", {"city": "Paris"}, player=None, speak=None,
            task_id="t-svc-1", caller_class=policy.SERVICE_SUPPORT,
        )
        self.assertEqual(result, "'weather_report' is not permitted.")
        self.assertEqual(self.calls, [], "hard-denied service call reached the real tool")

    def test_desktop_default_still_auto_allows_the_same_tool(self):
        # Same tool, same args, no caller_class passed — proves the
        # default parameter genuinely preserves desktop's existing
        # behavior side by side with the new service:* behavior above.
        result = gate.dispatch_tool(
            "weather_report", {"city": "Paris"}, player=None, speak=None, task_id="t-desktop-1",
        )
        self.assertEqual(result, "Sunny.")
        self.assertEqual(len(self.calls), 1)


class FileControllerPathScopeTest(_PolicyTestCase):
    def setUp(self):
        super().setUp()
        self.calls = []
        self._orig_tool = tdisp.TOOL_DISPATCH.get("file_controller")
        tdisp.TOOL_DISPATCH["file_controller"] = self._fake_tool
        self._orig_wait = None
        import core.task_approval as ta_mod
        self._ta_mod    = ta_mod
        self._orig_wait = ta_mod.TASK_APPROVAL.wait_for_approval
        ta_mod.TASK_APPROVAL.wait_for_approval = lambda task_id, prompt, timeout=None: (True, "approved")

    def tearDown(self):
        if self._orig_tool is not None:
            tdisp.TOOL_DISPATCH["file_controller"] = self._orig_tool
        self._ta_mod.TASK_APPROVAL.wait_for_approval = self._orig_wait

    def _fake_tool(self, args, player, speak):
        self.calls.append(args)
        return "ok"

    def test_service_bugfix_path_inside_repo_root_is_allowed_through_scope_check(self):
        repo_root = gate.SERVICE_PROJECT_ROOTS[0]
        result = gate.dispatch_tool(
            "file_controller", {"action": "read", "path": str(repo_root), "name": "README.md"},
            player=None, speak=None, task_id="t-scope-1", caller_class=policy.SERVICE_BUGFIX,
        )
        self.assertEqual(result, "ok")
        self.assertEqual(len(self.calls), 1)

    def test_service_bugfix_path_outside_repo_root_is_rejected_even_though_allow_listed(self):
        result = gate.dispatch_tool(
            "file_controller", {"action": "read", "path": str(Path.home() / "Desktop"), "name": "x"},
            player=None, speak=None, task_id="t-scope-2", caller_class=policy.SERVICE_BUGFIX,
        )
        self.assertTrue(result.startswith("Rejected — "), result)
        self.assertIn("outside the allowed project directories", result)
        self.assertEqual(self.calls, [], "path-scope violation reached the real tool")

    def test_service_bugfix_shortcut_keyword_desktop_resolves_and_is_rejected(self):
        # "desktop" is file_controller's own default/shortcut keyword
        # (resolves to the real Desktop folder via _resolve_path) — must
        # be evaluated as that real, resolved path, not the literal
        # string "desktop", and must be rejected since it's outside the
        # project root.
        result = gate.dispatch_tool(
            "file_controller", {"action": "list"},
            player=None, speak=None, task_id="t-scope-3", caller_class=policy.SERVICE_BUGFIX,
        )
        self.assertTrue(result.startswith("Rejected — "), result)
        self.assertEqual(self.calls, [])

    def test_desktop_caller_class_is_never_subject_to_the_project_root_scope_check(self):
        # Desktop keeps its own, broader _SAFE_ROOTS (Path.home()) check
        # inside actions/file_controller.py itself — this dispatch-level
        # scope check must be a complete no-op for desktop, even for a
        # path that would fail the service:* scope check.
        result = gate.dispatch_tool(
            "file_controller", {"action": "read", "path": str(Path.home() / "Desktop"), "name": "x"},
            player=None, speak=None, task_id="t-scope-4",
        )
        self.assertEqual(result, "ok")
        self.assertEqual(len(self.calls), 1)


class DelegatedToolsCaveatTest(_PolicyTestCase):
    """The finding from Step 1.2's own investigation: DELEGATED_TOOLS
    bypasses the permission_policy check entirely at dispatch_tool(),
    regardless of caller_class — regression-tested directly rather than
    just documented in a comment, so a future refactor that accidentally
    fixes the branch order (moving the policy check before the
    DELEGATED_TOOLS check) doesn't silently change dev_agent/
    vision_fix_code's behavior without anyone noticing either way."""

    def test_dev_agent_reaches_invocation_regardless_of_caller_class_hard_deny_default(self):
        calls = []
        orig = tdisp.TOOL_DISPATCH.get("dev_agent")
        tdisp.TOOL_DISPATCH["dev_agent"] = lambda args, player, speak: (calls.append(args), "ran")[1]
        try:
            # service:promotions has an EMPTY allow-list — by the
            # ordinary policy model this would hard-deny dev_agent
            # outright. DELEGATED_TOOLS bypasses that check, so the tool
            # is still invoked (outcome='delegated') — this is the
            # documented caveat, not a bug this test is proving absent.
            result = gate.dispatch_tool(
                "dev_agent", {"description": "x"}, player=None, speak=None,
                task_id="t-deleg-1", caller_class=policy.SERVICE_PROMOTIONS,
            )
        finally:
            if orig is not None:
                tdisp.TOOL_DISPATCH["dev_agent"] = orig
        self.assertEqual(result, "ran")
        self.assertEqual(len(calls), 1)

    def test_but_confirm_request_fails_closed_with_no_player_the_actual_real_gate(self):
        """The real protection: dev_agent's own internal CONFIRM.request()
        calls (pip install, run command) fail closed whenever player is
        None — true for every service:* caller and today's own
        background task queue alike, independent of caller_class or the
        permission_policy table entirely."""
        approved = confirm.CONFIRM.request(None, "I need to pip install requests")
        self.assertFalse(approved)


class AgentTaskGapTest(unittest.TestCase):
    """Step 1.2b: agent_task bypasses dispatch_tool()/permission_policy
    entirely (submitted via main.py's _execute_tool() special case, not
    TOOL_DISPATCH) — investigated and confirmed the narrower case: once
    submitted, agent/task_queue.py's worker runs the plan through the
    normal AgentExecutor.execute() -> _call_tool() -> dispatch_tool()
    path for every step, so only the *submission* decision itself needed
    a separate gate. core.policy.is_agent_task_allowed() is that gate.

    main.py itself is never imported here — it runs heavy PyQt6/audio
    bootstrap as an import-time side effect (see
    core/tool_declarations.py's own docstring on exactly this), which is
    why no other test in this suite imports it either. The wiring at the
    actual call site is instead verified by reading main.py's real
    source text — same static-regression-probe convention this project
    already uses in tests/test_subprocess_timeouts.py's
    NoBareSubprocessRunTest for a check that can't practically be
    exercised at runtime without instantiating the whole desktop app."""

    def test_desktop_is_allowed(self):
        self.assertTrue(policy.is_agent_task_allowed(policy.DESKTOP))

    def test_every_service_class_is_denied(self):
        for caller_class in policy.SERVICE_CALLER_CLASSES:
            self.assertFalse(
                policy.is_agent_task_allowed(caller_class),
                f"agent_task must be denied for {caller_class}",
            )

    def test_unconditional_no_allow_list_exception_even_for_bugfix(self):
        # service:bugfix has the broadest allow-list of the four service
        # classes elsewhere in this file — agent_task must still be
        # denied for it specifically, since the plan calls for no
        # exception at this phase for any class.
        self.assertFalse(policy.is_agent_task_allowed(policy.SERVICE_BUGFIX))

    def test_main_py_agent_task_branch_actually_calls_the_gate(self):
        """Static wiring check: the real source that will actually run,
        not a description of intent. Confirms the gate is called and its
        negative result is actually returned early, inside the
        agent_task branch specifically — not merely present somewhere
        else in the file."""
        main_src = (Path(__file__).resolve().parent.parent / "main.py").read_text(encoding="utf-8")
        branch_start = main_src.find('if name == "agent_task":')
        self.assertNotEqual(branch_start, -1, "agent_task branch not found in main.py")

        # The branch body, up to the next top-level `if name ==` sibling
        # branch (a generous but bounded slice — the real check must
        # appear near the very top of this branch, before any of the
        # branch's other logic runs).
        next_branch = main_src.find('if name ==', branch_start + 10)
        branch_body = main_src[branch_start:next_branch if next_branch != -1 else branch_start + 800]

        self.assertIn("is_agent_task_allowed(caller_class)", branch_body)
        self.assertIn("is not permitted", branch_body)

    def test_main_py_execute_tool_accepts_caller_class_with_a_desktop_default(self):
        """The signature change itself — default-preserving, same
        pattern as dispatch_tool()'s own caller_class parameter."""
        main_src = (Path(__file__).resolve().parent.parent / "main.py").read_text(encoding="utf-8")
        self.assertIn("def _execute_tool(self, name: str, args: dict, caller_class: str = _DESKTOP_CALLER)", main_src)


class PreExistingInstallMigrationTest(unittest.TestCase):
    """Simulates a real pre-phase-1 install: a permission_policy table
    that predates the caller_class column, already holding real seeded
    desktop rows, on disk before core.db.get_conn() ever runs against it
    in this process."""

    def test_existing_desktop_rows_are_backfilled_to_caller_class_desktop(self):
        tmp_dir = Path(tempfile.mkdtemp(prefix="jarvis_test_migration_"))
        db_path = tmp_dir / "legacy.db"

        # Build the OLD-shaped table directly, bypassing core.db entirely
        # — exactly what a real pre-existing on-disk database looks like.
        raw = sqlite3.connect(str(db_path))
        raw.execute("CREATE TABLE permission_policy (tool_name TEXT NOT NULL, action TEXT, level TEXT NOT NULL)")
        raw.execute(
            "INSERT INTO permission_policy (tool_name, action, level) VALUES (?, ?, ?)",
            ("weather_report", None, "auto-allow"),
        )
        raw.commit()
        raw.close()

        db.DB_DIR  = tmp_dir
        db.DB_PATH = db_path
        db._local  = threading.local()

        conn = db.get_conn()  # runs the real ALTER TABLE ADD COLUMN migration
        row = conn.execute(
            "SELECT caller_class FROM permission_policy WHERE tool_name = 'weather_report'"
        ).fetchone()
        self.assertIsNotNone(row, "pre-existing row was lost during migration")
        self.assertEqual(row["caller_class"], "desktop", "pre-existing row was not backfilled to 'desktop'")

    def test_migration_is_idempotent(self):
        tmp_dir = Path(tempfile.mkdtemp(prefix="jarvis_test_migration_"))
        db_path = tmp_dir / "legacy2.db"
        raw = sqlite3.connect(str(db_path))
        raw.execute("CREATE TABLE permission_policy (tool_name TEXT NOT NULL, action TEXT, level TEXT NOT NULL)")
        raw.commit()
        raw.close()

        db.DB_DIR  = tmp_dir
        db.DB_PATH = db_path
        db._local  = threading.local()

        db.get_conn()
        db._local = threading.local()   # force a second first-time-per-thread pass
        conn = db.get_conn()            # must not error on an already-migrated table
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(permission_policy)")}
        self.assertIn("caller_class", cols)


if __name__ == "__main__":
    unittest.main(verbosity=2)
