"""
tests/test_approval_routing.py
================================
Headless-extraction phase, Step 1.6: a service:bugfix-triggered task
whose step lands on ask-and-wait resolves through the existing single
human approval channel (core/task_approval.py's TASK_APPROVAL), with
task_events.triggered_by correctly recording service:bugfix.

A real, non-obvious interaction with Step 1.2b found while writing this:
core.policy.is_agent_task_allowed() hard-denies agent_task submission for
EVERY service:* class, service:bugfix included — so a
service:bugfix-triggered task cannot actually be created via POST /tasks
or main.py's _execute_tool() at all today (confirmed directly:
is_agent_task_allowed('service:bugfix') is False). That gate lives only
at the submission entry points (api/hermes_app.py's submit_task,
main.py's agent_task branch) — agent/task_queue.py's TaskQueue.submit()
itself enforces nothing and trusts its caller to have already gated
appropriately, by design. Step 1.2b and Step 1.6 are testing two
genuinely separate layers — "should this caller be allowed to submit a
task at all" vs. "once a task exists under this caller_class, does its
ask-and-wait routing work and get attributed correctly" — so this file
submits directly through TaskQueue.submit() (the same real machinery a
future named exception to the Step 1.2b gate would use — see
core/policy.py's SERVICE_POLICY docstring) rather than through the
HTTP-layer gate that would otherwise make this scenario untestable
without contradicting Step 1.2b's own, already-verified restriction.

Step 1.6b (see ApprovalResolutionAuthTest below): a related finding from
this file's own first pass was that POST /approvals/{id} had no
caller_class restriction at all — ANY authenticated caller (including
service:promotions, whose tool allow-list is empty) could resolve ANY
pending approval, including one belonging to a different caller's task.
Fixed there: only 'desktop' may resolve an approval, regardless of which
caller_class the underlying task was triggered by, enforced as its own
router-level dependency (api/hermes_app.py's
approval_resolution_router/_require_desktop_caller) rather than a
per-route check.

Safety note, worth stating explicitly because it's the one way this
specific test class can go wrong in a way that's easy to miss locally
and expensive in CI: core/task_approval.py's wait_for_approval() blocks
the real background TaskQueue worker thread for up to DEFAULT_TIMEOUT
(1800s / 30 minutes) if nothing ever answers it. A test that submits a
task, lets it land on ask-and-wait, and then just polls
get_status()/get_queue().get_status() for a terminal state with its own
short timeout does NOT avoid this — the test method returns/fails
quickly, but the real worker thread keeps blocking for the real 30
minutes regardless, in the background, for the rest of the process (this
exact failure mode produced a genuinely stuck ~30-minute background shell
earlier in this phase — see git history). Every test below that reaches
ask-and-wait therefore drives the approval to a real resolution itself
(poll for it to land in the real channel, then answer it, in a `finally`
so it happens even if an assertion above it fails) before ever waiting
for the task's terminal status.

Reuses tests/test_hermes_api.py's HermesApiTestCase for the same
temp-DB/temp-tokens/rate-limiter/create_plan-stub setup — this file is
about the approval-routing mechanism specifically, not a reason to
reinvent that harness.
"""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.db as db
import core.policy        as policy
import core.tool_dispatch as tdisp
import core.task_approval as ta_mod
import agent.executor      as executor
from tests.test_hermes_api import HermesApiTestCase, _use_temp_db  # noqa: E402


class ApprovalRoutingTest(HermesApiTestCase):
    def _submit_ask_and_wait_step(self, caller_class: str, tool: str, parameters: dict, goal: str = "do a gated thing"):
        """Submits a task whose single step is `tool` (must resolve to
        ask-and-wait for `caller_class`) directly through the real
        agent.task_queue.TaskQueue.submit() — see the module docstring
        for why this deliberately does not go through POST /tasks: that
        endpoint's is_agent_task_allowed() gate (Step 1.2b) hard-denies
        every service:* class unconditionally, which would make a
        service:bugfix routing scenario untestable through that specific
        entry point without contradicting an already-verified,
        intentional restriction. TaskQueue.submit() itself is real,
        unmocked machinery either way. Returns task_id. Caller is
        responsible for driving the resulting approval to resolution —
        see the module docstring."""
        executor.create_plan = lambda g, task_id=None: {
            "steps": [{"step": 1, "tool": tool, "description": goal, "parameters": parameters}],
        }
        from agent.task_queue import get_queue
        return get_queue().submit(goal=goal, submitted_interactively=False, caller_class=caller_class)

    def _find_pending_approval(self, task_id: str, timeout: float = 5.0) -> dict | None:
        """Bounded poll for the step to actually reach the real
        ask-and-wait channel — this, not a terminal-status wait, is what
        proves routing. Short timeout: if the step never lands here,
        that's a real routing failure this test should report promptly,
        not hang trying to find out."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            for row in ta_mod.TASK_APPROVAL.list_pending():
                if row["task_id"] == task_id:
                    return row
            time.sleep(0.02)
        return None

    def test_service_bugfix_ask_and_wait_step_reaches_the_real_approval_channel(self):
        """The core routing claim: a service:bugfix step that lands on
        ask-and-wait surfaces through the exact same TASK_APPROVAL
        channel a desktop-submitted task would — not a separate,
        caller-class-forked mechanism."""
        orig_tool = tdisp.TOOL_DISPATCH.get("code_helper")
        # code_helper is allow-listed (ask-and-wait) for service:bugfix —
        # stubbed so this test is never accidentally satisfied by the
        # real tool actually running before/without approval.
        tdisp.TOOL_DISPATCH["code_helper"] = lambda args, player, speak: "SHOULD NEVER RUN WITHOUT APPROVAL"

        task_id  = self._submit_ask_and_wait_step(
            policy.SERVICE_BUGFIX, "code_helper", {"action": "run", "file_path": "x.py"},
        )
        approval = self._find_pending_approval(task_id)

        try:
            self.assertIsNotNone(approval, "step never reached the real ask-and-wait approval channel")

            # Expected per the plan: "task_events.triggered_by correctly
            # recording service:bugfix" — checked here, BEFORE resolving
            # the approval, against the step's own 'started' row (logged
            # before dispatch_tool's ask-and-wait branch ever blocks) —
            # this is the routing claim itself, not just the eventual
            # outcome.
            started_row = db.get_conn().execute(
                "SELECT triggered_by, caller_class FROM task_events "
                "WHERE task_id = ? AND status = 'started'",
                (task_id,),
            ).fetchone()
            self.assertIsNotNone(started_row, "no 'started' task_events row was logged before the approval blocked")
            self.assertEqual(started_row["triggered_by"], policy.SERVICE_BUGFIX)
            self.assertEqual(started_row["caller_class"], policy.SERVICE_BUGFIX)

            self.assertEqual(tdisp.TOOL_DISPATCH["code_helper"](None, None, None), "SHOULD NEVER RUN WITHOUT APPROVAL")
        finally:
            # Drive it to resolution regardless of whether the assertions
            # above passed — an unresolved approval leaves the real
            # background worker thread blocked on the real 30-minute
            # default timeout (see module docstring).
            if approval is not None:
                ta_mod.TASK_APPROVAL.answer(approval["approval_id"], approve=True)
            if orig_tool is not None:
                tdisp.TOOL_DISPATCH["code_helper"] = orig_tool

        status = self._wait_for_task(task_id, timeout=5.0)
        self.assertEqual(status["status"], "completed", status)

        done_row = db.get_conn().execute(
            "SELECT triggered_by FROM task_events WHERE task_id = ? AND status = 'done'",
            (task_id,),
        ).fetchone()
        self.assertIsNotNone(done_row)
        self.assertEqual(done_row["triggered_by"], policy.SERVICE_BUGFIX)

    def test_desktop_ask_and_wait_step_reaches_the_same_channel(self):
        """Contrast case: desktop's own ask-and-wait routing is
        unchanged, through the identical channel — proves this isn't a
        new, service:*-only path bolted on beside the real one."""
        orig_tool = tdisp.TOOL_DISPATCH.get("code_helper")
        tdisp.TOOL_DISPATCH["code_helper"] = lambda args, player, speak: "ran"

        task_id  = self._submit_ask_and_wait_step(policy.DESKTOP, "code_helper", {"action": "run", "file_path": "x.py"})
        approval = self._find_pending_approval(task_id)
        try:
            self.assertIsNotNone(approval, "desktop's own ask-and-wait step never reached the real approval channel")
            started_row = db.get_conn().execute(
                "SELECT triggered_by FROM task_events WHERE task_id = ? AND status = 'started'", (task_id,),
            ).fetchone()
            self.assertEqual(started_row["triggered_by"], policy.DESKTOP)
        finally:
            if approval is not None:
                ta_mod.TASK_APPROVAL.answer(approval["approval_id"], approve=True)
            if orig_tool is not None:
                tdisp.TOOL_DISPATCH["code_helper"] = orig_tool

        status = self._wait_for_task(task_id, timeout=5.0)
        self.assertEqual(status["status"], "completed", status)

    def test_denial_is_routed_and_recorded_the_same_way(self):
        """The approval channel resolving to 'denied' is just as much a
        real resolution as 'approved' — same routing claim, opposite
        answer, still must not hang the worker thread.

        A denial does NOT immediately fail the task — _short_circuit_reason
        classifies it 'approval_denied', which (correctly, same as any
        other step failure) falls through to AgentExecutor's real
        replan() cascade before the task actually gives up (see D.22's
        adversarial-gate test earlier in this phase for the same
        replan-after-denial shape). replan() is stubbed here to end the
        task immediately rather than let it make a real, slow LLM call —
        found the hard way, the same class of leftover-background-work
        risk as the weather_report/create_plan lessons elsewhere in this
        phase.

        Second finding, real and now fixed (Step 1.6c): AgentExecutor.
        execute() never raises for "ran out of replan attempts" / "no
        valid plan" — it returns a plain string either way — so
        agent/task_queue.py's TaskQueue used to mark the overall Task
        'completed' regardless of whether the goal was actually achieved
        (only an uncaught exception used to produce 'failed' at the Task
        level). Fixed by checking the real, durable outcome
        (core.db.get_step_outcomes(), the same helper _summarize()
        itself trusts) after execute() returns, rather than trusting a
        lack of exception as success — see
        agent/task_queue.py's _last_terminal_failure_detail(). This test
        now asserts the fixed behavior directly."""
        orig_tool   = tdisp.TOOL_DISPATCH.get("code_helper")
        orig_replan = executor.replan
        tdisp.TOOL_DISPATCH["code_helper"] = lambda args, player, speak: "SHOULD NEVER RUN — DENIED"
        executor.replan = lambda goal, completed_steps, failed_step, error, task_id=None: {"steps": []}

        try:
            task_id  = self._submit_ask_and_wait_step(policy.SERVICE_BUGFIX, "code_helper", {"action": "run", "file_path": "x.py"})
            approval = self._find_pending_approval(task_id)
            try:
                self.assertIsNotNone(approval)
            finally:
                if approval is not None:
                    ta_mod.TASK_APPROVAL.answer(approval["approval_id"], approve=False)

            # Expected (Step 1.6c): Task.status must reflect the real
            # denial, not silently read as 'completed'.
            status = self._wait_for_task(task_id, timeout=5.0)
            self.assertEqual(status["status"], "failed", status)
            self.assertIn("approval_denied", status["error"])
        finally:
            if orig_tool is not None:
                tdisp.TOOL_DISPATCH["code_helper"] = orig_tool
            executor.replan = orig_replan

        # The real outcome: the step itself is recorded as failed, with
        # the specific approval_denied reason, correctly attributed.
        step_row = db.get_conn().execute(
            "SELECT status, detail, triggered_by FROM task_events "
            "WHERE task_id = ? AND tool = 'code_helper' AND status = 'failed' "
            "ORDER BY event_id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        self.assertIsNotNone(step_row)
        self.assertEqual(step_row["status"], "failed")
        self.assertIn("approval_denied", step_row["detail"])
        self.assertEqual(step_row["triggered_by"], policy.SERVICE_BUGFIX)

    def test_approvals_rest_endpoint_can_resolve_a_service_bugfix_routed_approval(self):
        """The same routing claim exercised through the real REST
        endpoint (POST /approvals/{id}) rather than calling
        TASK_APPROVAL.answer() directly — the full boundary, not just
        the underlying primitive."""
        orig_tool = tdisp.TOOL_DISPATCH.get("code_helper")
        tdisp.TOOL_DISPATCH["code_helper"] = lambda args, player, speak: "ran"

        task_id  = self._submit_ask_and_wait_step(policy.SERVICE_BUGFIX, "code_helper", {"action": "run", "file_path": "x.py"})
        approval = self._find_pending_approval(task_id)
        desktop_token = hermes_auth_mint(policy.DESKTOP)
        try:
            self.assertIsNotNone(approval)
            r = self.client.post(
                f"/approvals/{approval['approval_id']}", json={"approve": True},
                headers={"Authorization": f"Bearer {desktop_token}"},
            )
            self.assertEqual(r.status_code, 200, r.text)
            approval = None  # resolved via the REST call above — finally must not double-answer
        finally:
            if approval is not None:
                ta_mod.TASK_APPROVAL.answer(approval["approval_id"], approve=True)
            if orig_tool is not None:
                tdisp.TOOL_DISPATCH["code_helper"] = orig_tool

        status = self._wait_for_task(task_id, timeout=5.0)
        self.assertEqual(status["status"], "completed", status)


class ApprovalResolutionAuthTest(ApprovalRoutingTest):
    """Step 1.6b: an approval can only be resolved by the desktop
    caller_class, regardless of which caller_class triggered the
    underlying task. Subclasses ApprovalRoutingTest to reuse its
    _submit_ask_and_wait_step()/_find_pending_approval() helpers rather
    than reinventing them."""

    def test_service_support_cannot_resolve_a_service_bugfix_triggered_approval(self):
        orig_tool = tdisp.TOOL_DISPATCH.get("code_helper")
        tdisp.TOOL_DISPATCH["code_helper"] = lambda args, player, speak: "SHOULD NEVER RUN — WRONG CALLER RESOLVED IT"

        task_id  = self._submit_ask_and_wait_step(policy.SERVICE_BUGFIX, "code_helper", {"action": "run", "file_path": "x.py"})
        approval = self._find_pending_approval(task_id)
        support_token = hermes_auth_mint(policy.SERVICE_SUPPORT)
        try:
            self.assertIsNotNone(approval)

            r = self.client.post(
                f"/approvals/{approval['approval_id']}", json={"approve": True},
                headers={"Authorization": f"Bearer {support_token}"},
            )
            # Expected: rejected — service:support is not the desktop
            # caller, even though it authenticated correctly and even
            # though it isn't the caller_class that triggered this task
            # either way (neither fact should matter — only 'desktop' can
            # ever resolve an approval).
            self.assertEqual(r.status_code, 403, r.text)

            # And the approval must still genuinely be pending — the
            # rejected request must not have silently resolved it anyway.
            still_pending = any(
                p["approval_id"] == approval["approval_id"] for p in ta_mod.TASK_APPROVAL.list_pending()
            )
            self.assertTrue(still_pending, "approval was resolved despite the 403")
        finally:
            if approval is not None:
                ta_mod.TASK_APPROVAL.answer(approval["approval_id"], approve=True)
            if orig_tool is not None:
                tdisp.TOOL_DISPATCH["code_helper"] = orig_tool

        # Cleaned up via the real desktop-equivalent primitive above —
        # the task itself still completes normally once actually resolved.
        status = self._wait_for_task(task_id, timeout=5.0)
        self.assertEqual(status["status"], "completed", status)

    def test_desktop_can_still_resolve_any_approval_regardless_of_triggering_caller(self):
        """Contrast case: the fix must not accidentally scope resolution
        to 'only the same caller_class that submitted the task' — the
        plan is explicit that desktop resolves everything, unconditionally."""
        orig_tool = tdisp.TOOL_DISPATCH.get("code_helper")
        tdisp.TOOL_DISPATCH["code_helper"] = lambda args, player, speak: "ran"

        task_id  = self._submit_ask_and_wait_step(policy.SERVICE_BUGFIX, "code_helper", {"action": "run", "file_path": "x.py"})
        approval = self._find_pending_approval(task_id)
        desktop_token = hermes_auth_mint(policy.DESKTOP)
        try:
            self.assertIsNotNone(approval)
            r = self.client.post(
                f"/approvals/{approval['approval_id']}", json={"approve": True},
                headers={"Authorization": f"Bearer {desktop_token}"},
            )
            self.assertEqual(r.status_code, 200, r.text)
            approval = None
        finally:
            if approval is not None:
                ta_mod.TASK_APPROVAL.answer(approval["approval_id"], approve=True)
            if orig_tool is not None:
                tdisp.TOOL_DISPATCH["code_helper"] = orig_tool

        status = self._wait_for_task(task_id, timeout=5.0)
        self.assertEqual(status["status"], "completed", status)

    def test_every_non_desktop_caller_class_is_rejected(self):
        orig_tool = tdisp.TOOL_DISPATCH.get("code_helper")
        tdisp.TOOL_DISPATCH["code_helper"] = lambda args, player, speak: "SHOULD NEVER RUN"

        task_id  = self._submit_ask_and_wait_step(policy.SERVICE_BUGFIX, "code_helper", {"action": "run", "file_path": "x.py"})
        approval = self._find_pending_approval(task_id)
        try:
            self.assertIsNotNone(approval)
            for caller_class in policy.SERVICE_CALLER_CLASSES:
                token = hermes_auth_mint(caller_class)
                r = self.client.post(
                    f"/approvals/{approval['approval_id']}", json={"approve": True},
                    headers={"Authorization": f"Bearer {token}"},
                )
                self.assertEqual(r.status_code, 403, f"{caller_class} was allowed to resolve an approval: {r.text}")
        finally:
            if approval is not None:
                ta_mod.TASK_APPROVAL.answer(approval["approval_id"], approve=True)
            if orig_tool is not None:
                tdisp.TOOL_DISPATCH["code_helper"] = orig_tool

        status = self._wait_for_task(task_id, timeout=5.0)
        self.assertEqual(status["status"], "completed", status)

    def test_no_route_reachable_without_auth_still_covers_the_new_router(self):
        """Structural check mirroring
        tests/test_hermes_api.py's AuthRejectionTest — the new
        approval_resolution_router must still reject a completely
        unauthenticated request (401), before ever reaching the
        desktop-only check (403) — the two dependencies must be layered
        in the right order, not the desktop check accidentally running
        (and maybe passing/failing oddly) before auth itself."""
        import api.hermes_app as hermes_app
        r = self.client.post("/approvals/1", json={"approve": True})
        self.assertEqual(r.status_code, 401)


def hermes_auth_mint(caller_class: str) -> str:
    import core.hermes_auth as hermes_auth
    return hermes_auth.mint_token(caller_class)


if __name__ == "__main__":
    unittest.main(verbosity=2)
