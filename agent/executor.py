"""
MARK XL — Agent Executor
Replaces google.generativeai with local Ollama via core.llm_client.
"""
import json
import logging
import re
import sys
import threading
import subprocess
import tempfile
import os
import time
from pathlib import Path
from typing import Callable

from agent.planner          import create_plan, replan
from agent.error_handler    import analyze_error, generate_fix, ErrorDecision
from agent.step_references  import resolve_references, UnresolvedReferenceError
from core.llm_client        import call_llm_text
from core.tool_dispatch     import TOOL_DISPATCH
from core.tool_gate         import dispatch_tool
from core.tool_contracts    import get_contract
from core.db                import log_task_event, get_step_outcomes
from config                 import BASE_DIR

_log = logging.getLogger("jarvis.executor")


# ---------------------------------------------------------------------------
# Code generation helper (replaces _run_generated_code with Gemini)
# ---------------------------------------------------------------------------

def _run_generated_code(
    description: str,
    speak:       Callable | None = None,
    task_id:     str | None      = None,
    step_num                     = None,
) -> str:
    if speak:
        speak("Writing custom code for this task, sir.")

    home      = Path.home()
    desktop   = home / "Desktop"
    downloads = home / "Downloads"
    documents = home / "Documents"

    if not desktop.exists():
        try:
            import winreg
            key     = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders",
            )
            desktop = Path(winreg.QueryValueEx(key, "Desktop")[0])
        except Exception:
            pass

    system = (
        "You are an expert Python developer. "
        "Write clean, complete, working Python code. "
        "Use standard library + common packages. "
        "Install missing packages with subprocess + pip if needed. "
        "Return ONLY the Python code. No explanation, no markdown, no backticks.\n\n"
        f"SYSTEM PATHS:\n"
        f"  Desktop   = r'{desktop}'\n"
        f"  Downloads = r'{downloads}'\n"
        f"  Documents = r'{documents}'\n"
        f"  Home      = r'{home}'\n"
    )
    prompt = f"Write Python code to accomplish this task:\n\n{description}"

    try:
        code = call_llm_text(
            prompt, system=system,
            task_id=task_id, step_num=step_num, purpose="generate code",
        )
        code = re.sub(r"```(?:python)?", "", code).strip().rstrip("`").strip()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name

        _log.debug("Running generated code: %s", tmp_path)

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True,
            timeout=120, cwd=str(Path.home()),
        )

        try:
            os.unlink(tmp_path)
        except Exception:
            pass

        output = result.stdout.strip()
        error  = result.stderr.strip()

        if result.returncode == 0 and output:
            return output
        elif result.returncode == 0:
            return "Task completed successfully."
        elif error:
            raise RuntimeError(f"Code error: {error[:400]}")
        return "Completed."

    except subprocess.TimeoutExpired:
        raise RuntimeError("Generated code timed out after 120 seconds.")
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Generated code failed: {e}")


# ---------------------------------------------------------------------------
# Context injection
# ---------------------------------------------------------------------------

def _detect_language(text: str, task_id: str | None = None, step_num=None) -> str:
    try:
        return call_llm_text(
            f"What language is this text written in? "
            f"Reply with ONLY the language name in English (e.g. Turkish, English, French).\n\n"
            f"Text: {text[:200]}",
            task_id=task_id, step_num=step_num, purpose="detect language",
        ).strip()
    except Exception:
        return "English"


def _translate_to_goal_language(content: str, goal: str, task_id: str | None = None, step_num=None) -> str:
    if not goal:
        return content
    try:
        target_lang = _detect_language(goal, task_id=task_id, step_num=step_num)
        _log.debug("Translating to: %s", target_lang)
        prompt = (
            f"You are a professional translator. "
            f"Translate the following text into {target_lang}.\n"
            f"IMPORTANT:\n"
            f"- Translate EVERYTHING, leave nothing in English\n"
            f"- Keep all facts, numbers, and data intact\n"
            f"- Keep the structure and formatting\n"
            f"- Output ONLY the translated text, nothing else\n\n"
            f"Text to translate:\n{content[:4000]}"
        )
        translated = call_llm_text(prompt, task_id=task_id, step_num=step_num, purpose="translate")
        _log.debug("Translation done (%s)", target_lang)
        return translated
    except Exception as e:
        _log.warning("Translation failed: %s", e)
        return content


def _inject_context(
    params: dict, tool: str, step_results: dict, goal: str = "",
    task_id: str | None = None, step_num=None,
) -> dict:
    if not step_results:
        return params
    params = dict(params)
    if tool == "file_controller" and params.get("action") in ("write", "create_file"):
        content = params.get("content", "")
        if not content or len(content) < 50:
            all_results = [
                v for v in step_results.values()
                if v and len(v) > 100 and v not in ("Done.", "Completed.")
            ]
            if all_results:
                combined   = "\n\n---\n\n".join(all_results)
                translated = _translate_to_goal_language(combined, goal, task_id=task_id, step_num=step_num)
                params["content"] = translated
                _log.debug("Injected + translated content")
    return params


# ---------------------------------------------------------------------------
# Tool routing
# ---------------------------------------------------------------------------

# code_helper's "run" action returns this when generate_fix()'s fixed_step
# never included a file_path (only raw "code") — _run_action() short-
# circuits before any permission gate is even reached. Distinct from an
# approval-not-granted result (see _approval_outcome below), which is
# detected by content rather than an exact marker since dispatch_tool's
# two cancellation messages ("timed out waiting for approval" / "was not
# approved") need to stay distinguishable, not collapsed into one marker.
_RECOVERY_NOT_EXECUTED_MARKERS = (
    "Please provide a file path to run",
)


def _approval_outcome(result) -> str | None:
    """If `result` is one of core/tool_gate.py's ask-and-wait cancellation
    strings, returns 'timeout' or 'denied'; else None. This executor
    always calls dispatch_tool with player=None, so an ask-and-wait tool
    blocks on core/task_approval.py's TASK_APPROVAL (no live player to
    answer synchronously) — up to TASK_APPROVAL.DEFAULT_TIMEOUT waiting for
    a human to call approve_task, then this fires on timeout just as it
    would on an explicit deny.

    Callers must treat either outcome as a real, terminal step failure —
    dispatch_tool returns a plain string here (not an exception, matching
    every other action module's error-signaling convention in this
    codebase), so nothing else makes it fail; skipping this check would
    silently log the step as 'done'."""
    if not isinstance(result, str):
        return None
    if "timed out waiting for approval" in result:
        return "timeout"
    if "was not approved" in result:
        return "denied"
    return None


def _short_circuit_reason(result, tool: str | None = None) -> str | None:
    """Classifies a string result that must fail the step immediately —
    never retried, never routed through analyze_error's RETRY branch —
    because retrying wouldn't help (a bad permission decision, invalid
    parameters, or a ${step_N.output} reference that has nothing to
    resolve to won't change on a blind re-run) or could be actively
    harmful: an execution-timeout's underlying call was *abandoned*, not
    cancelled (see core/tool_gate.py's _invoke_with_timeout) — its daemon
    thread may still be running, so an immediate retry risks a second
    concurrent execution of a tool that was deemed non-idempotent enough
    to time-box in the first place.

    Returns a short reason tag ('approval_timeout' | 'approval_denied' |
    'validation_failed' | 'execution_timeout' | 'unresolved_reference' |
    'hard_denied') or None if `result` is an ordinary tool result that
    should proceed normally. 'unresolved_reference' is defense in depth,
    not the expected path — agent/planner.py's _validate_and_fix_plan
    already rejects any self/forward/nonexistent step reference before a
    plan is ever handed to the executor; the only way this fires in
    practice is a referenced step being legitimately skipped at runtime,
    which can't be caught at plan-creation time.

    'hard_denied' needs `tool` (the exact tool name the caller just
    invoked) to recognize core/tool_gate.py's hard-deny refusal, since
    that string embeds the tool's own name and nothing else distinctive —
    an exact-string match against the specific tool just called is both
    simpler and strictly more precise than a generic substring match
    would be. Same non-retryable reasoning as approval_denied: hard-deny
    is unconditional by design, so no rephrasing/retry/replan of the same
    tool call can ever turn a denial into an allow."""
    outcome = _approval_outcome(result)
    if outcome:
        return f"approval_{outcome}"
    if isinstance(result, str):
        if result.startswith("Rejected — "):
            return "validation_failed"
        if "did not complete within" in result and "was abandoned" in result:
            return "execution_timeout"
        if result.startswith("Unresolved — "):
            return "unresolved_reference"
        if tool and result == f"'{tool}' is not permitted.":
            return "hard_denied"
    return None


def _is_postcondition_failure(result) -> bool:
    """True if `result` is one of core/tool_gate.py's postcondition-check
    failure strings (see core/postconditions.py) — the tool call itself
    completed and returned a normal-looking success string, but an
    independent check (a file that should now exist, an OS scheduler
    entry that should now be registered, ...) found no evidence the
    claimed effect actually happened.

    Deliberately NOT part of _short_circuit_reason()'s vocabulary: those
    four reasons are all unconditionally non-retryable for specific,
    provable reasons (an approval decision won't change, invalid
    parameters need different values, a timed-out call's thread may still
    be running, a reference has nothing to resolve to). A postcondition
    failure doesn't share that property — a transient filesystem hiccup
    could legitimately succeed on retry — so instead of a hard
    short-circuit, this is detected and re-raised as a real exception at
    the call site below, feeding the *existing* except-Exception ->
    analyze_error() -> _maybe_downgrade_retry() path, exactly like any
    other runtime failure: retried, skipped, replanned, or aborted based
    on context and the tool's own contract.retryable, not unconditionally
    terminal."""
    return isinstance(result, str) and result.startswith("Postcondition unmet — ")


def _maybe_downgrade_retry(tool: str, decision: ErrorDecision, task_id: str | None) -> ErrorDecision:
    """A genuinely caught exception (unlike a short-circuited timeout —
    see _short_circuit_reason) means the previous attempt has definitely
    finished, so retrying isn't itself unsafe the way it is there. But if
    the tool's own contract says re-running the exact same call isn't
    idempotency-safe, a RETRY decision from analyze_error is downgraded to
    REPLAN (try a different approach) rather than honored as-is."""
    if decision == ErrorDecision.RETRY and not get_contract(tool).retryable:
        _log.info(
            "Tool '%s' is not retryable per its contract — treating RETRY as REPLAN.",
            tool, extra={"task_id": task_id},
        )
        return ErrorDecision.REPLAN
    return decision


def _call_tool(
    tool: str, parameters: dict, speak: Callable | None,
    task_id: str | None = None, step_num=None,
    submitted_interactively: bool = True,
) -> str:
    if tool == "generated_code":
        description = parameters.get("description", "")
        if not description:
            raise ValueError("generated_code requires a 'description' parameter.")
        return _run_generated_code(description, speak=speak, task_id=task_id, step_num=step_num)

    elif tool in TOOL_DISPATCH:
        # player=None: this executor runs without a live UI/session — the
        # shared wrappers in core/tool_dispatch.py already handle that
        # (e.g. file_processor's current_file lookup no-ops when player
        # has no such attribute). Policy evaluation + gating happens in
        # dispatch_tool(), not here — see core/tool_gate.py.
        return dispatch_tool(
            tool, parameters, None, speak,
            task_id=task_id, submitted_interactively=submitted_interactively,
        )

    else:
        raise ValueError(
            f"Unknown tool '{tool}' — no such tool exists. "
            f"Available tools: {sorted(TOOL_DISPATCH.keys())}"
        )


# ---------------------------------------------------------------------------
# Task summary — sourced from get_step_outcomes(), not step descriptions
# ---------------------------------------------------------------------------

def _describe_outcome(o: dict) -> str:
    """One task_events-derived outcome -> a short human-readable phrase.
    Shared by both the deterministic fallback and the LLM prompt's
    structured input, so the two can't describe the same step
    differently."""
    desc = o["description"] or o["tool"] or f"step {o['step_num']}"
    if o["status"] == "done":
        if o["detail"].startswith("recovered via fix"):
            return f"{desc} (accomplished via an alternative approach after the first attempt failed)"
        return desc
    if o["status"] == "skipped":
        return f"{desc} — SKIPPED ({o['detail'] or 'no reason recorded'})"
    if o["status"] == "failed":
        return f"{desc} — FAILED ({o['detail'] or 'no reason recorded'})"
    return f"{desc} — outcome unknown"


def _build_fallback_summary(goal: str, done: list, skipped: list, failed: list) -> str:
    """No LLM call — used when call_llm_text fails/returns nothing, or
    when there's no task_id to look outcomes up for at all. Must be
    accurate on its own; it's reachable independently of the LLM path."""
    if not skipped and not failed:
        return f"All done, sir. Completed {len(done)} step(s) for: {goal[:60]}."
    total = len(done) + len(skipped) + len(failed)
    parts = [f"Completed {len(done)} of {total} step(s) for: {goal[:60]}, sir."]
    if skipped:
        parts.append("Skipped: " + "; ".join(_describe_outcome(o) for o in skipped) + ".")
    if failed:
        parts.append("Failed: " + "; ".join(_describe_outcome(o) for o in failed) + ".")
    return " ".join(parts)


def _build_summary_prompt(goal: str, done: list, skipped: list, failed: list) -> str:
    lines = (
        [f"- DONE: {_describe_outcome(o)}" for o in done]
        + [f"- SKIPPED: {_describe_outcome(o)}" for o in skipped]
        + [f"- FAILED: {_describe_outcome(o)}" for o in failed]
    )
    steps_str = "\n".join(lines) if lines else "(no steps recorded)"
    return (
        f'User goal: "{goal}"\n'
        f"Step outcomes (accurate — not every step necessarily succeeded):\n{steps_str}\n\n"
        "Write a single natural sentence summarising what actually happened. "
        "Accurately reflect what was done, what was skipped, and what failed — "
        "do not claim a skipped or failed step was completed. "
        "Address the user as 'sir'."
    )


# ---------------------------------------------------------------------------
# AgentExecutor
# ---------------------------------------------------------------------------

class AgentExecutor:

    MAX_REPLAN_ATTEMPTS = 2

    def execute(
        self,
        goal:        str,
        speak:       Callable | None        = None,
        cancel_flag: threading.Event | None = None,
        task_id:     str | None             = None,
        submitted_interactively: bool       = True,
        caller_class: str                   = "desktop",
    ) -> str:
        """caller_class (headless-extraction phase, Step 1.3): who
        submitted this task — one of core/policy.py's CALLER_CLASSES,
        defaulting to 'desktop' so every pre-existing caller of execute()
        is unaffected. Threaded into every task_events row this method
        logs via the local _log_event() wrapper below, so the resulting
        audit trail is attributable regardless of which caller_class
        submitted the task (see tests/test_caller_attribution.py's
        Step-1.6-equivalent routing test)."""
        _log.info("Goal: %s", goal, extra={"task_id": task_id, "caller_class": caller_class})

        def _log_event(step_num, tool, desc, status, detail="", duration_ms=None):
            log_task_event(
                task_id, step_num, tool, desc, status, detail, duration_ms=duration_ms,
                caller_id=caller_class, caller_class=caller_class, triggered_by=caller_class,
            )

        replan_attempts = 0
        completed_steps: list = []
        plan = create_plan(goal, task_id=task_id)

        while True:
            steps = plan.get("steps", [])
            if not steps:
                msg = "I couldn't create a valid plan for this task, sir."
                if speak: speak(msg)
                return msg

            # Fresh per plan/replan attempt, not just once outside this
            # loop: replan() re-numbers steps from 1 again in the revised
            # plan, which has no defined relationship to the previous
            # (superseded) plan's numbering. Carrying old entries forward
            # would let a stale step_results[1] from a discarded attempt
            # silently resolve into an unrelated step's ${step_1.output}
            # reference in the new plan. completed_steps is NOT reset
            # here — replan()'s prompt uses it to tell the LLM what's
            # already done so it doesn't repeat those steps.
            step_results: dict = {}

            success      = True
            failed_step  = None
            failed_error = ""

            for step in steps:
                if cancel_flag and cancel_flag.is_set():
                    if speak: speak("Task cancelled, sir.")
                    return "Task cancelled."

                step_num = step.get("step", "?")
                tool     = step.get("tool", "generated_code")
                desc     = step.get("description", "")
                params   = step.get("parameters", {})
                params   = _inject_context(params, tool, step_results, goal=goal, task_id=task_id, step_num=step_num)

                step_start = time.monotonic()
                attempt    = 1
                step_ok    = False

                # ${step_N.output} substitution — resolved once here, before
                # "started" is even logged and before _call_tool()/
                # dispatch_tool()'s input validation runs, so validation
                # checks the actual resulting value rather than the literal
                # "${step_N.output}" template text. Not inside the retry
                # loop below: step_results can't change mid-retry (no other
                # step runs concurrently), so re-resolving on every attempt
                # would be redundant — and a resolution failure is never
                # retryable in the first place (see _short_circuit_reason).
                try:
                    params = resolve_references(params, step_results)
                except UnresolvedReferenceError as ref_err:
                    result = (
                        f"Unresolved — step {step_num} references step "
                        f"{ref_err.step_num}'s output, which is not "
                        f"available (skipped, failed, or never ran)."
                    )
                    short_circuit = _short_circuit_reason(result, tool)
                    detail = f"{short_circuit}: {result}"
                    _log_event(
                        step_num, tool, desc, "failed", detail,
                        duration_ms=int((time.monotonic() - step_start) * 1000),
                    )
                    failed_step  = step
                    failed_error = detail
                    success      = False
                    break

                _log_event(step_num, tool, desc, "started")

                while attempt <= 3:
                    if cancel_flag and cancel_flag.is_set():
                        break
                    try:
                        result = _call_tool(
                            tool, params, speak, task_id=task_id, step_num=step_num,
                            submitted_interactively=submitted_interactively,
                        )

                        short_circuit = _short_circuit_reason(result, tool)
                        if short_circuit:
                            # None of these are transient or fixable by
                            # blindly re-running the same call — see
                            # _short_circuit_reason's docstring for why
                            # each of the five reasons ends here instead
                            # of going through analyze_error's RETRY
                            # branch or the REPLAN-fix cascade (which
                            # itself retargets code_helper — also gated —
                            # and could stack another long wait for no
                            # benefit). Fail this step immediately with a
                            # distinct, greppable reason and let the
                            # *task-level* replanner (outside this step,
                            # further down) decide whether a different
                            # overall approach is worth trying.
                            detail = f"{short_circuit}: {result}"
                            _log_event(
                                step_num, tool, desc, "failed", detail,
                                duration_ms=int((time.monotonic() - step_start) * 1000),
                            )
                            failed_step  = step
                            failed_error = detail
                            success      = False
                            break

                        if _is_postcondition_failure(result):
                            # Unlike the four short_circuit reasons above,
                            # this genuinely might be transient — raise it
                            # as a real exception so it flows into the
                            # except-Exception block below and gets the
                            # same retry/skip/replan/abort treatment (and
                            # the same contract.retryable gate) as any
                            # other runtime failure, instead of always
                            # failing the step outright. See
                            # _is_postcondition_failure's docstring.
                            raise RuntimeError(result)

                        step_results[step_num] = result
                        completed_steps.append(step)
                        _log_event(
                            step_num, tool, desc, "done", str(result)[:200],
                            duration_ms=int((time.monotonic() - step_start) * 1000),
                        )
                        step_ok = True
                        break

                    except Exception as e:
                        error_msg = str(e)
                        # Non-terminal — a retry/skip/replan-fix may still turn
                        # this step into a success. Kept distinct from the
                        # terminal 'failed' status below so a tool's true
                        # failure rate isn't inflated by every transient retry.
                        _log_event(step_num, tool, desc, "attempt_failed", f"attempt {attempt}: {error_msg}")

                        recovery = analyze_error(step, error_msg, attempt=attempt, task_id=task_id)
                        decision = recovery["decision"]
                        user_msg = recovery.get("user_message", "")

                        # A genuinely caught exception (unlike a timeout,
                        # this means the previous attempt has definitely
                        # finished — no thread left running) can still be
                        # gated by the tool's own idempotency.
                        decision = _maybe_downgrade_retry(tool, decision, task_id)

                        if speak and user_msg:
                            speak(user_msg)

                        if decision == ErrorDecision.RETRY:
                            _log_event(step_num, tool, desc, "retried", f"attempt {attempt} -> {attempt + 1}")
                            attempt += 1
                            time.sleep(2)
                            continue

                        elif decision == ErrorDecision.SKIP:
                            _log_event(
                                step_num, tool, desc, "skipped", "skipped (non-critical)",
                                duration_ms=int((time.monotonic() - step_start) * 1000),
                            )
                            completed_steps.append(step)
                            step_ok = True
                            break

                        elif decision == ErrorDecision.ABORT:
                            msg = f"Task aborted, sir. {recovery.get('reason', '')}"
                            _log_event(
                                step_num, tool, desc, "failed", f"aborted: {recovery.get('reason', '')}",
                                duration_ms=int((time.monotonic() - step_start) * 1000),
                            )
                            if speak: speak(msg)
                            return msg

                        else:  # REPLAN
                            fix_suggestion = recovery.get("fix_suggestion", "")
                            if fix_suggestion and tool != "generated_code":
                                try:
                                    fixed_step = generate_fix(step, error_msg, fix_suggestion, task_id=task_id)
                                    if speak: speak("Trying an alternative approach, sir.")
                                    res = _call_tool(
                                        fixed_step["tool"],
                                        fixed_step["parameters"],
                                        speak,
                                        task_id=task_id, step_num=step_num,
                                        submitted_interactively=submitted_interactively,
                                    )
                                    res_short_circuit    = _short_circuit_reason(res, fixed_step.get("tool"))
                                    res_no_file_path     = isinstance(res, str) and any(
                                        marker in res for marker in _RECOVERY_NOT_EXECUTED_MARKERS
                                    )
                                    # A postcondition failure on the recovery attempt itself is
                                    # treated the same way as the other two — "this fix didn't
                                    # actually take" — rather than re-raised into another round of
                                    # analyze_error from *within* an already-in-progress REPLAN fix
                                    # attempt, which isn't how this recovery path is structured.
                                    res_postcondition_failed = _is_postcondition_failure(res)
                                    if res_short_circuit or res_no_file_path or res_postcondition_failed:
                                        if res_short_circuit:
                                            reason = res_short_circuit
                                            _log.info(
                                                "Recovery via code_helper did not complete (%s; no live "
                                                "player to confirm synchronously, or it was rejected/timed "
                                                "out). Autonomous code-fix recovery cannot complete in "
                                                "this context.",
                                                res_short_circuit, extra={"task_id": task_id},
                                            )
                                        elif res_postcondition_failed:
                                            reason = "postcondition_failed"
                                            _log.info(
                                                "Recovery via code_helper completed but its postcondition "
                                                "check failed: %s", res, extra={"task_id": task_id},
                                            )
                                        else:
                                            reason = "missing_file_path"
                                            _log.info(
                                                "Recovery via code_helper did not run — no file_path in "
                                                "the generated fix step.",
                                                extra={"task_id": task_id},
                                            )
                                        _log_event(
                                            step_num, tool, desc, "failed",
                                            f"recovery fix did not execute ({reason}): {res}",
                                            duration_ms=int((time.monotonic() - step_start) * 1000),
                                        )
                                    else:
                                        _log_event(
                                            step_num, fixed_step.get("tool"), desc, "done",
                                            f"recovered via fix: {str(res)[:150]}",
                                            duration_ms=int((time.monotonic() - step_start) * 1000),
                                        )
                                        step_results[step_num] = res
                                        completed_steps.append(step)
                                        step_ok = True
                                        break
                                except Exception as fix_err:
                                    _log_event(
                                        step_num, tool, desc, "failed", f"fix generation failed: {fix_err}",
                                        duration_ms=int((time.monotonic() - step_start) * 1000),
                                    )

                            failed_step  = step
                            failed_error = error_msg
                            success      = False
                            break

                if not step_ok and not failed_step:
                    failed_step  = step
                    failed_error = "Max retries exceeded"
                    success      = False
                    _log_event(
                        step_num, tool, desc, "failed", "max retries exceeded",
                        duration_ms=int((time.monotonic() - step_start) * 1000),
                    )

                if not success:
                    break

            if success:
                return self._summarize(goal, speak, task_id=task_id)

            if replan_attempts >= self.MAX_REPLAN_ATTEMPTS:
                msg = f"Task failed after {replan_attempts} replan attempts, sir."
                if speak: speak(msg)
                return msg

            if speak: speak("Adjusting my approach, sir.")
            _log_event(
                failed_step.get("step") if failed_step else None,
                failed_step.get("tool") if failed_step else None, goal[:200],
                "replanned", f"after step failure: {failed_error[:150]}",
            )
            replan_attempts += 1
            plan = replan(goal, completed_steps, failed_step, failed_error, task_id=task_id)

    def _summarize(self, goal: str, speak: Callable | None, task_id: str | None = None) -> str:
        """Sourced from task_events (get_step_outcomes), not from a
        parallel completed_steps list — completed_steps conflated a
        genuinely 'done' step with one that was merely 'skipped'
        (ErrorDecision.SKIP appends to it exactly like a real success
        does), so both the LLM-fed prompt and the deterministic fallback
        below used to describe skipped work as accomplished. Both paths
        are fixed here, not just the LLM one — the fallback has the
        identical bug and is independently reachable if the LLM call
        fails."""
        outcomes = get_step_outcomes(task_id)
        done     = [o for o in outcomes if o["status"] == "done"]
        skipped  = [o for o in outcomes if o["status"] == "skipped"]
        # A 'failed' segment shouldn't normally reach _summarize() — a
        # failed step aborts the whole plan before success=True is ever
        # set — but included for robustness rather than assuming it can't
        # happen (e.g. a future change to the success/failure wiring).
        failed   = [o for o in outcomes if o["status"] == "failed"]

        fallback = _build_fallback_summary(goal, done, skipped, failed)

        if not outcomes:
            # No task_id, or nothing logged for it — fall back to the
            # old, coarser wording rather than claiming a precision the
            # data doesn't support.
            if speak: speak(fallback)
            return fallback

        prompt = _build_summary_prompt(goal, done, skipped, failed)
        try:
            summary = call_llm_text(prompt, task_id=task_id, purpose="summarize task")
            if summary:
                if speak: speak(summary)
                return summary
        except Exception:
            pass
        if speak: speak(fallback)
        return fallback
