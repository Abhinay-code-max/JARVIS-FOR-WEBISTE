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

from agent.planner       import create_plan, replan
from agent.error_handler import analyze_error, generate_fix, ErrorDecision
from core.llm_client     import call_llm_text
from core.tool_dispatch  import TOOL_DISPATCH
from core.db             import log_task_event
from config              import BASE_DIR

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

# Strings code_helper's "run" action returns when it did NOT actually
# execute the auto-generated fix code — this executor always calls with
# player=None, so:
#   - generate_fix()'s fixed_step never includes a file_path (only raw
#     "code"), so _run_action() short-circuits with the first marker
#     before CONFIRM is even reached;
#   - if that parameter gap is ever closed, the flow would instead reach
#     core/confirm.py's CONFIRM.request(), which always denies with no
#     live player (core/confirm.py:48) and returns the second marker.
# Either way "recovery ran" is false — treat both as a failed step, not
# a silent success.
_RECOVERY_NOT_EXECUTED_MARKERS = (
    "Please provide a file path to run",
    "Cancelled — did not run",
)


def _call_tool(
    tool: str, parameters: dict, speak: Callable | None,
    task_id: str | None = None, step_num=None,
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
        # has no such attribute).
        return TOOL_DISPATCH[tool](parameters, None, speak)

    else:
        raise ValueError(
            f"Unknown tool '{tool}' — no such tool exists. "
            f"Available tools: {sorted(TOOL_DISPATCH.keys())}"
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
    ) -> str:
        _log.info("Goal: %s", goal, extra={"task_id": task_id})

        replan_attempts = 0
        completed_steps: list = []
        step_results:    dict = {}
        plan = create_plan(goal, task_id=task_id)

        while True:
            steps = plan.get("steps", [])
            if not steps:
                msg = "I couldn't create a valid plan for this task, sir."
                if speak: speak(msg)
                return msg

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

                log_task_event(task_id, step_num, tool, desc, "started")
                step_start = time.monotonic()

                attempt = 1
                step_ok = False

                while attempt <= 3:
                    if cancel_flag and cancel_flag.is_set():
                        break
                    try:
                        result = _call_tool(tool, params, speak, task_id=task_id, step_num=step_num)
                        step_results[step_num] = result
                        completed_steps.append(step)
                        log_task_event(
                            task_id, step_num, tool, desc, "done", str(result)[:200],
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
                        log_task_event(task_id, step_num, tool, desc, "attempt_failed", f"attempt {attempt}: {error_msg}")

                        recovery = analyze_error(step, error_msg, attempt=attempt, task_id=task_id)
                        decision = recovery["decision"]
                        user_msg = recovery.get("user_message", "")

                        if speak and user_msg:
                            speak(user_msg)

                        if decision == ErrorDecision.RETRY:
                            log_task_event(task_id, step_num, tool, desc, "retried", f"attempt {attempt} -> {attempt + 1}")
                            attempt += 1
                            time.sleep(2)
                            continue

                        elif decision == ErrorDecision.SKIP:
                            log_task_event(
                                task_id, step_num, tool, desc, "skipped", "skipped (non-critical)",
                                duration_ms=int((time.monotonic() - step_start) * 1000),
                            )
                            completed_steps.append(step)
                            step_ok = True
                            break

                        elif decision == ErrorDecision.ABORT:
                            msg = f"Task aborted, sir. {recovery.get('reason', '')}"
                            log_task_event(
                                task_id, step_num, tool, desc, "failed", f"aborted: {recovery.get('reason', '')}",
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
                                    )
                                    if isinstance(res, str) and any(
                                        marker in res for marker in _RECOVERY_NOT_EXECUTED_MARKERS
                                    ):
                                        _log.info(
                                            "Recovery via code_helper was denied — no live user to confirm "
                                            "(player=None). Autonomous code-fix recovery cannot run in this context.",
                                            extra={"task_id": task_id},
                                        )
                                        log_task_event(
                                            task_id, step_num, tool, desc, "failed",
                                            "recovery fix denied — no live user to confirm (player=None)",
                                            duration_ms=int((time.monotonic() - step_start) * 1000),
                                        )
                                    else:
                                        log_task_event(
                                            task_id, step_num, fixed_step.get("tool"), desc, "done",
                                            f"recovered via fix: {str(res)[:150]}",
                                            duration_ms=int((time.monotonic() - step_start) * 1000),
                                        )
                                        step_results[step_num] = res
                                        completed_steps.append(step)
                                        step_ok = True
                                        break
                                except Exception as fix_err:
                                    log_task_event(
                                        task_id, step_num, tool, desc, "failed", f"fix generation failed: {fix_err}",
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
                    log_task_event(
                        task_id, step_num, tool, desc, "failed", "max retries exceeded",
                        duration_ms=int((time.monotonic() - step_start) * 1000),
                    )

                if not success:
                    break

            if success:
                return self._summarize(goal, completed_steps, speak, task_id=task_id)

            if replan_attempts >= self.MAX_REPLAN_ATTEMPTS:
                msg = f"Task failed after {replan_attempts} replan attempts, sir."
                if speak: speak(msg)
                return msg

            if speak: speak("Adjusting my approach, sir.")
            log_task_event(
                task_id, failed_step.get("step") if failed_step else None,
                failed_step.get("tool") if failed_step else None, goal[:200],
                "replanned", f"after step failure: {failed_error[:150]}",
            )
            replan_attempts += 1
            plan = replan(goal, completed_steps, failed_step, failed_error, task_id=task_id)

    def _summarize(self, goal: str, completed_steps: list, speak: Callable | None, task_id: str | None = None) -> str:
        fallback  = f"All done, sir. Completed {len(completed_steps)} steps for: {goal[:60]}."
        steps_str = "\n".join(f"- {s.get('description', '')}" for s in completed_steps)
        prompt    = (
            f'User goal: "{goal}"\n'
            f"Completed steps:\n{steps_str}\n\n"
            "Write a single natural sentence summarising what was accomplished. "
            "Address the user as 'sir'. Be direct and positive."
        )
        try:
            summary = call_llm_text(prompt, task_id=task_id, purpose="summarize task")
            if summary:
                if speak: speak(summary)
                return summary
        except Exception:
            pass
        if speak: speak(fallback)
        return fallback
