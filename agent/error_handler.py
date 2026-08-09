"""
MARK XL — Error Handler
Replaces google.generativeai with local Ollama via core.llm_client.
"""
import json
import logging
import re
from enum import Enum

from core.llm_client import call_llm_text
from config import BASE_DIR

_log = logging.getLogger("jarvis.error_handler")


class ErrorDecision(Enum):
    RETRY  = "retry"
    SKIP   = "skip"
    REPLAN = "replan"
    ABORT  = "abort"


ERROR_ANALYST_PROMPT = """You are the error recovery module of MARK XL AI assistant.

A task step has failed. Analyze the error and decide what to do.

DECISIONS:
- retry   : Transient error (network timeout, temporary file lock, race condition).
- skip    : This step is not critical and the task can succeed without it.
- replan  : The approach was wrong. A different tool or method should be tried.
- abort   : The task is fundamentally impossible or unsafe to continue.

Return ONLY valid JSON:
{
  "decision": "retry|skip|replan|abort",
  "reason": "why it failed",
  "fix_suggestion": "what to try instead (for replan)",
  "max_retries": 1,
  "user_message": "Short message to tell the user (max 15 words)"
}
"""


def analyze_error(
    step:         dict,
    error:        str,
    attempt:      int = 1,
    max_attempts: int = 2,
    task_id:      str | None = None,
) -> dict:
    if attempt >= max_attempts:
        _log.warning("Max attempts for step %s — forcing replan", step.get('step'))
        return {
            "decision":       ErrorDecision.REPLAN,
            "reason":         f"Failed {attempt} times: {error[:100]}",
            "fix_suggestion": "Try a completely different approach or tool",
            "max_retries":    0,
            "user_message":   "Trying a different approach, sir.",
        }

    prompt = f"""Failed step:
Tool: {step.get('tool')}
Description: {step.get('description')}
Parameters: {json.dumps(step.get('parameters', {}), indent=2)}
Critical: {step.get('critical', False)}

Error:
{error[:500]}

Attempt number: {attempt}"""

    try:
        text   = call_llm_text(
            prompt, system=ERROR_ANALYST_PROMPT,
            task_id=task_id, step_num=step.get("step"), purpose="analyze error",
        )
        text   = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
        result = json.loads(text)

        decision_str = result.get("decision", "replan").lower()
        decision_map = {
            "retry":  ErrorDecision.RETRY,
            "skip":   ErrorDecision.SKIP,
            "replan": ErrorDecision.REPLAN,
            "abort":  ErrorDecision.ABORT,
        }
        result["decision"] = decision_map.get(decision_str, ErrorDecision.REPLAN)

        if step.get("critical") and result["decision"] == ErrorDecision.SKIP:
            result["decision"]     = ErrorDecision.REPLAN
            result["user_message"] = "This step is critical — finding alternative approach, sir."

        _log.info("Decision: %s — %s", result['decision'].value, result.get('reason', ''))
        return result

    except Exception as e:
        _log.warning("Analysis failed: %s — defaulting to replan", e)
        return {
            "decision":       ErrorDecision.REPLAN,
            "reason":         str(e),
            "fix_suggestion": "Try alternative approach",
            "max_retries":    1,
            "user_message":   "Encountered an issue, adjusting approach, sir.",
        }


def generate_fix(step: dict, error: str, fix_suggestion: str, task_id: str | None = None) -> dict:
    prompt = f"""A task step failed. Generate a replacement step.

Original step:
Tool: {step.get('tool')}
Description: {step.get('description')}
Parameters: {json.dumps(step.get('parameters', {}), indent=2)}

Error: {error[:300]}
Fix suggestion: {fix_suggestion}

Write a Python script that accomplishes the same goal differently.
Return ONLY the Python code, no explanation."""

    try:
        code = call_llm_text(
            prompt,
            task_id=task_id, step_num=step.get("step"), purpose="generate fix",
        )
        code = re.sub(r"```(?:python)?", "", code).strip().rstrip("`").strip()
        return {
            "step":        step.get("step"),
            "tool":        "code_helper",
            "description": f"Auto-fix for: {step.get('description')}",
            "parameters": {
                "action":      "run",
                "description": fix_suggestion,
                "code":        code,
                "language":    "python",
            },
            "depends_on": step.get("depends_on", []),
            "critical":   step.get("critical", False),
        }
    except Exception as e:
        _log.warning("Fix generation failed: %s", e)
        raise RuntimeError(f"Could not generate a fix: {e}")
