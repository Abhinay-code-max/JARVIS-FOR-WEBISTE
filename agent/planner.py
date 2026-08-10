"""
MARK XL — Task Planner
Replaces google.generativeai with local Ollama via core.llm_client.
"""
import json
import logging
import re

from core.llm_client import call_llm_text
from core.tool_dispatch import TOOL_DISPATCH
from core.tool_declarations import build_planner_tool_reference
from config import BASE_DIR

_log = logging.getLogger("jarvis.planner")

# Tool names _call_tool() actually knows how to run. Any step whose "tool"
# isn't in this set (or the literal "generated_code", handled separately
# below) would otherwise reach executor.py's unknown-tool path — catching
# it here means a hallucinated/misspelled tool name never leaves the
# planner in the first place.
_VALID_TOOLS = set(TOOL_DISPATCH.keys())


# Generated from core/tool_declarations.py's TOOL_DECLARATIONS — the same
# schema every tool call is validated against (core/tool_contracts.py) and
# the same schema sent to the interactive LLM (main.py's OLLAMA_TOOLS) —
# instead of being a third, separately hand-maintained copy. The previous
# hand-written version had already drifted: it was missing file_processor,
# daily_briefing, and vision_fix_code entirely, and had a stale action
# list for code_helper (missing build/optimize/screen_debug).
PLANNER_PROMPT = f"""You are the planning module of MARK XL, a personal AI assistant.
Your job: break any user goal into a sequence of steps using ONLY the tools listed below.

ABSOLUTE RULES:
- NEVER use generated_code or write Python scripts. It does not exist.
- NEVER reference previous step results in parameters. Every step is independent.
- Use web_search for ANY information retrieval, research, or current data.
- Use file_controller to save content to disk.
- Max 5 steps. Use the minimum steps needed.

AVAILABLE TOOLS AND THEIR PARAMETERS:

{build_planner_tool_reference(_VALID_TOOLS)}

OUTPUT — return ONLY valid JSON, no markdown, no explanation, no code blocks:
{{
  "goal": "...",
  "steps": [
    {{
      "step": 1,
      "tool": "tool_name",
      "description": "what this step does",
      "parameters": {{}},
      "critical": true
    }}
  ]
}}
"""


def create_plan(goal: str, context: str = "", task_id: str | None = None) -> dict:
    user_input = f"Goal: {goal}"
    if context:
        user_input += f"\n\nContext: {context}"

    try:
        text = call_llm_text(
            user_input, system=PLANNER_PROMPT,
            task_id=task_id, purpose="create plan",
        )
        text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()

        plan = json.loads(text)
        if "steps" not in plan or not isinstance(plan["steps"], list):
            raise ValueError("Invalid plan structure")

        for step in plan["steps"]:
            tool = step.get("tool")
            if tool == "generated_code":
                _log.warning("generated_code in step %s — replacing with web_search", step.get('step'))
                step["tool"]       = "web_search"
                step["parameters"] = {"query": step.get("description", goal)[:200]}
            elif tool not in _VALID_TOOLS:
                _log.warning("Unknown tool '%s' in step %s — replacing with web_search", tool, step.get('step'))
                step["tool"]       = "web_search"
                step["parameters"] = {"query": step.get("description", goal)[:200]}

        _log.debug("Plan: %d steps", len(plan['steps']))
        for s in plan["steps"]:
            _log.debug("  Step %s: [%s] %s", s['step'], s['tool'], s['description'])
        return plan

    except json.JSONDecodeError as e:
        _log.warning("JSON parse failed: %s", e)
        return _fallback_plan(goal)
    except Exception as e:
        _log.warning("Planning failed: %s", e)
        return _fallback_plan(goal)


def _fallback_plan(goal: str) -> dict:
    _log.info("Fallback plan")
    return {
        "goal":  goal,
        "steps": [
            {
                "step":        1,
                "tool":        "web_search",
                "description": f"Search for: {goal}",
                "parameters":  {"query": goal},
                "critical":    True,
            }
        ],
    }


def replan(goal: str, completed_steps: list, failed_step: dict, error: str, task_id: str | None = None) -> dict:
    completed_summary = "\n".join(
        f"  - Step {s['step']} ({s['tool']}): DONE" for s in completed_steps
    )
    prompt = f"""Goal: {goal}

Already completed:
{completed_summary if completed_summary else '  (none)'}

Failed step: [{failed_step.get('tool')}] {failed_step.get('description')}
Error: {error}

Create a REVISED plan for the remaining work only. Do not repeat completed steps."""

    try:
        text = call_llm_text(
            prompt, system=PLANNER_PROMPT,
            task_id=task_id, purpose="replan",
        )
        text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
        plan = json.loads(text)

        for step in plan.get("steps", []):
            tool = step.get("tool")
            if tool == "generated_code":
                step["tool"]       = "web_search"
                step["parameters"] = {"query": step.get("description", goal)[:200]}
            elif tool not in _VALID_TOOLS:
                _log.warning("Unknown tool '%s' in step %s — replacing with web_search", tool, step.get('step'))
                step["tool"]       = "web_search"
                step["parameters"] = {"query": step.get("description", goal)[:200]}

        _log.debug("Revised plan: %d steps", len(plan['steps']))
        return plan
    except Exception as e:
        _log.warning("Replan failed: %s", e)
        return _fallback_plan(goal)
