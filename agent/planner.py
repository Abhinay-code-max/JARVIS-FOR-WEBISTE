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
from agent.step_references import find_all_references
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
# list for code_helper (missing build/optimize/screen_debug). Only the
# ABSOLUTE RULES / OUTPUT text below is hand-written — the tool reference
# section stays generated, so this is still one canonical source, not a
# second hand-edited copy of it.
PLANNER_PROMPT = f"""You are the planning module of MARK XL, a personal AI assistant.
Your job: break any user goal into a sequence of steps using ONLY the tools listed below.

ABSOLUTE RULES:
- NEVER use generated_code or write Python scripts. It does not exist.
- A step CAN reference an EARLIER step's result using ${{step_N.output}}
  (e.g. "message_text": "Cheapest flight: ${{step_1.output}}") — this
  substitutes the ENTIRE text result of step N, not a specific field.
  N must be a step number that appears EARLIER in this same plan (a lower
  step number than the one using it) — never itself or a later step, and
  never a step number that doesn't exist in this plan. Steps still run in
  order, one at a time; this does not let you reorder or parallelize them.
- Use web_search for ANY information retrieval, research, or current data.
- Use file_controller to save content to disk.
- Max 5 steps. Use the minimum steps needed.
- For computer_control actions that target a specific UI element (a button,
  icon, menu item, link, field) that you can describe in words, prefer
  action="screen_click" with a "description" of the element over guessing
  raw x/y coordinates — screen_click locates the element visually and
  clicks it, which is far more reliable than a guessed pixel position.
  Only use raw x/y for precise or relative positioning you already know
  (e.g. a small mouse nudge, or coordinates an earlier screen_find step
  already returned).
- Use screen_process to check what's currently on screen mid-plan (e.g.
  confirm a window actually opened, or a dialog appeared) when a later
  step's success depends on an earlier step's uncertain outcome.

EXAMPLE — opening an app and clicking a described element instead of
guessing coordinates:
  {{"step": 1, "tool": "open_app", "parameters": {{"app_name": "notepad"}}}}
  {{"step": 2, "tool": "computer_control", "parameters": {{"action": "screen_click", "description": "the File menu"}}}}

EXAMPLE — a short GUI chain, typing into a described field after opening
the app:
  {{"step": 1, "tool": "open_app", "parameters": {{"app_name": "notepad"}}}}
  {{"step": 2, "tool": "computer_control", "parameters": {{"action": "screen_click", "description": "the main text editing area"}}}}
  {{"step": 3, "tool": "computer_control", "parameters": {{"action": "type", "text": "Hello, world!"}}}}

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


def _validate_and_fix_plan(plan: dict, goal: str) -> dict:
    """Applied to every freshly-parsed plan from both create_plan() and
    replan() — one shared pass so the two can't validate differently.

    - Duplicate step numbers: not a single step's problem to fix (which
      of the two "step 2"s would a later ${{step_2.output}} even mean?) —
      the plan's numbering is unreliable, so the whole plan is rejected
      in favor of _fallback_plan(), same as an unparseable one.
    - generated_code / unknown tool: unchanged from before — replace that
      one step with a web_search of its own description.
    - A step referencing itself, a later step, or a step number that
      doesn't exist in this plan: under strictly sequential execution
      that reference can never resolve, so the step's whole intent as
      written is unsafe to keep — same per-step web_search fallback as
      an unknown tool (which already discards the old `parameters` dict
      wholesale, so the bad reference goes with it)."""
    steps = plan.get("steps", [])

    step_nums = [s.get("step") for s in steps]
    if len(step_nums) != len(set(step_nums)):
        _log.warning("Duplicate step numbers in plan — falling back")
        return _fallback_plan(goal)

    valid_step_nums = set(step_nums)

    for step in steps:
        tool    = step.get("tool")
        own_num = step.get("step")

        if tool == "generated_code":
            _log.warning("generated_code in step %s — replacing with web_search", step.get('step'))
            step["tool"]       = "web_search"
            step["parameters"] = {"query": step.get("description", goal)[:200]}
            continue

        if tool not in _VALID_TOOLS:
            _log.warning("Unknown tool '%s' in step %s — replacing with web_search", tool, step.get('step'))
            step["tool"]       = "web_search"
            step["parameters"] = {"query": step.get("description", goal)[:200]}
            continue

        refs = find_all_references(step.get("parameters", {}))
        if refs:
            bad_refs = refs if own_num is None else [
                r for r in refs if r not in valid_step_nums or r >= own_num
            ]
            if bad_refs:
                _log.warning(
                    "Step %s references invalid/forward/self step number(s) %s — replacing with web_search",
                    own_num, bad_refs,
                )
                step["tool"]       = "web_search"
                step["parameters"] = {"query": step.get("description", goal)[:200]}

    return plan


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

        plan = _validate_and_fix_plan(plan, goal)

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

        plan = _validate_and_fix_plan(plan, goal)

        _log.debug("Revised plan: %d steps", len(plan.get('steps', [])))
        return plan
    except Exception as e:
        _log.warning("Replan failed: %s", e)
        return _fallback_plan(goal)
