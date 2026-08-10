"""
agent/step_references.py
==========================
${step_N.output} reference syntax — one regex, one place, shared between
agent/planner.py (forward-reference validation at plan-creation time) and
agent/executor.py (substitution at execution time) so the two can't drift
apart.

Whole-string substitution only: a reference always resolves to the
*entire* prior step's result string (every TOOL_DISPATCH tool returns
plain text today — see the tool-contracts phase's investigation).
${step_2.output.price}-style field access is out of scope; it would need
per-tool structured output_schema work, deferred as a separate future
phase.
"""
from __future__ import annotations

import re

_STEP_REF_PATTERN = re.compile(r"\$\{step_(\d+)\.output\}")


def find_references(value) -> list[int]:
    """Every step number referenced by ${step_N.output} inside `value`.
    Recurses into list items (e.g. web_search's `items` array param);
    parameter values in TOOL_DECLARATIONS are never nested any deeper
    than that today, so this doesn't need to go further."""
    if isinstance(value, str):
        return [int(m) for m in _STEP_REF_PATTERN.findall(value)]
    if isinstance(value, list):
        refs: list[int] = []
        for item in value:
            refs.extend(find_references(item))
        return refs
    return []


def find_all_references(parameters: dict) -> list[int]:
    """Every step number referenced anywhere in a step's parameters dict."""
    refs: list[int] = []
    for value in parameters.values():
        refs.extend(find_references(value))
    return refs


class UnresolvedReferenceError(Exception):
    """Raised by resolve_references() the first time a referenced step
    has no entry in step_results — skipped, failed, or (should already be
    impossible by the time execution reaches this — see
    agent/planner.py's forward-reference validation) simply hasn't run
    yet. Callers must not silently proceed with an unsubstituted
    template or an empty string; this is a real, terminal step failure,
    not a value to paper over."""
    def __init__(self, step_num: int):
        self.step_num = step_num
        super().__init__(f"step {step_num} has no result to reference")


def resolve_references(parameters: dict, step_results: dict) -> dict:
    """Returns a new dict with every ${step_N.output} in a string (or
    list-of-strings) value substituted for str(step_results[N])."""
    def _sub(value):
        if isinstance(value, str):
            def _replace(m):
                ref_num = int(m.group(1))
                if ref_num not in step_results:
                    raise UnresolvedReferenceError(ref_num)
                return str(step_results[ref_num])
            return _STEP_REF_PATTERN.sub(_replace, value)
        if isinstance(value, list):
            return [_sub(v) for v in value]
        return value

    return {key: _sub(value) for key, value in parameters.items()}
