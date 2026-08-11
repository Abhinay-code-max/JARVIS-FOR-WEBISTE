"""
tests/test_headless_extraction.py
===================================
The headless-extraction phase's Step 1.4: package core as UI-independent.

Step 1.1's investigation found the seam already clean (no real PyQt6/
ui.py imports anywhere under core/ or agent/ — one comment-only mention
in core/tool_declarations.py). This file turns that one-time manual grep
into a permanent, executable guarantee: a real subprocess imports the
full orchestration surface Step 1.5's FastAPI app will need and confirms
neither PyQt6 nor sounddevice — both genuinely installed in this
environment, confirmed below, so this isn't a vacuous pass — ever lands
in sys.modules as a result. Nothing here is a restructuring: no package
was moved, split, or given new metadata (this project has no
setup.py/pyproject.toml to begin with, and the plan's own Step 1.4
description says "nothing structural expected... just the packaging
move" once the seam is already clean) — this is that move's real
regression test.

Uses a real subprocess (not sys.modules introspection in-process) because
this test process may itself have already imported something (e.g. via
an earlier test file's own imports) that would pollute an in-process
check — a subprocess starts with a genuinely empty sys.modules.
"""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Every module Step 1.5's FastAPI/WebSocket app will need to import to
# actually dispatch a tool call and run a task — the real orchestration
# surface, not a token sample. core/calendar_auth.py is deliberately
# excluded: its own top-level imports (google-auth etc.) genuinely aren't
# installed in this environment, an unrelated pre-existing condition, not
# a UI-independence question this test is about.
ORCHESTRATION_MODULES = [
    "core.db",
    "core.policy",
    "core.tool_dispatch",
    "core.tool_contracts",
    "core.tool_declarations",
    "core.postconditions",
    "core.confirm",
    "core.task_approval",
    "core.tool_gate",
    "core.proactive",
    "core.github_ci_auth",
    "core.llm_client",
    "core.logging_setup",
    "agent.executor",
    "agent.task_queue",
    "agent.planner",
    "agent.error_handler",
    "agent.step_references",
    "memory.memory_manager",
]

# Importable in *this* environment, confirmed directly rather than
# assumed — see test_pyqt6_and_sounddevice_are_genuinely_installed_here.
# If either weren't actually installed, an import failing to appear in
# sys.modules would prove nothing about the code's own cleanliness.
_UI_MARKERS = ["PyQt6", "sounddevice"]


def _run_probe(imports: list[str]) -> subprocess.CompletedProcess:
    script = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        f"for m in {imports!r}:\n"
        "    __import__(m)\n"
        f"hit = [m for m in {_UI_MARKERS!r} if m in sys.modules]\n"
        "print(json.dumps({'ui_modules_loaded': hit}))\n"
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=60, cwd=str(REPO_ROOT),
    )


class UiIndependenceTest(unittest.TestCase):
    def test_pyqt6_and_sounddevice_are_genuinely_installed_here(self):
        """Precondition for the real test below to mean anything: if
        these weren't actually installed in this environment, their
        absence from sys.modules after importing the orchestration layer
        would be trivially true and prove nothing."""
        result = _run_probe(["PyQt6.QtWidgets", "sounddevice"])
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_orchestration_layer_never_loads_pyqt6_or_sounddevice(self):
        result = _run_probe(ORCHESTRATION_MODULES)
        self.assertEqual(
            result.returncode, 0,
            f"orchestration layer failed to import standalone:\n{result.stderr}",
        )
        import json
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(
            payload["ui_modules_loaded"], [],
            "core/agent orchestration modules pulled in a UI dependency — "
            "the headless-extraction boundary broke",
        )

    def test_main_py_itself_does_pull_in_pyqt6(self):
        """Contrast case, proving the probe methodology actually detects
        a real UI dependency when one genuinely exists — not just that
        the orchestration layer happens to pass. Without this, a probe
        that always reports [] regardless of what's imported would give
        the test above a false sense of security."""
        result = _run_probe(["main"])
        # main.py itself is expected to succeed or fail for reasons
        # unrelated to this check (missing audio hardware, etc.) — only
        # the *sys.modules* signal matters here, checked either way.
        script = (
            "import sys, json\n"
            f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
            "try:\n"
            "    import main\n"
            "except Exception:\n"
            "    pass\n"
            f"hit = [m for m in {_UI_MARKERS!r} if any(k.startswith(m) for k in sys.modules)]\n"
            "print(json.dumps({'ui_modules_loaded': hit}))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=60, cwd=str(REPO_ROOT),
        )
        import json
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertIn("PyQt6", payload["ui_modules_loaded"], result.stderr)


class ExtractionSeamStaticCheckTest(unittest.TestCase):
    """The original Step 1.1 grep, made permanent — a real import
    boundary (above) is the stronger guarantee, but this catches a
    genuine string reference (not just an executed import statement)
    creeping back in, cheaply, without spawning a subprocess."""

    def test_no_real_pyqt6_or_ui_import_under_core_or_agent(self):
        import ast

        offenders = []
        for pkg in ("core", "agent"):
            for path in (REPO_ROOT / pkg).rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        names = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        names = [node.module] if node.module else []
                    else:
                        continue
                    for name in names:
                        if name and (name == "PyQt6" or name.startswith("PyQt6.") or name == "ui" or name.startswith("ui.")):
                            offenders.append(f"{path.relative_to(REPO_ROOT)}: {name}")

        self.assertEqual(offenders, [], f"real PyQt6/ui import(s) found under core/ or agent/: {offenders}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
