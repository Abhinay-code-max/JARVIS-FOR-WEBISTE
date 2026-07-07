"""
actions/auto_fix_code.py
=========================
Runs a Python (or other) script, captures the REAL error from stdout/stderr
(much more reliable than reading screenshots), sends the error + file content
to the LLM for a fix, writes the fix back, and re-runs to verify.
"""

from __future__ import annotations
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional


from config import load_config as _load_config
from core.llm_client import call_llm_text as _call_llm_text


def _run_script(file_path: str, timeout: int = 15) -> tuple[bool, str, str]:
    """
    Runs the script and captures stdout/stderr.
    Returns (success, stdout, stderr).
    """
    try:
        result = subprocess.run(
            [sys.executable, file_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        success = result.returncode == 0
        return success, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"Script timed out after {timeout} seconds (may be waiting for input or running an infinite loop)."
    except Exception as e:
        return False, "", f"Could not run script: {e}"


def _parse_traceback(stderr: str) -> dict:
    """
    Extracts the key info from a Python traceback:
      - error_type (e.g. IndexError)
      - error_message
      - line_number
      - file_path (from the traceback)
    """
    info = {
        "error_type":    "",
        "error_message": "",
        "line_number":   None,
        "file_path":     "",
    }

    if not stderr:
        return info

    # Last line is usually "ErrorType: message"
    lines = stderr.strip().split("\n")
    if lines:
        last_line = lines[-1]
        m = re.match(r"(\w+(?:Error|Exception|Warning)):\s*(.*)", last_line)
        if m:
            info["error_type"]    = m.group(1)
            info["error_message"] = m.group(2)

    # Find the LAST "File "...", line N" entry (closest to actual error)
    file_line_matches = re.findall(r'File "([^"]+)", line (\d+)', stderr)
    if file_line_matches:
        info["file_path"], line_no = file_line_matches[-1]
        info["line_number"] = int(line_no)

    return info


def _call_llm_for_fix(file_content: str, error_info: dict, stderr: str, config: dict) -> Optional[str]:
    """
    Sends the broken file + real error to the LLM and asks for a corrected
    full-file replacement. Returns the fixed code, or None on failure.
    """
    model = config.get("llm_model", "qwen2.5:3b")

    prompt = f"""You are an expert Python debugger. A script failed with a real error.

ERROR TYPE: {error_info.get('error_type', 'Unknown')}
ERROR MESSAGE: {error_info.get('error_message', '')}
LINE NUMBER: {error_info.get('line_number', 'Unknown')}

FULL TRACEBACK:
{stderr}

CURRENT FILE CONTENT:
```python
{file_content}
```

Fix the bug. Output ONLY the complete corrected Python file content, with no explanation, no markdown code fences, no commentary — just the raw corrected source code that should replace the file entirely."""

    try:
        fixed_code = _call_llm_text(
            prompt,
            system="You are a precise code-fixing assistant. Output only raw code, never markdown fences or explanations.",
            model=model,
            timeout=60,
        )

        # Strip markdown fences if the model added them anyway
        fixed_code = re.sub(r"^```(?:python)?\n", "", fixed_code)
        fixed_code = re.sub(r"\n```$", "", fixed_code)

        return fixed_code.strip()
    except Exception as e:
        print(f"[AutoFixCode] LLM call failed: {e}")
        return None


def auto_fix_code(parameters: dict, player=None, speak=None) -> str:
    """
    Main entry point.
    parameters:
        file_path (str): path to the Python file to run and fix
        max_attempts (int): how many fix attempts before giving up (default 2)
    """
    params       = parameters or {}
    file_path    = params.get("file_path", "").strip()
    max_attempts = int(params.get("max_attempts", 2))

    if not file_path:
        return "Please specify which file to run and fix."

    fpath = Path(file_path)
    if not fpath.exists():
        return f"File not found: {file_path}"

    config = _load_config()

    def _log(msg: str):
        print(f"[AutoFixCode] {msg}")
        if player:
            try:
                player.write_log(f"[autofix] {msg}")
            except Exception:
                pass

    for attempt in range(1, max_attempts + 1):
        _log(f"Attempt {attempt}: running {fpath.name}...")
        success, stdout, stderr = _run_script(str(fpath))

        if success:
            if attempt == 1:
                return f"I ran {fpath.name} and it executed successfully with no errors. Output: {stdout.strip()[:200] if stdout.strip() else '(no output)'}"
            else:
                return f"Fixed! {fpath.name} now runs successfully after {attempt - 1} fix(es). Output: {stdout.strip()[:200] if stdout.strip() else '(no output)'}"

        # Parse the real error
        error_info = _parse_traceback(stderr)
        _log(f"Error found: {error_info.get('error_type')}: {error_info.get('error_message')} (line {error_info.get('line_number')})")

        if attempt >= max_attempts:
            return (
                f"I ran {fpath.name} and found a {error_info.get('error_type', 'error')}: "
                f"{error_info.get('error_message', '')} at line {error_info.get('line_number', '?')}. "
                f"I attempted {max_attempts} fix(es) but couldn't fully resolve it. "
                f"You may want to review it manually."
            )

        # Read current file content
        try:
            file_content = fpath.read_text(encoding="utf-8")
        except Exception as e:
            return f"Could not read {fpath.name}: {e}"

        _log("Asking the AI to generate a fix...")
        fixed_code = _call_llm_for_fix(file_content, error_info, stderr, config)

        if not fixed_code:
            return f"Found the error ({error_info.get('error_type')}: {error_info.get('error_message')}) but couldn't generate a fix. You may want to review it manually."

        # Backup original before overwriting
        backup_path = fpath.with_suffix(fpath.suffix + ".backup")
        try:
            backup_path.write_text(file_content, encoding="utf-8")
        except Exception:
            pass

        # Write the fix
        try:
            fpath.write_text(fixed_code, encoding="utf-8")
            _log(f"Applied fix to {fpath.name} (backup saved as {backup_path.name})")
        except Exception as e:
            return f"Found a fix but couldn't write it to the file: {e}"

    return f"Finished attempting fixes on {fpath.name}."
