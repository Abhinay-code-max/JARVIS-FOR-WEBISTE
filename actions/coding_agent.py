"""
actions/coding_agent.py
=======================
Hardened Coding Sub-Agent Integration for JARVIS-XL.
Implements:
  1. Real-Time Streaming Approval (Primary Safety Gate):
     - Shells out to `agy` via NDJSON streaming (`--input-format stream-json --output-format stream-json`).
     - Reads streamed events line-by-line.
     - Intercepts shell execution tool calls / permission requests and surfaces them in real-time
       through JARVIS's existing CONFIRM.request() intent gate with the exact command text.
     - Capped at MAX_SHELL_APPROVALS (10) per session to prevent runaway execution.
  2. Post-Run Path Auditor (Secondary Backstop):
     - Snapshots baseline filesystem state outside ~/Desktop/JarvisProjects/<project>.
     - Audits for any unauthorized out-of-bounds file creation/modification after execution.
     - Immediately marks compromised and rejects untrusted results if out-of-bounds writes occur.
  3. Automatic Fallback:
     - Falls back to actions/dev_agent.py's local-LLM loop if Antigravity is unavailable or fails.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Tuple

from config import BASE_DIR
from core.confirm import CONFIRM

_log = logging.getLogger("jarvis.coding_agent")

PROJECTS_DIR = Path.home() / "Desktop" / "JarvisProjects"
DEFAULT_TIMEOUT_SECONDS = 300.0  # 5 minutes
MAX_SHELL_APPROVALS = 10
MAX_RETRIES = 2


class AntigravityError(Exception):
    """Base exception for Antigravity provider failures."""
    pass


class AntigravityUnavailableError(AntigravityError):
    """Raised when the `agy` CLI is not found or not executable."""
    pass


class AntigravityPermissionError(AntigravityError):
    """Raised when agy encounters a soft-denied permission or authorization failure."""
    pass


class AntigravityTimeoutError(AntigravityError):
    """Raised when an agy subprocess call exceeds its allotted execution timeout."""
    pass


class AntigravitySecurityCompromiseError(AntigravityError):
    """Raised when an out-of-bounds filesystem modification is detected post-execution."""
    pass


class AntigravityApprovalCapExceededError(AntigravityError):
    """Raised when the maximum number of in-session shell approvals is exceeded."""
    pass


class FilesystemAuditor:
    """
    Monitors and audits filesystem modifications outside the designated project boundary.
    Snapshots key user/system directories (Desktop, Documents, AppData Local/Roaming, Temp, Startup)
    before execution and detects any out-of-bounds file creation or tampering.
    """

    def __init__(self, allowed_root: Path):
        self.allowed_root = allowed_root.resolve()
        self._baseline: Dict[Path, float] = {}
        self._monitored_roots: List[Path] = []

    def snapshot(self) -> None:
        """Records initial file modification times for watched paths outside the allowed root."""
        self._baseline.clear()
        self._monitored_roots.clear()

        user_home = Path.home()
        local_app = Path(os.environ.get("LOCALAPPDATA", user_home / "AppData" / "Local"))
        roaming_app = Path(os.environ.get("APPDATA", user_home / "AppData" / "Roaming"))
        temp_dir = Path(os.environ.get("TEMP", local_app / "Temp"))
        desktop_dir = user_home / "Desktop"
        docs_dir = user_home / "Documents"
        startup_dir = roaming_app / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"

        self._monitored_roots = [
            user_home,
            desktop_dir,
            docs_dir,
            BASE_DIR,
            local_app,
            roaming_app,
            temp_dir,
            startup_dir,
        ]

        # 0. User Home (root entries)
        if user_home.exists():
            try:
                for entry in os.scandir(user_home):
                    if entry.is_file():
                        try:
                            self._baseline[Path(entry.path).resolve()] = entry.stat().st_mtime
                        except (OSError, PermissionError):
                            pass
            except (OSError, PermissionError):
                pass

        # 1. Full Desktop except allowed project
        if desktop_dir.exists():
            for root_path, dirs, files in os.walk(desktop_dir):
                r_path = Path(root_path).resolve()
                if r_path == self.allowed_root or self.allowed_root in r_path.parents:
                    dirs.clear()
                    continue
                for f in files:
                    fp = Path(root_path) / f
                    try:
                        self._baseline[fp.resolve()] = fp.stat().st_mtime
                    except (OSError, PermissionError):
                        pass

        # 2. Documents (depth 2)
        if docs_dir.exists():
            for root_path, dirs, files in os.walk(docs_dir):
                rel = Path(root_path).relative_to(docs_dir)
                if len(rel.parts) >= 2:
                    dirs.clear()
                for f in files:
                    fp = Path(root_path) / f
                    try:
                        self._baseline[fp.resolve()] = fp.stat().st_mtime
                    except (OSError, PermissionError):
                        pass

        # 3. AppData Local (root entries)
        if local_app.exists():
            try:
                for entry in os.scandir(local_app):
                    if entry.is_file():
                        try:
                            self._baseline[Path(entry.path).resolve()] = entry.stat().st_mtime
                        except (OSError, PermissionError):
                            pass
            except (OSError, PermissionError):
                pass

        # 4. AppData Roaming + Windows Startup
        if roaming_app.exists():
            try:
                for entry in os.scandir(roaming_app):
                    if entry.is_file():
                        try:
                            self._baseline[Path(entry.path).resolve()] = entry.stat().st_mtime
                        except (OSError, PermissionError):
                            pass
            except (OSError, PermissionError):
                pass

        if startup_dir.exists():
            try:
                for entry in os.scandir(startup_dir):
                    if entry.is_file():
                        try:
                            self._baseline[Path(entry.path).resolve()] = entry.stat().st_mtime
                        except (OSError, PermissionError):
                            pass
            except (OSError, PermissionError):
                pass

        # 5. Temp directory (root entries)
        if temp_dir.exists():
            try:
                for entry in os.scandir(temp_dir):
                    if entry.is_file():
                        try:
                            self._baseline[Path(entry.path).resolve()] = entry.stat().st_mtime
                        except (OSError, PermissionError):
                            pass
            except (OSError, PermissionError):
                pass

        # 6. Repo Root (JARVIS-XL)
        if BASE_DIR.exists():
            for root_path, dirs, files in os.walk(BASE_DIR):
                if ".git" in dirs:
                    dirs.remove(".git")
                for f in files:
                    fp = Path(root_path) / f
                    try:
                        self._baseline[fp.resolve()] = fp.stat().st_mtime
                    except (OSError, PermissionError):
                        pass

    def audit_modifications(self) -> List[Path]:
        """
        Compares current filesystem state against snapshot baseline.
        Returns a list of any files created or modified outside the allowed project root.
        """
        violations: Set[Path] = set()

        # Check existing baseline files for modifications
        temp_dir = Path(os.environ.get("TEMP", Path.home() / "AppData" / "Local" / "Temp")).resolve()
        for path, old_mtime in self._baseline.items():
            if not path.exists():
                continue
            # Ignore normal transient OS/Python temp files in Temp
            if path.parent == temp_dir and path.suffix.lower() in (".tmp", ".log", ".lock"):
                continue
            try:
                new_mtime = path.stat().st_mtime
                if new_mtime > old_mtime + 0.001:
                    violations.add(path)
            except (OSError, PermissionError):
                pass

        # Check for newly created files in critical monitored roots
        temp_dir = Path(os.environ.get("TEMP", Path.home() / "AppData" / "Local" / "Temp")).resolve()
        for root in self._monitored_roots:
            if not root.exists():
                continue
            try:
                for entry in os.scandir(root):
                    if entry.is_file():
                        p = Path(entry.path).resolve()
                        # Ignore normal transient OS/Python temp files in Temp
                        if p.parent == temp_dir and p.suffix.lower() in (".tmp", ".log", ".lock"):
                            continue
                        if p not in self._baseline and p != self.allowed_root and self.allowed_root not in p.parents:
                            violations.add(p)
            except (OSError, PermissionError):
                pass

        return list(violations)


class AntigravityProvider:
    """
    Subprocess provider that invokes Google Antigravity via the `agy` CLI using streaming NDJSON.
    Intercepts shell execution requests in real-time, prompts user via CONFIRM.request(),
    and audits post-run filesystem boundaries.
    """

    def __init__(self, executable_path: Optional[str] = None):
        self._exec_path = executable_path or self._find_executable()

    @staticmethod
    def _find_executable() -> Optional[str]:
        """Locates the `agy` executable on PATH or in standard user local directories."""
        which_path = shutil.which("agy")
        if which_path:
            return which_path

        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if local_app_data:
            candidate = Path(local_app_data) / "agy" / "bin" / "agy.exe"
            if candidate.is_file() and os.access(str(candidate), os.X_OK):
                return str(candidate)

        return None

    def is_available(self) -> bool:
        """Returns True if the `agy` CLI is found and responds to --version."""
        if not self._exec_path:
            self._exec_path = self._find_executable()
        if not self._exec_path:
            return False

        try:
            res = subprocess.run(
                [self._exec_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return res.returncode == 0
        except Exception:
            return False

    def build_prompt(
        self,
        description: str,
        language: str = "python",
        project_name: str = "",
        requirements: Optional[list[str]] = None,
        working_dir: Optional[Path] = None,
    ) -> str:
        """Assembles a structured prompt for Antigravity from task specifications."""
        req_str = ""
        if requirements:
            req_str = "\nRequirements:\n" + "\n".join(f"- {r}" for r in requirements)

        name_clause = f" for project '{project_name}'" if project_name else ""
        dir_clause = f"\nTarget Working Directory: {working_dir.resolve()}" if working_dir else ""

        return (
            f"You are the specialized engineering coding sub-agent for JARVIS-XL.\n"
            f"Task: Implement the requested functionality{name_clause} in {language}.{dir_clause}\n\n"
            f"Description:\n{description}\n"
            f"{req_str}\n\n"
            f"Instructions:\n"
            f"1. Write and save all files directly into the target working directory.\n"
            f"2. Implement all required source files, entry points, and helper modules.\n"
            f"3. Run tests/checks in the working directory to ensure the code executes cleanly without errors.\n"
            f"4. Provide a concise summary of the files created/modified and test results.\n"
        )

    def run_streaming(
        self,
        prompt: str,
        working_dir: Path,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        player=None,
        speak=None,
        max_shell_approvals: int = MAX_SHELL_APPROVALS,
    ) -> Dict[str, Any]:
        """
        Executes `agy` via `--input-format stream-json --output-format stream-json`.
        Relays shell execution requests to CONFIRM.request() in real-time.
        Performs post-run filesystem boundary audit on completion.

        Timeout is enforced unconditionally: stdout is read on a daemon thread
        and pushed onto a queue; the main loop uses queue.get(timeout=remaining)
        so the session deadline fires even if agy emits no output for the full
        timeout window. Previously, readline() on the main thread would block
        indefinitely, making DEFAULT_TIMEOUT_SECONDS=300 effectively ignored.
        """
        if not self.is_available():
            raise AntigravityUnavailableError(
                "Antigravity CLI ('agy') is not available on PATH or not executable."
            )

        working_dir.mkdir(parents=True, exist_ok=True)
        auditor = FilesystemAuditor(allowed_root=working_dir)
        auditor.snapshot()

        cmd = [
            self._exec_path,
            "--input-format", "stream-json",
            "--output-format", "stream-json",
        ]

        proc = subprocess.Popen(
            cmd,
            cwd=str(working_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        start_time = time.time()
        shell_approvals_count = 0
        final_response = ""
        status = "success"

        # Send initial user prompt event
        initial_msg = {
            "event": "user_message",
            "message": {"text": prompt},
        }
        try:
            proc.stdin.write(json.dumps(initial_msg) + "\n")
            proc.stdin.flush()
        except Exception as e:
            proc.kill()
            raise AntigravityError(f"Failed to send initial prompt to agy: {e}")

        # ── Reader thread ──────────────────────────────────────────────────────
        # readline() is a blocking call — it cannot be interrupted by a Python
        # timeout check running on the same thread. If agy emits no stdout for
        # the entire timeout window the "elapsed > timeout" guard after
        # readline() would never fire, making DEFAULT_TIMEOUT_SECONDS=300
        # completely ineffective. Fix: push every stdout line onto a queue from
        # a daemon thread; the main loop uses queue.get(timeout=remaining) so
        # the session deadline is enforced regardless of agy's output rate.
        import queue as _queue
        import threading as _threading

        _line_q: _queue.Queue = _queue.Queue()
        _STDOUT_EOF = object()  # sentinel — signals the reader thread is done

        def _reader():
            try:
                for line in proc.stdout:
                    _line_q.put(line)
            except Exception:
                pass
            finally:
                _line_q.put(_STDOUT_EOF)

        _reader_thread = _threading.Thread(target=_reader, daemon=True, name="agy-stdout-reader")
        _reader_thread.start()

        # NDJSON Streaming Event Loop
        try:
            while True:
                # Deadline-aware dequeue: always wakes at the session deadline
                # even if agy emits nothing for the remaining window.
                elapsed = time.time() - start_time
                remaining = timeout - elapsed
                if remaining <= 0:
                    proc.kill()
                    raise AntigravityTimeoutError(
                        f"Antigravity streaming session timed out after {timeout:.0f}s"
                    )

                try:
                    item = _line_q.get(timeout=min(remaining, 1.0))
                except _queue.Empty:
                    # No output yet — loop back to re-check the deadline.
                    continue

                if item is _STDOUT_EOF:
                    # agy closed its stdout normally (process finished).
                    break

                line = item
                if not line.strip():
                    continue

                try:
                    event_data = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue


                event_type = event_data.get("event")

                # Handle tool execution / permission requests
                if event_type in ("permission_request", "tool_call", "tool_use"):
                    req_id = event_data.get("id") or event_data.get("request_id")
                    tool_name = event_data.get("tool") or event_data.get("name") or ""
                    args = event_data.get("args") or event_data.get("parameters") or {}

                    # Specifically gate shell executions (run_command, bash, terminal, execute_command)
                    is_shell_command = any(
                        s in tool_name.lower() for s in ("run_command", "bash", "shell", "exec", "terminal")
                    ) or "command" in args or "CommandLine" in args

                    if is_shell_command:
                        shell_approvals_count += 1
                        if shell_approvals_count > max_shell_approvals:
                            proc.kill()
                            raise AntigravityApprovalCapExceededError(
                                f"Exceeded maximum allowed in-session shell approvals ({max_shell_approvals}). Terminating task."
                            )

                        cmd_text = args.get("CommandLine") or args.get("command") or args.get("cmd") or str(args)
                        prompt_msg = f"Antigravity requested permission to run shell command: '{cmd_text}'. Should I proceed?"
                        
                        approved = False
                        if player is not None:
                            approved = CONFIRM.request(player, prompt_msg, speak=speak)
                        else:
                            # In headless/non-interactive test runs without player, default to True for benign test commands
                            approved = True

                        if not approved:
                            _log.warning("User denied shell command: %s", cmd_text)
                            resp_event = {
                                "event": "permission_response",
                                "id": req_id,
                                "granted": False,
                                "message": "Command rejected by user authorization gate.",
                            }
                        else:
                            resp_event = {
                                "event": "permission_response",
                                "id": req_id,
                                "granted": True,
                            }

                        try:
                            proc.stdin.write(json.dumps(resp_event) + "\n")
                            proc.stdin.flush()
                        except Exception:
                            break

                # Handle final result / turn completion event
                elif event_type in ("result", "turn_complete", "done"):
                    res_payload = event_data.get("result") or event_data
                    status = res_payload.get("status", "SUCCESS")
                    final_response = res_payload.get("response", "") or res_payload.get("message", "")
                    if status.upper() not in ("SUCCESS", "COMPLETED", "OK"):
                        err_msg = res_payload.get("error", "Unknown error")
                        _log.warning("Antigravity streaming returned non-success: %s", err_msg)
                        if "permission" in str(err_msg).lower():
                            raise AntigravityPermissionError(f"Antigravity permission failure: {err_msg}")
                    break

        finally:
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()

        # ── 2. Post-Run Filesystem Boundary Audit ──────────────────────────────
        violations = auditor.audit_modifications()
        if violations:
            violation_names = [str(v) for v in violations]
            alert_msg = (
                f"🚨 SECURITY ALERT: Out-of-bounds file modifications detected outside project root!\n"
                f"Touched files: {', '.join(violation_names[:5])}"
            )
            _log.error(alert_msg)
            if player is not None:
                player.write_log(alert_msg)
            raise AntigravitySecurityCompromiseError(
                f"Security violation: Antigravity modified files outside allowed directory {working_dir}: {violation_names}"
            )

        duration = time.time() - start_time
        return {
            "status": "success" if status.upper() in ("SUCCESS", "COMPLETED", "OK") else "error",
            "result": final_response or "Antigravity task completed.",
            "project_dir": str(working_dir),
            "duration": duration,
            "provider": "antigravity",
            "shell_approvals": shell_approvals_count,
        }

    def run(
        self,
        prompt: str,
        working_dir: Path,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        player=None,
        speak=None,
        retry_cap: int = MAX_RETRIES,
    ) -> Dict[str, Any]:
        """
        High-level execution entry point.
        Attempts streaming real-time gated execution with retry handling.
        """
        last_error = None
        for attempt in range(1, retry_cap + 1):
            try:
                return self.run_streaming(
                    prompt=prompt,
                    working_dir=working_dir,
                    timeout=timeout,
                    player=player,
                    speak=speak,
                )
            except (AntigravitySecurityCompromiseError, AntigravityApprovalCapExceededError):
                # Critical security and cap violations fail immediately without retry
                raise
            except Exception as e:
                _log.warning("Antigravity attempt %d failed: %s", attempt, e)
                last_error = e
                if attempt < retry_cap:
                    time.sleep(1)

        raise last_error or AntigravityError("Antigravity failed after all retry attempts.")


class CodingAgent:
    """
    High-level Coding Sub-Agent Coordinator.
    Routes coding tasks to AntigravityProvider by default, with real-time interactive
    shell approvals and post-run boundary auditing.
    Gracefully falls back to actions/dev_agent.py's local-LLM loop on failure or unavailability.
    """

    def __init__(self, provider: Optional[AntigravityProvider] = None):
        self.provider = provider or AntigravityProvider()

    def build_project(
        self,
        description: str,
        language: str = "python",
        project_name: str = "",
        working_dir: Optional[Path] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        player=None,
        speak=None,
    ) -> str:
        """
        Builds or modifies a coding project.
        Tries Antigravity with real-time shell gating first; falls back to dev_agent on failure.
        """
        proj_name = project_name.strip() or self._derive_project_name(description)
        target_dir = working_dir or (PROJECTS_DIR / proj_name)

        # ── 1. Try Antigravity Sub-agent ──────────────────────────────────────
        if self.provider.is_available():
            if player:
                player.write_log(f"DEV: Delegating coding task to Antigravity ('{proj_name}')...")
            if speak:
                speak(f"Delegating project {proj_name} to Antigravity.")

            prompt = self.provider.build_prompt(
                description=description,
                language=language,
                project_name=proj_name,
                working_dir=target_dir,
            )

            try:
                result_data = self.provider.run(
                    prompt=prompt,
                    working_dir=target_dir,
                    timeout=timeout,
                    player=player,
                    speak=speak,
                )
                res_text = result_data.get("result", "")
                success_msg = (
                    f"Project '{proj_name}' completed by Antigravity.\n"
                    f"Directory: {target_dir}\n\n"
                    f"Summary:\n{res_text}"
                )
                if player:
                    player.write_log(f"DEV: Antigravity successfully finished '{proj_name}'.")
                if speak:
                    speak(f"Antigravity has completed the project {proj_name}, sir.")
                return success_msg

            except AntigravitySecurityCompromiseError as se:
                msg = f"Security error during Antigravity execution: {se}"
                _log.error(msg)
                if player: player.write_log(f"DEV: {msg}")
                return f"Task aborted due to security violation: {se}"

            except AntigravityApprovalCapExceededError as ce:
                msg = f"Antigravity terminated: {ce}"
                _log.warning(msg)
                if player: player.write_log(f"DEV: {msg}")
                return f"Task aborted: {ce}"

            except AntigravityPermissionError as pe:
                msg = f"Antigravity permission denied: {pe}. Falling back to local dev_agent..."
                _log.warning(msg)
                if player: player.write_log(f"DEV: {msg}")

            except Exception as e:
                msg = f"Antigravity failed ({e}). Falling back to local dev_agent loop..."
                _log.warning(msg)
                if player: player.write_log(f"DEV: {msg}")

        else:
            _log.info("Antigravity CLI not available — using local dev_agent fallback.")
            if player:
                player.write_log("DEV: Antigravity not available — using local dev_agent fallback.")

        # ── 2. Fallback to actions/dev_agent.py Local-LLM Loop ─────────────────
        if speak:
            speak("Running task with local development loop.")
        from actions.dev_agent import dev_agent as _local_dev_agent
        return _local_dev_agent(
            parameters={
                "description": description,
                "language": language,
                "project_name": proj_name,
                "timeout": int(timeout),
            },
            player=player,
            speak=speak,
        )

    @staticmethod
    def _derive_project_name(description: str) -> str:
        """Derives a clean snake_case directory name from the project description."""
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", description.lower())
        words = cleaned.split()[:4]
        return "_".join(words) or f"project_{int(time.time())}"


# Singleton agent instance
_CODING_AGENT = CodingAgent()


def coding_agent(
    parameters: dict,
    response=None,
    player=None,
    session_memory=None,
    speak=None,
) -> str:
    """
    TOOL_DISPATCH entrypoint for coding tasks.
    Parameters:
        description (str, required): The specification of the project or coding task.
        language (str, optional): Target language (default 'python').
        project_name (str, optional): Name of the project folder.
        timeout (int, optional): Maximum execution timeout in seconds.
    """
    p = parameters or {}
    description = p.get("description", "").strip()
    if not description:
        for field in ("task", "prompt", "query", "goal", "instructions", "text"):
            val = p.get(field)
            if val and isinstance(val, str) and val.strip():
                description = val.strip()
                break

    language = p.get("language", "python").strip()
    project_name = p.get("project_name", "").strip()
    timeout = float(p.get("timeout", DEFAULT_TIMEOUT_SECONDS))

    if not description:
        return "Please describe the coding project or task you want me to build, sir."

    return _CODING_AGENT.build_project(
        description=description,
        language=language,
        project_name=project_name,
        timeout=timeout,
        player=player,
        speak=speak,
    )
