"""
Local LLM client for MARK XL.

Supports two backends — selected via  "llm_provider"  in config/api_keys.json:

  "llm_provider": "ollama"   (default)
        Uses Ollama's native /api/chat endpoint.
        Download: https://ollama.com
        Default port: 11434

  "llm_provider": "openai"
        Uses any OpenAI-compatible server: LM Studio, Jan, LocalAI,
        llama.cpp server, vLLM, etc.
        LM Studio download: https://lmstudio.ai   (default port: 1234)
        Set  "llm_url": "http://localhost:1234"  in config.
        Note: tool-calling support depends on the model; use a model that
        supports function/tool calls (e.g. Qwen2.5, Llama-3.1, Mistral).
"""
import json
import logging
import re
import subprocess
import sys
import time
from typing import Generator

import requests

from config import load_config as _load_config
from core.db import log_task_event

_log = logging.getLogger("jarvis.llm")

# Matches a sentence boundary: [.!?] followed by whitespace, or a blank line.
# Avoids splitting on decimals (3.5) because those have no space after the dot.
_SENT_END = re.compile(r'(?<=[.!?])\s+|(?<=\n)\s*\n')

_DEFAULTS = {
    "llm_url":      "http://localhost:11434",
    "llm_model":    "llama3.2",
    "llm_provider": "ollama",   # "ollama" | "openai"
}


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def _log_llm_outcome(
    task_id:     str | None,
    step_num,
    purpose:     str,
    status:      str,
    duration_ms: int,
    detail:      str = "",
) -> None:
    """Records one LLM call's outcome. Callers with a task_id in scope
    (planner/executor/error_handler — all task-scoped) land in
    task_events as tool='llm', alongside the step timeline. Callers with
    no task_id (the live conversation loop, and any LLM call made from
    inside a TOOL_DISPATCH tool — task_id isn't threaded through tool
    calls in this phase, see the logging-phase plan) land in log_events
    instead, via the general logger."""
    if task_id:
        log_task_event(task_id, step_num, "llm", purpose, status, detail, duration_ms)
        return
    extra = {"duration_ms": duration_ms}
    if detail:
        extra["detail"] = detail
    if status == "failed":
        _log.error("LLM call failed: %s", purpose, extra=extra)
    else:
        _log.info("LLM call: %s", purpose, extra=extra)


def get_llm_provider() -> str:
    """Returns 'ollama' or 'openai' (covers LM Studio, LocalAI, Jan, etc.)."""
    raw = _load_config().get("llm_provider", "ollama").strip().lower()
    return "openai" if raw in ("openai", "lmstudio", "localai", "jan", "llamacpp") else "ollama"


def ensure_ollama_running(timeout: int = 15) -> bool:
    """
    For Ollama: ping /api/tags; auto-launch 'ollama serve' if not running.
    For OpenAI-compatible providers: just ping /v1/models (server must be started manually).
    Returns True if the LLM server is reachable.
    """
    url, _   = get_llm_settings()
    provider = get_llm_provider()

    if provider == "openai":
        # OpenAI-compatible servers (LM Studio, LocalAI, etc.) must be started
        # by the user — we just check if they're reachable.
        health = f"{url}/v1/models"
        try:
            ok = requests.get(health, timeout=5).status_code == 200
            if ok:
                _log.info("OpenAI-compatible server reachable at %s", url)
            else:
                _log.warning("Server at %s returned non-200. Is it running?", url)
            return ok
        except Exception as e:
            _log.warning(
                "Cannot reach OpenAI-compatible server at %s — make sure LM Studio / "
                "LocalAI / Jan is running and the server is started: %s", url, e,
            )
            return False

    # ── Ollama ──────────────────────────────────────────────────────────────
    health = f"{url}/api/tags"

    def _is_up() -> bool:
        try:
            return requests.get(health, timeout=3).status_code == 200
        except Exception:
            return False

    if _is_up():
        return True

    _log.info("Ollama not running — launching 'ollama serve'…")
    try:
        kwargs: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        subprocess.Popen(["ollama", "serve"], **kwargs)
    except FileNotFoundError:
        _log.error("'ollama' command not found. Install Ollama from https://ollama.com")
        return False
    except Exception as e:
        _log.error("Could not launch Ollama: %s", e)
        return False

    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(1.0)
        if _is_up():
            _log.info("Ollama started successfully.")
            return True

    _log.error("Ollama did not respond within the timeout.")
    return False


def warmup_model(system_prompt: str | None = None) -> bool:
    """
    Pre-load the model AND prime Ollama's KV prefix cache.

    Why the system_prompt matters
    ─────────────────────────────
    Ollama caches the KV attention state of the prompt prefix across requests.
    If warmup includes the same system prompt that real requests will use, Ollama
    evaluates those tokens ONCE at startup.  Every subsequent request only needs
    to evaluate the small delta (user message ± time context) instead of the full
    300-500 token system prompt → drops first-token latency from ~17 s to <1 s.

    Pass the *static* part of the system prompt (the JARVIS protocol text, without
    timestamps or per-minute context) so the prefix stays valid across calls.
    """
    url, model = get_llm_settings()
    provider   = get_llm_provider()
    _log.info("Warming up '%s' (%s)…", model, provider)

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": "hi"})

    if provider == "openai":
        # OpenAI-compatible: just fire a minimal request to ensure the model is loaded.
        # No keep_alive or KV-cache priming available — server manages this internally.
        payload = {
            "model":      model,
            "messages":   messages,
            "stream":     False,
            "max_tokens": 1,
        }
        try:
            resp = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=180)
            resp.raise_for_status()
            _log.info("'%s' ready (OpenAI-compatible server).", model)
            return True
        except Exception as e:
            _log.warning("Warmup failed (non-fatal): %s", e)
            return False

    # ── Ollama ──────────────────────────────────────────────────────────────
    payload = {
        "model":      model,
        "messages":   messages,
        "stream":     False,
        "keep_alive": -1,
        # num_gpu:99 → push ALL transformer layers to GPU (Ollama caps at available)
        # This is safe even without a GPU — Ollama silently ignores if n_gpu_layers=0
        "options":    {"num_predict": 1, "num_gpu": 99},
    }
    try:
        resp = requests.post(f"{url}/api/chat", json=payload, timeout=180)
        resp.raise_for_status()
        _log.info("'%s' loaded and KV cache primed.", model)
        return True
    except Exception as e:
        _log.warning("Warmup failed (non-fatal): %s", e)
        return False


def get_llm_settings() -> tuple[str, str]:
    """Returns (base_url, model_name)."""
    cfg   = _load_config()
    url   = cfg.get("llm_url",   _DEFAULTS["llm_url"]).rstrip("/")
    model = cfg.get("llm_model", _DEFAULTS["llm_model"])
    return url, model


def call_llm(
    messages: list,
    tools:    list | None = None,
    timeout:  int = 120,
    task_id:  str | None = None,
    step_num                = None,
    purpose:  str = "chat",
) -> dict:
    """
    Non-streaming chat request.  Routes to Ollama or OpenAI-compatible backend.
    Pass task_id/step_num when calling from inside a task-scoped context
    (AgentExecutor et al.) so the call's outcome+duration lands in
    task_events; otherwise it lands in log_events with task_id=NULL.

    Returns:
        {"content": str, "tool_calls": list}
    """
    start = time.monotonic()
    try:
        result = _call_llm_impl(messages, tools, timeout)
        _log_llm_outcome(task_id, step_num, purpose, "done", _elapsed_ms(start))
        return result
    except Exception as e:
        _log_llm_outcome(task_id, step_num, purpose, "failed", _elapsed_ms(start), str(e))
        raise


def _call_llm_impl(
    messages: list,
    tools:    list | None = None,
    timeout:  int = 120,
) -> dict:
    url, model = get_llm_settings()
    provider   = get_llm_provider()

    if provider == "openai":
        endpoint = f"{url}/v1/chat/completions"
        # 150 tokens is sized for a spoken reply; a tool call still has to fit
        # its name + JSON arguments in that same budget, so give tool-bearing
        # requests more headroom (500) to avoid truncating mid-JSON.
        payload: dict = {
            "model":      model,
            "messages":   messages,
            "stream":     False,
            "max_tokens": 500 if tools else 150,
        }
        if tools:
            payload["tools"]       = tools
            payload["tool_choice"] = "auto"
        try:
            resp = requests.post(endpoint, json=payload, timeout=timeout)
            resp.raise_for_status()
            choice = resp.json().get("choices", [{}])[0]
            msg    = choice.get("message", {})
            # OpenAI tool_calls format → normalise to Ollama-style
            raw_tc  = msg.get("tool_calls") or []
            tc_list = [
                {
                    "id":       t.get("id", ""),
                    "function": {
                        "name":      t["function"]["name"],
                        "arguments": (
                            json.loads(t["function"]["arguments"])
                            if isinstance(t["function"].get("arguments"), str)
                            else t["function"].get("arguments", {})
                        ),
                    },
                }
                for t in raw_tc
            ]
            return {
                "content":    (msg.get("content") or "").strip(),
                "tool_calls": tc_list,
            }
        except Exception as e:
            raise RuntimeError(f"OpenAI-compatible LLM call failed: {e}")

    # ── Ollama ──────────────────────────────────────────────────────────────
    endpoint = f"{url}/api/chat"
    # 150 tokens is sized for a spoken reply; a tool call still has to fit
    # its name + JSON arguments in that same budget, so give tool-bearing
    # requests more headroom (500) to avoid truncating mid-JSON.
    payload = {
        "model":      model,
        "messages":   messages,
        "stream":     False,
        "keep_alive": -1,
        "options":    {"num_predict": 500 if tools else 150, "num_gpu": 99},
    }
    if tools:
        payload["tools"] = tools

    try:
        resp = requests.post(endpoint, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        msg  = data.get("message", {})
        return {
            "content":    (msg.get("content") or "").strip(),
            "tool_calls": msg.get("tool_calls") or [],
        }
    except requests.exceptions.ConnectionError as e:
        _log.warning("ConnectionError — trying to restart Ollama… (%s)", e)
        if ensure_ollama_running():
            try:
                resp = requests.post(endpoint, json=payload, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
                msg  = data.get("message", {})
                return {
                    "content":    (msg.get("content") or "").strip(),
                    "tool_calls": msg.get("tool_calls") or [],
                }
            except Exception:
                pass
        raise RuntimeError(
            f"Cannot connect to Ollama at {url}. "
            "Make sure Ollama is installed and run: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama request timed out after 120 s.")
    except requests.exceptions.HTTPError as e:
        _log.error("HTTPError: %s — %s", e.response.status_code, e.response.text[:200])
        raise RuntimeError(f"Ollama HTTP error: {e.response.status_code}")
    except Exception as e:
        _log.error("Unexpected error: %s: %s", type(e).__name__, e)
        raise RuntimeError(f"LLM call failed: {e}")


def call_llm_text(
    prompt:  str,
    system:  str | None = None,
    model:   str | None = None,
    timeout: int = 120,
    task_id: str | None = None,
    step_num             = None,
    purpose: str = "text generation",
) -> str:
    """
    Simple text-only generation (no tools).
    Used by planner, executor, error_handler, code_helper, dev_agent.
    Pass task_id/step_num when calling from a task-scoped context — see
    call_llm's docstring.
    """
    start = time.monotonic()
    try:
        result = _call_llm_text_impl(prompt, system, model, timeout)
        _log_llm_outcome(task_id, step_num, purpose, "done", _elapsed_ms(start))
        return result
    except Exception as e:
        _log_llm_outcome(task_id, step_num, purpose, "failed", _elapsed_ms(start), str(e))
        raise


def _call_llm_text_impl(
    prompt:  str,
    system:  str | None = None,
    model:   str | None = None,
    timeout: int = 120,
) -> str:
    url, default_model = get_llm_settings()
    endpoint = f"{url}/api/chat"
    m        = model or default_model

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {"model": m, "messages": messages, "stream": False, "keep_alive": -1, "options": {"num_predict": 600}}

    try:
        resp = requests.post(endpoint, json=payload, timeout=timeout)
        resp.raise_for_status()
        return (resp.json().get("message", {}).get("content") or "").strip()
    except requests.exceptions.ConnectionError:
        if ensure_ollama_running():
            try:
                resp = requests.post(endpoint, json=payload, timeout=timeout)
                resp.raise_for_status()
                return (resp.json().get("message", {}).get("content") or "").strip()
            except Exception:
                pass
        raise RuntimeError(
            f"Cannot connect to Ollama at {url}. "
            "Make sure Ollama is installed and run: ollama serve"
        )
    except Exception as e:
        raise RuntimeError(f"LLM text call failed: {e}")


def call_llm_vision(
    prompt:      str,
    image_bytes: bytes,
    system:      str | None = None,
    model:       str | None = None,
    timeout:     int = 120,
    task_id:     str | None = None,
    step_num                 = None,
    purpose:     str = "vision",
) -> str:
    """
    Single-image vision generation via Ollama's /api/chat "images" field
    (base64-encoded). Requires a vision-capable model — config key
    "vision_model", default "qwen2.5vl:7b". Ollama-only: this codebase has
    no OpenAI-compatible vision path configured. Pass task_id/step_num
    when calling from a task-scoped context — see call_llm's docstring.
    """
    start = time.monotonic()
    try:
        result = _call_llm_vision_impl(prompt, image_bytes, system, model, timeout)
        _log_llm_outcome(task_id, step_num, purpose, "done", _elapsed_ms(start))
        return result
    except Exception as e:
        _log_llm_outcome(task_id, step_num, purpose, "failed", _elapsed_ms(start), str(e))
        raise


def _call_llm_vision_impl(
    prompt:      str,
    image_bytes: bytes,
    system:      str | None = None,
    model:       str | None = None,
    timeout:     int = 120,
) -> str:
    import base64
    cfg = _load_config()
    url = cfg.get("llm_url", _DEFAULTS["llm_url"]).rstrip("/")
    m   = model or cfg.get("vision_model") or "qwen2.5vl:7b"
    b64 = base64.b64encode(image_bytes).decode("ascii")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt, "images": [b64]})

    payload = {
        "model":      m,
        "stream":     False,
        "keep_alive": -1,
        "messages":   messages,
    }
    try:
        resp = requests.post(f"{url}/api/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
        return (resp.json().get("message", {}).get("content") or "").strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to Ollama at {url}. "
            "Make sure Ollama is installed and run: ollama serve"
        )
    except Exception as e:
        raise RuntimeError(f"LLM vision call failed: {e}")


def _stream_openai(
    messages: list,
    tools:    list | None,
    timeout:  int,
) -> Generator[dict, None, None]:
    """
    Streaming backend for OpenAI-compatible servers (LM Studio, LocalAI, Jan…).

    Parses Server-Sent Events (SSE) and accumulates streaming tool-call fragments
    so the output format is identical to the Ollama backend.
    """
    url, model = get_llm_settings()
    endpoint   = f"{url}/v1/chat/completions"

    # 150 tokens is sized for a spoken reply; a tool call still has to fit
    # its name + JSON arguments in that same budget, so give tool-bearing
    # requests more headroom (500) to avoid truncating mid-JSON.
    payload: dict = {
        "model":      model,
        "messages":   messages,
        "stream":     True,
        "max_tokens": 500 if tools else 150,
    }
    if tools:
        payload["tools"]       = tools
        payload["tool_choice"] = "auto"

    try:
        with requests.post(endpoint, json=payload, timeout=timeout, stream=True) as resp:
            resp.raise_for_status()
            full_content = ""
            buf          = ""
            # tool_call fragments: index → {"id", "function": {"name", "arguments"}}
            tc_fragments: dict[int, dict] = {}

            for raw in resp.iter_lines():
                if not raw:
                    continue
                # SSE lines look like: b"data: {...}" or b"data: [DONE]"
                line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choice = chunk.get("choices", [{}])[0]
                delta  = choice.get("delta", {})
                text   = delta.get("content") or ""

                full_content += text
                buf          += text

                # Accumulate sentence boundaries for streaming TTS
                while True:
                    m = _SENT_END.search(buf)
                    if not m:
                        break
                    sentence = buf[: m.start() + 1].strip()
                    buf      = buf[m.end():]
                    if sentence:
                        yield {"type": "sentence", "text": sentence}

                # Accumulate streaming tool-call fragments
                for tc in (delta.get("tool_calls") or []):
                    idx = tc.get("index", 0)
                    if idx not in tc_fragments:
                        tc_fragments[idx] = {"id": "", "function": {"name": "", "arguments": ""}}
                    frag = tc_fragments[idx]
                    frag["id"] = frag["id"] or tc.get("id", "")
                    fn = tc.get("function", {})
                    frag["function"]["name"]      += fn.get("name") or ""
                    frag["function"]["arguments"] += fn.get("arguments") or ""

                finish = choice.get("finish_reason")
                if finish in ("stop", "tool_calls", "length"):
                    break

            # Flush any trailing content
            if buf.strip():
                yield {"type": "sentence", "text": buf.strip()}

            # Parse accumulated tool-call argument strings → dicts
            tool_calls: list = []
            for idx in sorted(tc_fragments):
                frag = tc_fragments[idx]
                args = frag["function"]["arguments"]
                try:
                    args = json.loads(args)
                except Exception:
                    pass   # leave as raw string; _execute_tool handles it
                tool_calls.append({
                    "id":       frag["id"],
                    "function": {"name": frag["function"]["name"], "arguments": args},
                })

            yield {
                "type":       "done",
                "content":    full_content.strip(),
                "tool_calls": tool_calls,
            }

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot reach OpenAI-compatible server at {url}.\n"
            "Make sure LM Studio / LocalAI / Jan is running and the server is started."
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("OpenAI-compatible stream timed out.")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"OpenAI-compatible HTTP error: {e.response.status_code}")
    except Exception as e:
        raise RuntimeError(f"OpenAI-compatible stream failed: {e}")


def call_llm_stream(
    messages: list,
    tools:    list | None = None,
    timeout:  int = 120,
    task_id:  str | None = None,
    step_num                = None,
    purpose:  str = "chat stream",
) -> Generator[dict, None, None]:
    """
    Streaming chat request.  Routes to Ollama or OpenAI-compatible backend.
    Pass task_id/step_num when calling from a task-scoped context — see
    call_llm's docstring. The vast majority of calls to this function come
    from main.py's live conversation loop, which has no task_id at all —
    those log to log_events instead.

    Yields:
        {"type": "sentence", "text": str}   — each complete sentence as it arrives
        {"type": "done", "content": str, "tool_calls": list}  — when stream ends

    Sentences are split on [.!?] + whitespace so TTS can start immediately.
    Tool calls always appear in the final "done" event.
    """
    start = time.monotonic()
    try:
        for event in _call_llm_stream_impl(messages, tools, timeout):
            yield event
        _log_llm_outcome(task_id, step_num, purpose, "done", _elapsed_ms(start))
    except Exception as e:
        _log_llm_outcome(task_id, step_num, purpose, "failed", _elapsed_ms(start), str(e))
        raise


def _call_llm_stream_impl(
    messages: list,
    tools:    list | None = None,
    timeout:  int = 120,
) -> Generator[dict, None, None]:
    provider = get_llm_provider()
    if provider == "openai":
        yield from _stream_openai(messages, tools, timeout)
        return

    url, model = get_llm_settings()
    endpoint   = f"{url}/api/chat"

    payload: dict = {
        "model":      model,
        "messages":   messages,
        "stream":     True,
        "keep_alive": -1,
        # 150 tokens ≈ 100 words ≈ 3-4 sentences — enough for any voice reply
        # with no tools attached. But a tool call has to fit its name + JSON
        # arguments in that same budget, so tool-bearing requests get 500
        # instead — otherwise a multi-tool call or one with a long argument
        # (e.g. file_controller content, code_helper's code param) truncates
        # mid-JSON and fails to parse. Most turns still stop well short of
        # 500 via natural stop tokens — this is a ceiling, not a target.
        # num_gpu:99 pushes all layers to GPU; num_thread removed (Ollama auto-tunes).
        "options":    {"num_predict": 500 if tools else 150, "num_gpu": 99},
    }
    if tools:
        payload["tools"] = tools

    def _do_stream() -> Generator[dict, None, None]:
        with requests.post(endpoint, json=payload, timeout=timeout, stream=True) as resp:
            resp.raise_for_status()
            full_content = ""
            tool_calls:  list = []
            buf          = ""

            for raw in resp.iter_lines():
                if not raw:
                    continue
                try:
                    chunk = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                msg   = chunk.get("message", {})
                delta = msg.get("content") or ""

                full_content += delta
                buf          += delta

                # Yield complete sentences as they accumulate
                while True:
                    m = _SENT_END.search(buf)
                    if not m:
                        break
                    sentence = buf[: m.start() + 1].strip()
                    buf      = buf[m.end() :]
                    if sentence:
                        yield {"type": "sentence", "text": sentence}

                tc = msg.get("tool_calls")
                if tc:
                    tool_calls.extend(tc)

                if chunk.get("done"):
                    if buf.strip():
                        yield {"type": "sentence", "text": buf.strip()}

                    yield {
                        "type":       "done",
                        "content":    full_content.strip(),
                        "tool_calls": tool_calls,
                    }
                    return

    try:
        yield from _do_stream()
    except requests.exceptions.ConnectionError as e:
        _log.warning("Stream ConnectionError — trying to restart Ollama… (%s)", e)
        if ensure_ollama_running():
            yield from _do_stream()
            return
        raise RuntimeError(
            f"Cannot connect to Ollama at {url}. "
            "Make sure Ollama is installed and run: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama stream timed out.")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Ollama HTTP error: {e.response.status_code}")
    except Exception as e:
        _log.error("Stream error: %s: %s", type(e).__name__, e)
        raise RuntimeError(f"LLM stream failed: {e}")
