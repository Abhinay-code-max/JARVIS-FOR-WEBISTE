"""
core/logging_setup.py
======================
Structured logging for JARVIS-XL. One "jarvis" logger hierarchy replaces
the print() calls that used to be scattered app-wide; module loggers are
named jarvis.<tag> mirroring the old print prefixes 1:1 (e.g. the old
"[LLM] ..." prints now come from logging.getLogger("jarvis.llm")).

This is deliberately separate from task_events (core.db.log_task_event):
task_events is the durable, task-scoped step ledger written directly and
synchronously from AgentExecutor's own worker thread. This module is the
general sink for everything else — startup, config, wake-word decisions,
STT errors, LLM calls with no task_id — persisted to log_events.

Non-blocking guarantee
-----------------------
core/db.py's hard rule says CONFIRM.answer(), _transcribe_and_enqueue, and
_on_transcript (the audio callback and Deepgram-recv threads) must never
block on DB I/O — and this phase's brief widens that to "not even an
unbuffered stdout flush under contention". So NEITHER handler is attached
directly to the "jarvis" logger. The only handler on "jarvis" is a
QueueHandler: every logger.*() call anywhere in the app — including from
inside the PortAudio callback — only ever does an in-memory queue.put(),
identical in cost to the producer/consumer queues (_text_queue, _tts_queue)
those same threads already push through. A single dedicated
QueueListener thread is the only thing that ever runs the console
handler's write or the DB handler's INSERT.
"""
from __future__ import annotations

import logging
import logging.handlers
import queue

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)-20s %(message)s"

_listener: logging.handlers.QueueListener | None = None


class SQLiteHandler(logging.Handler):
    """Writes one row per record into log_events. Only ever invoked from
    the QueueListener's own dedicated thread (see init_logging()) — never
    attach this handler directly to a logger."""

    _exc_formatter = logging.Formatter()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            detail = getattr(record, "detail", None)
            if detail is None and record.exc_info:
                # logger.error(..., exc_info=True) callers (the old
                # traceback.print_exc() call sites) get the traceback
                # persisted here instead of only ever reaching a console
                # someone happened to be watching.
                detail = self._exc_formatter.formatException(record.exc_info)

            from core.db import get_conn  # lazy: avoid importing db at module load
            conn = get_conn()
            with conn:
                conn.execute(
                    "INSERT INTO log_events "
                    "(task_id, level, source, message, detail, duration_ms, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        getattr(record, "task_id", None),
                        record.levelname,
                        record.name,
                        record.getMessage(),
                        detail,
                        getattr(record, "duration_ms", None),
                        record.created,
                    ),
                )
        except Exception:
            # A logging failure must never crash the app or recurse back
            # into logging — drop it, same as every other best-effort DB
            # write in this codebase (see core/db.py).
            pass


def init_logging(level: int = logging.INFO) -> None:
    """Call once, as early as possible in main.py — safe to call before
    _bootstrap() since this only needs stdlib. Idempotent."""
    global _listener
    root = logging.getLogger("jarvis")
    if root.handlers:
        return

    root.setLevel(level)
    root.propagate = False

    log_queue: queue.Queue = queue.Queue(-1)  # unbounded — never Full, never drops
    root.addHandler(logging.handlers.QueueHandler(log_queue))

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_LOG_FORMAT))

    sqlite_handler = SQLiteHandler()

    _listener = logging.handlers.QueueListener(
        log_queue, console, sqlite_handler, respect_handler_level=False
    )
    _listener.start()


def shutdown_logging() -> None:
    """Flush and stop the listener thread. Not called from os._exit() paths
    (shutdown_jarvis, hard KeyboardInterrupt) — those bypass cleanup
    entirely, same as everything else in this app; a handful of buffered
    records lost on a hard exit is an acceptable trade for keeping every
    call site non-blocking."""
    global _listener
    if _listener is not None:
        _listener.stop()
        _listener = None
