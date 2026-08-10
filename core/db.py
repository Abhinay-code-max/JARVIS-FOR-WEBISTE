"""
core/db.py
==========
SQLite persistence layer for JARVIS-XL (tasks, task_events, memory_entries,
approvals, contacts). Single-user local app — one .db file, no ORM, no
network DB.

Connection model: one real sqlite3.Connection per thread (threading.local()),
never shared across threads, so check_same_thread stays at its default
(True) everywhere — there's nothing risky to opt out of. WAL mode lets
concurrent readers proceed without blocking behind a writer; busy_timeout
makes a genuinely-concurrent writer (e.g. two BEGIN IMMEDIATE transactions
racing) wait and retry instead of raising "database is locked".

Hard rule (see the persistence-plan investigation): never call into this
module — or anything that calls into it — from CONFIRM.answer(),
_transcribe_and_enqueue, or _on_transcript. Those run on the audio/
transcript threads and must stay free of blocking disk I/O.
"""
from __future__ import annotations
import logging
import sqlite3
import threading
import time

from config import BASE_DIR

_log = logging.getLogger("jarvis.db")

DB_DIR  = BASE_DIR / "data"
DB_PATH = DB_DIR / "jarvis.db"

_local = threading.local()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id     TEXT PRIMARY KEY,
    goal        TEXT NOT NULL,
    priority    INTEGER NOT NULL,
    status      TEXT NOT NULL,
    result      TEXT,
    error       TEXT,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS task_events (
    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     TEXT NOT NULL REFERENCES tasks(task_id),
    step_num    INTEGER,
    tool        TEXT,
    description TEXT,
    status      TEXT,
    detail      TEXT,
    duration_ms INTEGER,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_task_events_task_id ON task_events(task_id);

-- General structured-logging sink: everything that isn't a task_events
-- step (startup, config load, wake-word gate decisions, STT errors, LLM
-- calls with no task_id, etc). task_id is nullable and deliberately NOT
-- a widened task_events — see the logging-phase investigation for why
-- (task_events' columns are shaped for a step ledger, not a general
-- level/source/message record).
CREATE TABLE IF NOT EXISTS log_events (
    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     TEXT REFERENCES tasks(task_id),
    level       TEXT NOT NULL,
    source      TEXT NOT NULL,
    message     TEXT NOT NULL,
    detail      TEXT,
    duration_ms INTEGER,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_log_events_task_id    ON log_events(task_id);
CREATE INDEX IF NOT EXISTS idx_log_events_created_at ON log_events(created_at);

-- Two stores as of the memory-layers phase (see memory_manager.py's
-- _store_for_category and _migrate_profile_categories below):
-- 'profile' (identity, relationships) — low-write-frequency, high-value,
-- near-permanent facts, exempt from the normal shared eviction budget
-- (its own separate, generous backstop cap instead — see
-- memory_manager.PROFILE_MAX_CHARS). 'default' (preferences, projects,
-- wishes, notes) — everything else, unchanged behavior from before this
-- phase (memory_manager.MEMORY_MAX_CHARS, oldest-updated_at-first
-- eviction). The `store` column predates the actual split — it was added
-- during the persistence phase specifically so this later split would
-- have somewhere to land without a schema migration; this phase is that
-- split.
CREATE TABLE IF NOT EXISTS memory_entries (
    store       TEXT NOT NULL DEFAULT 'default',
    category    TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (store, category, key)
);
CREATE INDEX IF NOT EXISTS idx_memory_updated ON memory_entries(store, updated_at);

-- outcome vocabulary: NULL (interactive CONFIRM.request() mid-flight,
-- answered_at not yet set — always resolves within its own timeout, never
-- left this way), 'approved' | 'denied' | 'timeout' (a resolved yes/no —
-- 'timeout' covers both CONFIRM's ~45s interactive wait and
-- core/task_approval.py's longer background wait elapsing while the app
-- was still live), 'pending' (core/task_approval.py's TASK_APPROVAL: a
-- background ask-and-wait step paused awaiting approve_task), 'expired'
-- (a 'pending' row reconciled at startup with no live worker thread left
-- to resume it — see agent/task_queue.py's _reconcile_orphaned_approvals;
-- distinct from 'timeout' because the wait itself never got the chance to
-- elapse), 'no_player' (core/confirm.py's own instant fail-closed path,
-- still reachable for the delegated tools in core/tool_gate.DELEGATED_TOOLS
-- when called with no live player).
CREATE TABLE IF NOT EXISTS approvals (
    approval_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt       TEXT NOT NULL,
    requested_at REAL NOT NULL,
    answered_at  REAL,
    outcome      TEXT,
    task_id      TEXT
);

CREATE TABLE IF NOT EXISTS contacts (
    alias      TEXT PRIMARY KEY,
    exact_name TEXT NOT NULL
);

-- tool_name -> default permission level (auto-allow | notify-only |
-- ask-and-wait | hard-deny). `action` is nullable: NULL is the tool-wide
-- default row, a specific value overrides it for tools split by action
-- (computer_control, browser_control, file_controller, computer_settings,
-- file_processor, game_updater — see core/policy.py). No UNIQUE/PK
-- constraint on (tool_name, action) because SQLite doesn't dedupe NULLs
-- there; core/policy.py's seed function is idempotent via a row-count
-- check instead (same pattern as the memory/contacts migrations above).
CREATE TABLE IF NOT EXISTS permission_policy (
    tool_name TEXT NOT NULL,
    action    TEXT,
    level     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_permission_policy_tool ON permission_policy(tool_name);

-- One row per TOOL_DISPATCH call (core/tool_gate.py's dispatch_tool),
-- regardless of the level it evaluated to — auto-allow and notify-only
-- included, not just the subset that reach a blocking confirmation.
-- `outcome` for delegated tools (dev_agent, vision_fix_code) is literally
-- 'delegated' — the real yes/no decision(s) for those live in `approvals`,
-- joined by task_id/time, because those tools gate mid-flow with
-- content-specific prompts a single dispatch-entry row can't capture.
CREATE TABLE IF NOT EXISTS policy_decisions (
    decision_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_name       TEXT NOT NULL,
    action          TEXT,
    level_evaluated TEXT NOT NULL,
    outcome         TEXT NOT NULL,
    task_id         TEXT,
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_policy_decisions_task_id ON policy_decisions(task_id);
"""


def get_conn() -> sqlite3.Connection:
    """Returns this thread's own sqlite3.Connection, creating and
    initializing it (schema + one-time legacy-data migration) on first use.
    Never pass the returned object to another thread."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        return conn

    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA)
    # CREATE TABLE IF NOT EXISTS above doesn't add columns to a
    # task_events table that already existed before duration_ms was
    # introduced — do that explicitly, idempotently.
    _ensure_column(conn, "task_events", "duration_ms", "INTEGER")
    conn.commit()
    _local.conn = conn

    _migrate_legacy_memory(conn)
    _migrate_legacy_contacts(conn)
    _migrate_profile_categories(conn)

    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, coltype: str) -> None:
    """Adds `column` to `table` if it isn't already there. `table`/`column`/
    `coltype` are always internal constants we control, never user input."""
    cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")


def init_db() -> None:
    """Idempotent — safe to call on every startup. Ensures the schema (and
    any pending one-time legacy-data import) exists on the calling thread;
    every other thread does the same lazily the first time it calls
    get_conn()."""
    get_conn()


def _migrate_legacy_memory(conn: sqlite3.Connection) -> None:
    """One-time import of memory/long_term.json into memory_entries.
    Idempotent: skips if memory_entries already has rows, or if
    long_term.json doesn't exist (fresh install, or already migrated and
    renamed). Reuses memory_manager's own _all_entries()/_empty_memory()
    so the import walks the exact same shape memory_manager itself
    produces — imported lazily to avoid a circular import at module load
    (memory_manager imports get_conn from this module)."""
    import json

    from memory.memory_manager import MEMORY_PATH, _all_entries, _empty_memory

    if not MEMORY_PATH.exists():
        return

    row_count = conn.execute("SELECT COUNT(*) FROM memory_entries").fetchone()[0]
    if row_count > 0:
        return

    try:
        raw = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        _log.warning("Memory migration: could not read %s: %s", MEMORY_PATH, e)
        return

    if not isinstance(raw, dict):
        return

    base = _empty_memory()
    for key in base:
        if key not in raw:
            raw[key] = {}

    entries = _all_entries(raw)
    if entries:
        with conn:
            for cat, key, entry in entries:
                value   = entry.get("value", "")
                updated = entry.get("updated", "")
                conn.execute(
                    "INSERT OR IGNORE INTO memory_entries "
                    "(store, category, key, value, updated_at) VALUES ('default', ?, ?, ?, ?)",
                    (cat, key, value, updated),
                )

    try:
        migrated_path = MEMORY_PATH.parent / (MEMORY_PATH.name + ".migrated")
        MEMORY_PATH.rename(migrated_path)
        _log.info("Migrated %d memory entries from %s -> %s", len(entries), MEMORY_PATH.name, migrated_path.name)
    except Exception as e:
        _log.warning("Migrated %d memory entries, but could not rename source file: %s", len(entries), e)


def _migrate_legacy_contacts(conn: sqlite3.Connection) -> None:
    """One-time import of config/contacts.json into the contacts table.
    Idempotent: skips if contacts already has rows, or the file is
    missing. Per the resolved plan, contacts.json itself is left
    untouched by this migration (see memory_manager notes on why the
    rename treatment differs from memory's)."""
    import json

    contacts_path = BASE_DIR / "config" / "contacts.json"
    if not contacts_path.exists():
        return

    row_count = conn.execute("SELECT COUNT(*) FROM contacts").fetchone()[0]
    if row_count > 0:
        return

    try:
        raw = json.loads(contacts_path.read_text(encoding="utf-8"))
    except Exception as e:
        _log.warning("Contacts migration: could not read %s: %s", contacts_path, e)
        return

    if not isinstance(raw, dict) or not raw:
        return

    with conn:
        for alias, exact_name in raw.items():
            conn.execute(
                "INSERT OR IGNORE INTO contacts (alias, exact_name) VALUES (?, ?)",
                (alias, exact_name),
            )
    _log.info("Migrated %d contact(s) from %s.", len(raw), contacts_path.name)


def _migrate_profile_categories(conn: sqlite3.Connection) -> None:
    """One-time reclassification of identity/relationships rows from the
    flat 'default' store into the dedicated 'profile' store (the
    memory-layers phase — see memory_manager.py's _store_for_category and
    the schema comment on memory_entries below). A plain UPDATE, not a
    delete+reinsert — no data loss, no row ever leaves memory_entries.

    Must run after _migrate_legacy_memory/_migrate_legacy_contacts above:
    a fresh install importing memory/long_term.json lands everything in
    'default' first (that migration predates the store split and knows
    nothing about it), so this pass is what actually reclassifies those
    rows afterward.

    Idempotent: skips once any 'profile'-store row exists — covers both
    "already migrated" and "nothing to migrate" (a fresh DB with no memory
    data at all, or one that already has profile rows because
    memory_manager.py's own writes have gone straight to the right store
    since this phase landed)."""
    row_count = conn.execute(
        "SELECT COUNT(*) FROM memory_entries WHERE store = 'profile'"
    ).fetchone()[0]
    if row_count > 0:
        return

    # Imported lazily — same reason _migrate_legacy_memory imports from
    # memory_manager lazily: memory_manager imports get_conn from this
    # module, so a module-level import here would be circular.
    from memory.memory_manager import _PROFILE_CATEGORIES

    placeholders = ", ".join("?" * len(_PROFILE_CATEGORIES))
    with conn:
        cur = conn.execute(
            f"UPDATE memory_entries SET store = 'profile' "
            f"WHERE store = 'default' AND category IN ({placeholders})",
            _PROFILE_CATEGORIES,
        )
        n = cur.rowcount
    if n:
        _log.info("Migrated %d memory_entries row(s) from 'default' to 'profile' store.", n)


# ---------------------------------------------------------------------------
# Task event trace (task_events — durable, step-level, task-scoped only)
# ---------------------------------------------------------------------------

def log_task_event(
    task_id:     str | None,
    step_num,
    tool:        str | None,
    description: str,
    status:      str,
    detail:      str = "",
    duration_ms: int | None = None,
) -> None:
    """Insert one row into task_events. Callers run on a thread that
    already tolerates blocking (AgentExecutor's own per-task worker
    thread, or callers that borrow its task_id/step_num context), so a
    synchronous DB write here is safe — this is NOT the general logging
    path and must never be called from the audio/transcript threads.
    No-op if task_id is None. A DB hiccup here must never break the
    actual agent task, so failures are logged, not raised.

    Status vocabulary: 'started', 'done', 'skipped', 'attempt_failed'
    (a single attempt errored but a retry/replan/skip may still follow —
    NOT terminal), 'retried', 'replanned', 'failed' (terminal — the step
    will not succeed). Keeping 'attempt_failed' distinct from 'failed'
    is what makes `GROUP BY tool, status` give a real failure rate
    instead of counting every transient retry as a failure.
    """
    if not task_id:
        return
    try:
        conn = get_conn()
        with conn:
            conn.execute(
                "INSERT INTO task_events "
                "(task_id, step_num, tool, description, status, detail, duration_ms, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (task_id, step_num, tool, description, status, detail[:500], duration_ms, time.time()),
            )
    except Exception:
        _log.warning(
            "Could not log task event (task_id=%s, step=%s, status=%s)",
            task_id, step_num, status, exc_info=True,
        )


def get_task_trace(task_id: str) -> list[sqlite3.Row]:
    """All task_events + log_events rows for one task, chronological.
    task_events carries the step ledger (tool/status/duration_ms);
    log_events carries everything else that happened to be tagged with
    this task_id (LLM calls, warnings, errors) but isn't a step itself."""
    conn = get_conn()
    return conn.execute(
        "SELECT created_at, 'step' AS kind, step_num, tool AS source, "
        "       status AS level, description AS message, detail, duration_ms "
        "FROM task_events WHERE task_id = ? "
        "UNION ALL "
        "SELECT created_at, 'log' AS kind, NULL AS step_num, source, "
        "       level, message, detail, duration_ms "
        "FROM log_events WHERE task_id = ? "
        "ORDER BY created_at",
        (task_id, task_id),
    ).fetchall()
