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
import sqlite3
import threading

from config import BASE_DIR

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
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_task_events_task_id ON task_events(task_id);

-- `store` defaults to 'default' and every row uses that value in this
-- phase — column exists so a later split into multiple typed memory
-- stores has somewhere to land without a schema migration.
CREATE TABLE IF NOT EXISTS memory_entries (
    store       TEXT NOT NULL DEFAULT 'default',
    category    TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (store, category, key)
);
CREATE INDEX IF NOT EXISTS idx_memory_updated ON memory_entries(store, updated_at);

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
    conn.commit()
    _local.conn = conn

    _migrate_legacy_memory(conn)
    _migrate_legacy_contacts(conn)

    return conn


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
        print(f"[DB] Memory migration: could not read {MEMORY_PATH}: {e}")
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
        print(f"[DB] Migrated {len(entries)} memory entries from {MEMORY_PATH.name} -> {migrated_path.name}")
    except Exception as e:
        print(f"[DB] Migrated {len(entries)} memory entries, but could not rename source file: {e}")


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
        print(f"[DB] Contacts migration: could not read {contacts_path}: {e}")
        return

    if not isinstance(raw, dict) or not raw:
        return

    with conn:
        for alias, exact_name in raw.items():
            conn.execute(
                "INSERT OR IGNORE INTO contacts (alias, exact_name) VALUES (?, ?)",
                (alias, exact_name),
            )
    print(f"[DB] Migrated {len(raw)} contact(s) from {contacts_path.name}.")
