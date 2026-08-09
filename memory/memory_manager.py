from datetime import datetime

from config import BASE_DIR
from core.db import get_conn

# Legacy JSON location — no longer read/written directly (see
# core.db._migrate_legacy_memory), but memory_manager.py still owns this
# path constant since the migration script imports it from here.
MEMORY_PATH = BASE_DIR / "memory" / "long_term.json"

# Entries are capped at MAX_VALUE_LENGTH (380) chars each, so 2200 total
# left room for only ~6 entries before silently evicting the oldest ones.
# 8000 gives realistic day-to-day headroom (~20+ entries) while still
# keeping the blob small enough to stay cheap in the prompt context.
MAX_VALUE_LENGTH = 380
MEMORY_MAX_CHARS = 8000

# Every row in this phase uses the 'default' store — see memory_entries'
# schema comment in core/db.py for why the column exists anyway.
_STORE = "default"

_CATEGORIES = ("identity", "preferences", "projects", "relationships", "wishes", "notes")


def _empty_memory() -> dict:
    return {cat: {} for cat in _CATEGORIES}


def _rows_to_memory(rows) -> dict:
    """Reshape memory_entries rows into the exact nested dict shape every
    existing caller already expects: {category: {key: {"value":.., "updated":..}}}."""
    memory = _empty_memory()
    for row in rows:
        cat = row["category"]
        if cat not in memory:
            memory[cat] = {}   # tolerate a category outside the current 6 rather than drop data
        memory[cat][row["key"]] = {"value": row["value"], "updated": row["updated_at"]}
    return memory


def _load_memory_on(conn) -> dict:
    rows = conn.execute(
        "SELECT category, key, value, updated_at FROM memory_entries WHERE store = ?",
        (_STORE,),
    ).fetchall()
    return _rows_to_memory(rows)


def load_memory() -> dict:
    return _load_memory_on(get_conn())


def _all_entries(memory: dict) -> list[tuple]:
    entries = []
    for cat, items in memory.items():
        if not isinstance(items, dict):
            continue
        for key, entry in items.items():
            if isinstance(entry, dict) and "value" in entry:
                entries.append((cat, key, entry))
    return entries


def _upsert_entries(conn, memory: dict) -> None:
    for cat, key, entry in _all_entries(memory):
        value   = entry.get("value", "")
        updated = entry.get("updated") or datetime.now().strftime("%Y-%m-%d")
        conn.execute(
            "INSERT INTO memory_entries (store, category, key, value, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(store, category, key) DO UPDATE SET "
            "value = excluded.value, updated_at = excluded.updated_at",
            (_STORE, cat, key, value, updated),
        )


def _evict_over_budget(conn) -> list[tuple[str, str]]:
    """Delete oldest-updated_at rows until SUM(LENGTH(value)) across all
    'default'-store rows is back under MEMORY_MAX_CHARS. Must be called
    from within the caller's own transaction (it does not commit)."""
    total = conn.execute(
        "SELECT COALESCE(SUM(LENGTH(value)), 0) FROM memory_entries WHERE store = ?",
        (_STORE,),
    ).fetchone()[0]
    if total <= MEMORY_MAX_CHARS:
        return []

    rows = conn.execute(
        "SELECT category, key, LENGTH(value) AS vlen FROM memory_entries "
        "WHERE store = ? ORDER BY updated_at ASC",
        (_STORE,),
    ).fetchall()

    evicted = []
    for row in rows:
        if total <= MEMORY_MAX_CHARS:
            break
        conn.execute(
            "DELETE FROM memory_entries WHERE store = ? AND category = ? AND key = ?",
            (_STORE, row["category"], row["key"]),
        )
        total -= row["vlen"]
        evicted.append((row["category"], row["key"]))
        print(f"[Memory] 🗑️  Trimmed {row['category']}/{row['key']}")
    return evicted


def save_memory(memory: dict) -> list[tuple[str, str]]:
    if not isinstance(memory, dict):
        return []
    conn = get_conn()
    conn.execute("BEGIN IMMEDIATE")
    try:
        _upsert_entries(conn, memory)
        evicted = _evict_over_budget(conn)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return evicted


def _truncate_value(val: str) -> str:
    if isinstance(val, str) and len(val) > MAX_VALUE_LENGTH:
        return val[:MAX_VALUE_LENGTH].rstrip() + "…"
    return val


def _recursive_update(target: dict, updates: dict) -> bool:
    changed = False
    for key, value in updates.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, dict) and "value" not in value:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
                changed = True
            if _recursive_update(target[key], value):
                changed = True
        else:
            new_val  = _truncate_value(str(value["value"] if isinstance(value, dict) else value))
            entry    = {"value": new_val, "updated": datetime.now().strftime("%Y-%m-%d")}
            existing = target.get(key, {})
            if not isinstance(existing, dict) or existing.get("value") != new_val:
                target[key] = entry
                changed = True
    return changed


def update_memory(memory_update: dict) -> dict:
    """Single-transaction read-modify-write: BEGIN IMMEDIATE takes the
    write lock before the read happens, so a second concurrent
    update_memory()/forget() call blocks (up to busy_timeout) until this
    one commits, instead of both loading the same stale base and one
    silently clobbering the other's write — the race the old file-based
    load-then-separately-save flow had."""
    if not isinstance(memory_update, dict) or not memory_update:
        return load_memory()

    conn = get_conn()
    conn.execute("BEGIN IMMEDIATE")
    try:
        memory = _load_memory_on(conn)
        if _recursive_update(memory, memory_update):
            _upsert_entries(conn, memory)
            evicted = _evict_over_budget(conn)
            msg = f"[Memory] 💾 Saved: {list(memory_update.keys())}"
            if evicted:
                forgotten = ", ".join(f"{cat}/{key}" for cat, key in evicted)
                msg += f" | Memory limit reached — forgot: {forgotten}"
            print(msg)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return memory


def format_memory_for_prompt(memory: dict | None) -> str:
    if not memory:
        return ""

    lines = []

    identity  = memory.get("identity", {})
    id_fields = ["name", "age", "birthday", "city", "job", "language", "school", "nationality"]
    for field in id_fields:
        entry = identity.get(field)
        if entry:
            val = entry.get("value") if isinstance(entry, dict) else entry
            if val:
                lines.append(f"{field.title()}: {val}")
    for key, entry in identity.items():
        if key in id_fields:
            continue
        val = entry.get("value") if isinstance(entry, dict) else entry
        if val:
            lines.append(f"{key.replace('_', ' ').title()}: {val}")

    prefs = memory.get("preferences", {})
    if prefs:
        lines.append("")
        lines.append("Preferences:")
        for key, entry in list(prefs.items())[:15]:
            val = entry.get("value") if isinstance(entry, dict) else entry
            if val:
                lines.append(f"  - {key.replace('_', ' ').title()}: {val}")

    projects = memory.get("projects", {})
    if projects:
        lines.append("")
        lines.append("Active Projects / Goals:")
        for key, entry in list(projects.items())[:8]:
            val = entry.get("value") if isinstance(entry, dict) else entry
            if val:
                lines.append(f"  - {key.replace('_', ' ').title()}: {val}")

    rels = memory.get("relationships", {})
    if rels:
        lines.append("")
        lines.append("People in their life:")
        for key, entry in list(rels.items())[:10]:
            val = entry.get("value") if isinstance(entry, dict) else entry
            if val:
                lines.append(f"  - {key.replace('_', ' ').title()}: {val}")

    wishes = memory.get("wishes", {})
    if wishes:
        lines.append("")
        lines.append("Wishes / Plans / Wants:")
        for key, entry in list(wishes.items())[:8]:
            val = entry.get("value") if isinstance(entry, dict) else entry
            if val:
                lines.append(f"  - {key.replace('_', ' ').title()}: {val}")

    notes = memory.get("notes", {})
    if notes:
        lines.append("")
        lines.append("Other notes:")
        for key, entry in list(notes.items())[:8]:
            val = entry.get("value") if isinstance(entry, dict) else entry
            if val:
                lines.append(f"  - {key}: {val}")

    if not lines:
        return ""

    header = "[WHAT YOU KNOW ABOUT THIS PERSON — use naturally, never recite like a list]\n"
    result = header + "\n".join(lines)
    if len(result) > 2000:
        result = result[:1997] + "…"

    return result + "\n"


def remember(key: str, value: str, category: str = "notes") -> str:
    valid = {"identity", "preferences", "projects", "relationships", "wishes", "notes"}
    if category not in valid:
        category = "notes"
    update_memory({category: {key: {"value": value}}})
    return f"Remembered: {category}/{key} = {value}"


def forget(key: str, category: str = "notes") -> str:
    conn = get_conn()
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT 1 FROM memory_entries WHERE store = ? AND category = ? AND key = ?",
            (_STORE, category, key),
        ).fetchone()
        if not row:
            conn.rollback()
            return f"Not found: {category}/{key}"

        conn.execute(
            "DELETE FROM memory_entries WHERE store = ? AND category = ? AND key = ?",
            (_STORE, category, key),
        )
        evicted = _evict_over_budget(conn)   # deletion alone can't overflow; kept for parity with the original save_memory-after-forget behavior
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    if evicted:
        forgotten = ", ".join(f"{c}/{k}" for c, k in evicted)
        print(f"[Memory] 💾 Saved after forget | Memory limit reached — forgot: {forgotten}")
    return f"Forgotten: {category}/{key}"


forget_memory = forget
