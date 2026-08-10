import logging
from datetime import datetime

from config import BASE_DIR
from core.db import get_conn

_log = logging.getLogger("jarvis.memory")

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

# Backstop only, not a realistic budget — profile entries are a handful of
# near-permanent identity/relationship facts (name, city, a few family
# members), nowhere near this in normal use. It exists so a pathological
# case still has *some* bound rather than truly unbounded growth, not
# because we expect it to ever fire.
PROFILE_MAX_CHARS = 4000

_CATEGORIES = ("identity", "preferences", "projects", "relationships", "wishes", "notes")

# Two stores, not one, as of the memory-layers phase — see
# core/db.py's memory_entries schema comment and
# core/db.py's _migrate_profile_categories for the one-time migration.
#
# 'profile' (identity, relationships): low-write-frequency, high-value,
# near-permanent facts — exempt from the normal shared eviction budget.
# Investigation finding this fixes: eviction is oldest-updated_at-first
# with no category awareness, so a user's name (saved once, rarely
# re-touched) could have an older updated_at than a throwaway note added
# last week, and would legitimately be the first thing eviction picks —
# a real, structurally-possible failure mode under the old flat table,
# not a hypothetical one.
#
# 'default' (preferences, projects, wishes, notes): unchanged behavior
# from before this phase — same MEMORY_MAX_CHARS budget, same
# oldest-updated_at-first eviction. These are the genuinely more
# disposable/time-bound categories; none of the four has a stronger
# retention claim than the others, so they continue sharing one budget.
_PROFILE_STORE      = "profile"
_DEFAULT_STORE      = "default"
_PROFILE_CATEGORIES = ("identity", "relationships")


def _store_for_category(category: str) -> str:
    """Categories outside the current 6 (see _rows_to_memory's own
    tolerance for this) default to 'default', not 'profile' — an unknown
    category has no established claim to protected retention."""
    return _PROFILE_STORE if category in _PROFILE_CATEGORIES else _DEFAULT_STORE


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
    """Merges both stores into the one nested dict shape every existing
    caller already expects — the store split is entirely internal to this
    module; load_memory()'s return shape and every consumer of it
    (format_memory_for_prompt, computer_control.py's _user_profile,
    daily_briefing.py's _get_user_city) are unchanged by this phase."""
    rows = conn.execute(
        "SELECT category, key, value, updated_at FROM memory_entries WHERE store IN (?, ?)",
        (_PROFILE_STORE, _DEFAULT_STORE),
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
        store   = _store_for_category(cat)
        conn.execute(
            "INSERT INTO memory_entries (store, category, key, value, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(store, category, key) DO UPDATE SET "
            "value = excluded.value, updated_at = excluded.updated_at",
            (store, cat, key, value, updated),
        )


def _evict_over_budget(conn, store: str, budget: int) -> list[tuple[str, str]]:
    """Delete oldest-updated_at rows *within `store` only* until
    SUM(LENGTH(value)) for that store is back under `budget`. Must be
    called from within the caller's own transaction (it does not commit).

    Per-store, not global, as of the memory-layers phase: a write that
    pushes 'default' over budget can never delete a 'profile' row, and
    vice versa — see this module's _PROFILE_CATEGORIES comment for why
    that separation exists (a shared budget let a low-value 'notes' entry
    evict a high-value 'identity' entry purely because eviction is
    oldest-updated_at-first with no category awareness)."""
    total = conn.execute(
        "SELECT COALESCE(SUM(LENGTH(value)), 0) FROM memory_entries WHERE store = ?",
        (store,),
    ).fetchone()[0]
    if total <= budget:
        return []

    rows = conn.execute(
        "SELECT category, key, LENGTH(value) AS vlen FROM memory_entries "
        "WHERE store = ? ORDER BY updated_at ASC",
        (store,),
    ).fetchall()

    evicted = []
    for row in rows:
        if total <= budget:
            break
        conn.execute(
            "DELETE FROM memory_entries WHERE store = ? AND category = ? AND key = ?",
            (store, row["category"], row["key"]),
        )
        total -= row["vlen"]
        evicted.append((row["category"], row["key"]))
        _log.info("Trimmed %s/%s (store=%s)", row['category'], row['key'], store)
    return evicted


def _evict_all_stores(conn) -> list[tuple[str, str]]:
    """Runs eviction independently per store, both with their own budget,
    after every write — regardless of which store the write actually
    touched. Simpler and cheap (row counts are small) compared to tracking
    exactly which store(s) a given update_memory() call could have
    affected."""
    evicted = _evict_over_budget(conn, _DEFAULT_STORE, MEMORY_MAX_CHARS)
    evicted += _evict_over_budget(conn, _PROFILE_STORE, PROFILE_MAX_CHARS)
    return evicted


def save_memory(memory: dict) -> list[tuple[str, str]]:
    if not isinstance(memory, dict):
        return []
    conn = get_conn()
    conn.execute("BEGIN IMMEDIATE")
    try:
        _upsert_entries(conn, memory)
        evicted = _evict_all_stores(conn)
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
            evicted = _evict_all_stores(conn)
            msg = f"Saved: {list(memory_update.keys())}"
            if evicted:
                forgotten = ", ".join(f"{cat}/{key}" for cat, key in evicted)
                msg += f" | Memory limit reached — forgot: {forgotten}"
            _log.info(msg)
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
    conn  = get_conn()
    store = _store_for_category(category)
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT 1 FROM memory_entries WHERE store = ? AND category = ? AND key = ?",
            (store, category, key),
        ).fetchone()
        if not row:
            conn.rollback()
            return f"Not found: {category}/{key}"

        conn.execute(
            "DELETE FROM memory_entries WHERE store = ? AND category = ? AND key = ?",
            (store, category, key),
        )
        evicted = _evict_all_stores(conn)   # deletion alone can't overflow; kept for parity with the original save_memory-after-forget behavior
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    if evicted:
        forgotten = ", ".join(f"{c}/{k}" for c, k in evicted)
        _log.info("Saved after forget | Memory limit reached — forgot: %s", forgotten)
    return f"Forgotten: {category}/{key}"


forget_memory = forget
