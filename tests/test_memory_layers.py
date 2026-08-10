"""
tests/test_memory_layers.py
=============================
Covers the memory-layers phase: the profile/default store split, per-store
eviction, the 'default'->'profile' reclassification migration, and that
all 3 real consumers of load_memory() (main.py's format_memory_for_prompt,
computer_control.py's _user_profile, daily_briefing.py's _get_user_city)
still work unmodified against the merged two-store shape.

Redirects core.db at a fresh temp sqlite file, same pattern as the other
test modules in this package, so nothing here touches data/jarvis.db.
"""
from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.db as db


def _use_temp_db() -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="jarvis_test_db_"))
    db.DB_DIR  = tmp_dir
    db.DB_PATH = tmp_dir / "test.db"
    db._local  = threading.local()
    return db.DB_PATH


_use_temp_db()

import memory.memory_manager as mm  # noqa: E402


class StoreForCategoryTest(unittest.TestCase):
    def test_identity_and_relationships_map_to_profile(self):
        self.assertEqual(mm._store_for_category("identity"), "profile")
        self.assertEqual(mm._store_for_category("relationships"), "profile")

    def test_the_rest_map_to_default(self):
        for cat in ("preferences", "projects", "wishes", "notes"):
            self.assertEqual(mm._store_for_category(cat), "default")

    def test_unknown_category_falls_back_to_default(self):
        self.assertEqual(mm._store_for_category("some_future_category"), "default")


class FreshInstallMigrationTest(unittest.TestCase):
    """Migration behavior on a fresh DB with no prior data — must be a
    no-op, not an error."""

    def test_fresh_db_get_conn_does_not_error(self):
        _use_temp_db()
        conn = db.get_conn()  # runs all three migrations, including _migrate_profile_categories
        count = conn.execute("SELECT COUNT(*) FROM memory_entries").fetchone()[0]
        self.assertEqual(count, 0)

    def test_fresh_db_load_memory_returns_empty_shape(self):
        _use_temp_db()
        memory = mm.load_memory()
        self.assertEqual(memory, mm._empty_memory())

    def test_calling_migration_directly_twice_on_empty_db_is_a_noop(self):
        _use_temp_db()
        conn = db.get_conn()
        db._migrate_profile_categories(conn)
        db._migrate_profile_categories(conn)
        count = conn.execute("SELECT COUNT(*) FROM memory_entries").fetchone()[0]
        self.assertEqual(count, 0)


class ProfileReclassificationMigrationTest(unittest.TestCase):
    """Simulates pre-phase data: identity/relationships rows already sitting
    in 'default' (as every row was before this phase), then confirms the
    migration reclassifies them without data loss."""

    def setUp(self):
        _use_temp_db()

    def _insert_legacy_default_row(self, category, key, value, updated_at="2020-01-01"):
        conn = db.get_conn()
        with conn:
            conn.execute(
                "INSERT INTO memory_entries (store, category, key, value, updated_at) "
                "VALUES ('default', ?, ?, ?, ?)",
                (category, key, value, updated_at),
            )

    def test_identity_and_relationships_rows_move_to_profile(self):
        self._insert_legacy_default_row("identity", "name", "Alice")
        self._insert_legacy_default_row("relationships", "sister", "Beth")
        self._insert_legacy_default_row("notes", "reminder", "buy milk")

        conn = db.get_conn()
        db._migrate_profile_categories(conn)

        rows = {
            (r["store"], r["category"], r["key"]): r["value"]
            for r in conn.execute("SELECT store, category, key, value FROM memory_entries").fetchall()
        }
        self.assertEqual(rows[("profile", "identity", "name")], "Alice")
        self.assertEqual(rows[("profile", "relationships", "sister")], "Beth")
        # notes must NOT move — it stays in 'default'
        self.assertEqual(rows[("default", "notes", "reminder")], "buy milk")
        self.assertEqual(len(rows), 3)  # no row duplicated, none lost

    def test_migration_is_idempotent(self):
        self._insert_legacy_default_row("identity", "name", "Alice")
        conn = db.get_conn()
        db._migrate_profile_categories(conn)
        db._migrate_profile_categories(conn)  # second call must be a no-op, not a duplicate/error
        count = conn.execute(
            "SELECT COUNT(*) FROM memory_entries WHERE category = 'identity' AND key = 'name'"
        ).fetchone()[0]
        self.assertEqual(count, 1)

    def test_no_data_lost_value_and_updated_at_preserved(self):
        self._insert_legacy_default_row("identity", "city", "Portland", updated_at="2019-06-01")
        conn = db.get_conn()
        db._migrate_profile_categories(conn)
        row = conn.execute(
            "SELECT store, value, updated_at FROM memory_entries WHERE category='identity' AND key='city'"
        ).fetchone()
        self.assertEqual(row["store"], "profile")
        self.assertEqual(row["value"], "Portland")
        self.assertEqual(row["updated_at"], "2019-06-01")


class PerStoreEvictionTest(unittest.TestCase):
    def setUp(self):
        _use_temp_db()

    def test_default_store_eviction_never_touches_profile(self):
        """The exact scenario flagged in the investigation, forced
        directly: an old, rarely-touched identity/name entry, then enough
        eviction-pressure on 'default' to guarantee at least one eviction
        there — confirm identity/name survives untouched, not just
        'probably survives'."""
        mm.update_memory({"identity": {"name": {"value": "Alice"}}})

        # Backdate it to look like it hasn't been touched in years — under
        # the old flat-table/global-budget scheme this would make it the
        # *first* candidate for eviction (oldest updated_at, no category
        # awareness).
        conn = db.get_conn()
        with conn:
            conn.execute(
                "UPDATE memory_entries SET updated_at = '2000-01-01' "
                "WHERE store='profile' AND category='identity' AND key='name'"
            )

        # Flood 'default' well past MEMORY_MAX_CHARS (8000) with notes.
        for i in range(30):
            mm.update_memory({"notes": {f"note_{i}": {"value": "x" * 380}}})

        default_total = conn.execute(
            "SELECT COALESCE(SUM(LENGTH(value)),0) FROM memory_entries WHERE store='default'"
        ).fetchone()[0]
        self.assertLessEqual(default_total, mm.MEMORY_MAX_CHARS, "eviction should have kept 'default' under budget")

        default_count = conn.execute(
            "SELECT COUNT(*) FROM memory_entries WHERE store='default'"
        ).fetchone()[0]
        self.assertLess(default_count, 30, "some notes entries must have actually been evicted")

        # The whole point: identity/name is untouched, despite being the
        # oldest-updated_at row in the entire table by a wide margin.
        identity_row = conn.execute(
            "SELECT value FROM memory_entries WHERE store='profile' AND category='identity' AND key='name'"
        ).fetchone()
        self.assertIsNotNone(identity_row, "profile-store identity/name must never be evicted by default-store pressure")
        self.assertEqual(identity_row["value"], "Alice")

    def test_evict_over_budget_only_touches_the_given_store(self):
        conn = db.get_conn()
        with conn:
            conn.execute(
                "INSERT INTO memory_entries (store, category, key, value, updated_at) "
                "VALUES ('profile', 'identity', 'name', ?, '2000-01-01')",
                ("x" * 100,),
            )
            conn.execute(
                "INSERT INTO memory_entries (store, category, key, value, updated_at) "
                "VALUES ('default', 'notes', 'n1', ?, '2000-01-01')",
                ("x" * 100,),
            )
        # budget=0 forces eviction of everything in the targeted store only
        with conn:
            mm._evict_over_budget(conn, "default", 0)
        remaining = {r["store"] for r in conn.execute("SELECT store FROM memory_entries").fetchall()}
        self.assertEqual(remaining, {"profile"})

    def test_evict_all_stores_applies_each_budget_independently(self):
        conn = db.get_conn()
        with conn:
            conn.execute(
                "INSERT INTO memory_entries (store, category, key, value, updated_at) "
                "VALUES ('profile', 'identity', 'old', ?, '2000-01-01')",
                ("x" * (mm.PROFILE_MAX_CHARS + 100),),
            )
        with conn:
            evicted = mm._evict_all_stores(conn)
        self.assertIn(("identity", "old"), evicted)


class ConsumersUnmodifiedTest(unittest.TestCase):
    """The 3 real consumers found in the investigation — verified against
    live data split across both stores, through the actual public
    functions each module calls (not memory_manager internals)."""

    def setUp(self):
        _use_temp_db()
        mm.update_memory({
            "identity": {
                "name": {"value": "Alice"},
                "city": {"value": "Seattle"},
            },
            "notes": {"todo": {"value": "buy milk"}},
        })

    def test_format_memory_for_prompt_renders_both_stores_merged(self):
        memory   = mm.load_memory()
        rendered = mm.format_memory_for_prompt(memory)
        self.assertIn("Name: Alice", rendered)
        self.assertIn("City: Seattle", rendered)
        self.assertIn("buy milk", rendered)

    def test_computer_control_user_profile_reads_identity_via_merged_load(self):
        import actions.computer_control as cc
        profile = cc._user_profile()
        self.assertEqual(profile.get("name"), "Alice")
        self.assertEqual(profile.get("city"), "Seattle")

    def test_daily_briefing_get_user_city_reads_identity_via_merged_load(self):
        import actions.daily_briefing as db_briefing
        city = db_briefing._get_user_city()
        self.assertEqual(city, "Seattle")


if __name__ == "__main__":
    unittest.main(verbosity=2)
