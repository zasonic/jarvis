"""Tests for the cloud-backup sync tracking on conversation_summaries.

Covers the new ``synced_at`` column: present on fresh DBs, added by the
idempotent migration to pre-existing DBs, and set by ``mark_summary_synced``.
"""
import sqlite3

from jarvis.memory.db import Database


def _columns(db: Database, table: str) -> set:
    cur = db.conn.cursor()
    return {r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()}


def test_fresh_db_has_synced_at_column():
    db = Database(":memory:", None)
    assert "synced_at" in _columns(db, "conversation_summaries")


def test_migration_adds_synced_at_to_existing_db(tmp_path):
    # Simulate a database created before the column existed.
    path = tmp_path / "old.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE conversation_summaries (
          id INTEGER PRIMARY KEY,
          date_utc TEXT NOT NULL,
          ts_utc TEXT NOT NULL,
          summary TEXT NOT NULL,
          topics TEXT,
          source_app TEXT NOT NULL,
          UNIQUE(date_utc, source_app)
        );
        """
    )
    conn.commit()
    conn.close()

    db = Database(str(path), None)  # opening runs the idempotent migration
    assert "synced_at" in _columns(db, "conversation_summaries")


def test_mark_summary_synced_sets_timestamp():
    db = Database(":memory:", None)
    sid = db.upsert_conversation_summary(
        date_utc="2025-05-20", summary="went hiking", topics="hobbies"
    )
    # Not backed up yet.
    assert db.get_conversation_summary("2025-05-20")["synced_at"] is None

    db.mark_summary_synced(sid)
    assert db.get_conversation_summary("2025-05-20")["synced_at"] is not None


def test_mark_summary_synced_missing_row_is_noop():
    db = Database(":memory:", None)
    db.mark_summary_synced(99999)  # must not raise
