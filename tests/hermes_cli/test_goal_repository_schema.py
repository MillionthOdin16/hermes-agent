"""
Phase 1 schema tests for GoalRepository.

Verifies that:
- v14 schema tables and indexes initialize correctly
- Transaction helper supports nested savepoints and rollback
- Schema migration is idempotent (re-init doesn't fail)
- Foreign keys are enforced
- Goals.v2_repository=false = no runtime behavior change (verified via
  integration tests with the full goals module loaded)
"""
import sqlite3
import tempfile
import os
from pathlib import Path

import pytest

from hermes_state import (
    SessionDB,
    Transaction,
    SCHEMA_VERSION,
    MAX_GOAL_SESSION_REDIRECTS,
    _create_goal_indexes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mkdb() -> SessionDB:
    """Make a fresh temporary SessionDB."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    return SessionDB(Path(path)), path


def close(db: SessionDB, path: str) -> None:
    db.close()
    if os.path.exists(path):
        os.unlink(path)


# ---------------------------------------------------------------------------
# Schema initialization
# ---------------------------------------------------------------------------

def test_v14_schema_tables_exist():
    """All v14 goal tables are created on init."""
    db, path = mkdb()
    try:
        cur = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {r[0] for r in cur.fetchall()}
        expected = {"goals", "goal_sessions", "goal_events", "goal_legacy_migrations"}
        assert expected <= tables, f"missing tables: {expected - tables}"
    finally:
        close(db, path)


def test_v14_unique_indexes_created():
    """Alias-safe uniqueness indexes exist on goal_sessions."""
    db, path = mkdb()
    try:
        cur = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_one%'"
        )
        indexes = {r[0] for r in cur.fetchall()}
        assert "idx_one_active_goal_per_session" in indexes
        assert "idx_one_active_binding_per_goal" in indexes
    finally:
        close(db, path)


def test_v14_indexes_include_regular_indexes():
    """Regular (non-unique) indexes are created on all key columns."""
    db, path = mkdb()
    try:
        cur = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_goals%' OR name LIKE 'idx_goal_sessions%' OR name LIKE 'idx_goal_events%'"
        )
        indexes = {r[0] for r in cur.fetchall()}
        required = {
            "idx_goals_status",
            "idx_goals_session_key",
            "idx_goals_updated_at",
            "idx_goal_sessions_session",
            "idx_goal_sessions_goal",
            "idx_goal_events_goal",
        }
        assert required <= indexes, f"missing regular indexes: {required - indexes}"
    finally:
        close(db, path)


def test_foreign_keys_pragma_enabled():
    """PRAGMA foreign_keys=ON is set on the connection."""
    db, path = mkdb()
    try:
        cur = db._conn.execute("PRAGMA foreign_keys")
        fk = cur.fetchone()[0]
        assert fk == 1, "foreign_keys should be ON"
    finally:
        close(db, path)


def test_schema_version_bumped_to_14():
    """Schema version is set to 14 after migration."""
    db, path = mkdb()
    try:
        cur = db._conn.execute("SELECT version FROM schema_version LIMIT 1")
        row = cur.fetchone()
        assert row is not None
        assert row[0] == 14
    finally:
        close(db, path)


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def test_re_init_does_not_fail():
    """Calling _init_schema twice is safe (idempotent table creation)."""
    db, path = mkdb()
    try:
        # Run init again — should not raise
        db._init_schema()
        # Tables still present
        cur = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('goals','goal_sessions','goal_events','goal_legacy_migrations')"
        )
        assert len(cur.fetchall()) == 4
    finally:
        close(db, path)


def test_indexes_if_not_exists_idempotent():
    """Creating indexes via _create_goal_indexes twice does not fail."""
    db, path = mkdb()
    try:
        cur = db._conn.cursor()
        _create_goal_indexes(cur)  # first time
        _create_goal_indexes(cur)  # second time — should not raise
    finally:
        close(db, path)


# ---------------------------------------------------------------------------
# Transaction helper
# ---------------------------------------------------------------------------

def test_transaction_basic_insert_and_commit():
    """Transaction commits on exit when no exception raised."""
    db, path = mkdb()
    try:
        with db.transaction() as tx:
            tx.execute(
                "INSERT INTO goals "
                "(goal_id, schema_version, status, state_json, created_at, updated_at) "
                "VALUES ('tx_test1', 1, 'active', '{}', 0.0, 0.0)"
            )
        # committed — verify
        cur = db._conn.execute("SELECT goal_id FROM goals WHERE goal_id='tx_test1'")
        assert cur.fetchone() is not None
    finally:
        close(db, path)


def test_transaction_rollback_on_exception():
    """Transaction rolls back when an exception propagates."""
    db, path = mkdb()
    try:
        with pytest.raises(Exception):
            with db.transaction() as tx:
                tx.execute(
                    "INSERT INTO goals "
                    "(goal_id, schema_version, status, state_json, created_at, updated_at) "
                    "VALUES ('tx_test_exc', 1, 'active', '{}', 0.0, 0.0)"
                )
                raise Exception("simulated error")
        # rolled back — goal should not exist
        cur = db._conn.execute("SELECT goal_id FROM goals WHERE goal_id='tx_test_exc'")
        assert cur.fetchone() is None
    finally:
        close(db, path)


def test_transaction_nested_commit():
    """Nested transaction commits inner savepoint but not outer."""
    db, path = mkdb()
    try:
        with db.transaction() as outer:
            outer.execute(
                "INSERT INTO goals "
                "(goal_id, schema_version, status, state_json, created_at, updated_at) "
                "VALUES ('nested_outer', 1, 'active', '{}', 0.0, 0.0)"
            )
            with db.transaction() as inner:
                inner.execute(
                    "INSERT INTO goal_events "
                    "(goal_id, event_type, created_at) "
                    "VALUES ('nested_outer', 'inner_event', 0.0)"
                )
                # inner releases savepoint here on exit
        # outer still active — committed on outer exit
        cur = db._conn.execute("SELECT goal_id FROM goals WHERE goal_id='nested_outer'")
        assert cur.fetchone() is not None
        cur = db._conn.execute(
            "SELECT event_type FROM goal_events WHERE goal_id='nested_outer'"
        )
        assert cur.fetchone() is not None
    finally:
        close(db, path)


def test_transaction_nested_rollback_inner_only():
    """Rolling back inner transaction does not roll back outer."""
    db, path = mkdb()
    try:
        with db.transaction() as outer:
            outer.execute(
                "INSERT INTO goals "
                "(goal_id, schema_version, status, state_json, created_at, updated_at) "
                "VALUES ('nested_rbk', 1, 'active', '{}', 0.0, 0.0)"
            )
            with db.transaction() as inner:
                inner.execute(
                    "INSERT INTO goal_events "
                    "(goal_id, event_type, created_at) "
                    "VALUES ('nested_rbk', 'should_be_gone', 0.0)"
                )
                inner.rollback()
            # inner rolled back; outer still alive
        # outer committed
        cur = db._conn.execute("SELECT goal_id FROM goals WHERE goal_id='nested_rbk'")
        assert cur.fetchone() is not None
        cur = db._conn.execute(
            "SELECT event_type FROM goal_events WHERE goal_id='nested_rbk'"
        )
        assert cur.fetchone() is None, "inner rollback should have removed the event"
    finally:
        close(db, path)


def test_transaction_re_entrant_from_write_callback():
    """Transaction can be called inside _execute_write callback without deadlock."""
    db, path = mkdb()
    try:
        outer_started = False

        def callback():
            nonlocal outer_started
            outer_started = True
            # This starts a new transaction while outer is still "active"
            with db.transaction() as inner_tx:
                inner_tx.execute(
                    "INSERT INTO goal_events "
                    "(goal_id, event_type, created_at) "
                    "VALUES ('reentrant_cb', 'cb_event', 0.0)"
                )

        with db.transaction() as outer:
            outer.execute(
                "INSERT INTO goals "
                "(goal_id, schema_version, status, state_json, created_at, updated_at) "
                "VALUES ('reentrant_cb', 1, 'active', '{}', 0.0, 0.0)"
            )
            callback()

        assert outer_started
        cur = db._conn.execute(
            "SELECT event_type FROM goal_events WHERE goal_id='reentrant_cb'"
        )
        assert cur.fetchone() is not None
    finally:
        close(db, path)


def test_transaction_manual_rollback():
    """Explicit rollback() method works without exception context."""
    db, path = mkdb()
    try:
        tx = db.transaction()
        tx.__enter__()
        tx.execute(
            "INSERT INTO goals "
            "(goal_id, schema_version, status, state_json, created_at, updated_at) "
            "VALUES ('manual_rbk', 1, 'active', '{}', 0.0, 0.0)"
        )
        tx.rollback()
        tx.__exit__(None, None, None)

        cur = db._conn.execute("SELECT goal_id FROM goals WHERE goal_id='manual_rbk'")
        assert cur.fetchone() is None
    finally:
        close(db, path)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_max_redirects_constant():
    """MAX_GOAL_SESSION_REDIRECTS is a small positive integer."""
    assert isinstance(MAX_GOAL_SESSION_REDIRECTS, int)
    assert MAX_GOAL_SESSION_REDIRECTS >= 1
    assert MAX_GOAL_SESSION_REDIRECTS <= 10


# ---------------------------------------------------------------------------
# Column presence (reconciliation)
# ---------------------------------------------------------------------------

def test_goals_columns_reconciled():
    """All goals columns exist after init (via _reconcile_columns)."""
    db, path = mkdb()
    try:
        cur = db._conn.execute("PRAGMA table_info('goals')")
        cols = {row[1] for row in cur.fetchall()}
        required = {
            "goal_id", "schema_version", "revision", "status", "goal_text",
            "state_json", "source", "created_at", "updated_at", "completed_at",
            "archived_at", "archived_reason", "session_key", "created_by",
        }
        assert required <= cols, f"missing columns: {required - cols}"
    finally:
        close(db, path)


def test_goal_sessions_columns_reconciled():
    """All goal_sessions columns exist after init."""
    db, path = mkdb()
    try:
        cur = db._conn.execute("PRAGMA table_info('goal_sessions')")
        cols = {row[1] for row in cur.fetchall()}
        required = {
            "binding_id", "goal_id", "session_id", "session_key",
            "started_at", "ended_at", "end_reason", "redirect_to", "updated_at",
        }
        assert required <= cols, f"missing columns: {required - cols}"
    finally:
        close(db, path)
