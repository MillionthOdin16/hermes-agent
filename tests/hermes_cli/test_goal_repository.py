"""
Phase 2 tests for GoalRepository.

Verifies the canonical goal persistence layer in isolation — no runtime
paths (goals.v2_repository) are wired in these tests.

Tests cover:
  - set_active_goal: creates goal and binding
  - set_active_goal: replaces and archives existing goal
  - save_goal: revision increment
  - save_goal: rejects stale expected_revision
  - migrate_session: moves canonical binding
  - migrate_session: idempotent same-goal case
  - migrate_session: conflict does not overwrite
  - migrate_session: alias_window redirects old session
  - list_goals: filters by status
  - archive_goal: ends active bindings
  - repository events are emitted
  - duplicate active binding constraints hold
"""

import json
import os
import sqlite3
import tempfile
import time

import pytest

from hermes_cli.goal_core.repository import (
    GoalRepository,
    GoalRecord,
    GoalSummary,
    GoalRevisionConflict,
    GoalSessionConflict,
)
from hermes_cli.goals import GoalState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mkdb():
    """Make a fresh temporary SessionDB."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    from hermes_state import SessionDB
    from pathlib import Path
    db = SessionDB(Path(path))
    return db, path


def close(db, path):
    db.close()
    if os.path.exists(path):
        os.unlink(path)


# ---------------------------------------------------------------------------
# set_active_goal
# ---------------------------------------------------------------------------

def test_set_active_goal_creates_goal_and_binding():
    """set_active_goal inserts a goals row and a canonical goal_sessions binding.

    Initial revision is 0. The first save_goal call increments it to 1.
    """
    db, path = mkdb()
    try:
        repo = GoalRepository(db)
        state = GoalState(goal="Build the thing")
        state.created_at = time.time()

        rec = repo.set_active_goal("sess-abc", state, session_key="user-key")

        assert rec is not None
        assert rec.goal_id == state.goal_id
        assert rec.state.goal == "Build the thing"
        assert rec.state.status == "active"
        assert rec.session_id == "sess-abc"
        assert rec.session_key == "user-key"
        assert rec.revision == 0  # initial; save_goal bumps to 1

        # Verify binding
        binding = db._conn.execute(
            "SELECT * FROM goal_sessions WHERE goal_id=? AND ended_at IS NULL AND redirect_to IS NULL",
            (rec.goal_id,),
        ).fetchone()
        assert binding is not None
        assert binding["session_id"] == "sess-abc"

        # Verify event emitted
        events = db._conn.execute(
            "SELECT event_type FROM goal_events WHERE goal_id=?", (rec.goal_id,)
        ).fetchall()
        event_types = {r[0] for r in events}
        assert "goal_created" in event_types
        assert "session_bound" in event_types
    finally:
        close(db, path)


def test_set_active_goal_replaces_and_archives_existing_goal():
    """When a session already has an active goal, set_active_goal archives it and ends its binding."""
    db, path = mkdb()
    try:
        repo = GoalRepository(db)
        state1 = GoalState(goal="First goal")
        state1.created_at = time.time()
        rec1 = repo.set_active_goal("sess-abc", state1)

        state2 = GoalState(goal="Second goal")
        state2.created_at = time.time()
        rec2 = repo.set_active_goal("sess-abc", state2, archive_reason="replaced_by_user")

        # rec2 is the new active goal
        assert rec2.state.goal == "Second goal"
        assert rec2.session_id == "sess-abc"

        # Old goal is archived
        old_row = db._conn.execute(
            "SELECT status, archived_reason FROM goals WHERE goal_id=?", (rec1.goal_id,)
        ).fetchone()
        assert old_row["status"] == "archived"
        assert old_row["archived_reason"] == "replaced_by_user"

        # Old binding is ended with reason preserved through the archive_reason chain
        old_binding = db._conn.execute(
            "SELECT end_reason FROM goal_sessions WHERE goal_id=? AND ended_at IS NOT NULL",
            (rec1.goal_id,),
        ).fetchone()
        assert old_binding is not None
        assert old_binding["end_reason"] == "replaced_by_user"

        # Events for old goal
        old_events = db._conn.execute(
            "SELECT event_type FROM goal_events WHERE goal_id=?", (rec1.goal_id,)
        ).fetchall()
        old_event_types = {r[0] for r in old_events}
        assert "goal_archived" in old_event_types

        # Events for new goal — includes goal_replaced with old_goal_id
        new_events = db._conn.execute(
            "SELECT event_type, event_data FROM goal_events WHERE goal_id=?", (rec2.goal_id,)
        ).fetchall()
        new_event_types = {r[0] for r in new_events}
        assert "goal_replaced" in new_event_types
        replaced_row = next(r for r in new_events if r[0] == "goal_replaced")
        import json
        evt_data = json.loads(replaced_row[1]) if replaced_row[1] else {}
        assert evt_data.get("old_goal_id") == rec1.goal_id
        assert evt_data.get("reason") == "replaced_by_user"
    finally:
        close(db, path)


# ---------------------------------------------------------------------------
# save_goal
# ---------------------------------------------------------------------------

def test_save_goal_increments_revision():
    """save_goal bumps revision by exactly 1 each time."""
    db, path = mkdb()
    try:
        repo = GoalRepository(db)
        state = GoalState(goal="My goal")
        state.created_at = time.time()
        rec1 = repo.set_active_goal("sess-abc", state)

        state2 = GoalState.from_json(rec1.state.to_json())
        state2.goal_id = rec1.goal_id
        state2.revision = rec1.revision  # repository reads from row; this just sets it

        rec2 = repo.save_goal(rec1.goal_id, state2)

        assert rec2.revision == rec1.revision + 1

        # Third save
        state3 = GoalState.from_json(rec2.state.to_json())
        state3.goal_id = rec2.goal_id
        state3.revision = rec2.revision
        rec3 = repo.save_goal(rec2.goal_id, state3)
        assert rec3.revision == rec2.revision + 1
        assert rec3.revision == rec1.revision + 2
    finally:
        close(db, path)


def test_save_goal_rejects_stale_expected_revision():
    """save_goal raises GoalRevisionConflict when expected_revision doesn't match row."""
    db, path = mkdb()
    try:
        repo = GoalRepository(db)
        state = GoalState(goal="My goal")
        state.created_at = time.time()
        rec = repo.set_active_goal("sess-abc", state)

        # Bump row revision via another save
        state2 = GoalState.from_json(rec.state.to_json())
        state2.goal_id = rec.goal_id
        state2.revision = rec.revision
        repo.save_goal(rec.goal_id, state2)

        # Try to save with stale revision
        state3 = GoalState.from_json(rec.state.to_json())
        state3.goal_id = rec.goal_id
        state3.revision = rec.revision  # intentionally stale

        with pytest.raises(GoalRevisionConflict) as exc_info:
            repo.save_goal(rec.goal_id, state3, expected_revision=rec.revision)
        assert exc_info.value.goal_id == rec.goal_id
        assert exc_info.value.expected == rec.revision
        assert exc_info.value.actual == rec.revision + 1

        # Verify conflict event was emitted
        events = db._conn.execute(
            "SELECT event_type FROM goal_events WHERE goal_id=? AND event_type='revision_conflict'",
            (rec.goal_id,),
        ).fetchall()
        assert len(events) == 1
    finally:
        close(db, path)


# ---------------------------------------------------------------------------
# migrate_session
# ---------------------------------------------------------------------------

def test_migrate_session_moves_canonical_binding():
    """migrate_session ends old binding and creates new canonical binding for new session."""
    db, path = mkdb()
    try:
        repo = GoalRepository(db)
        state = GoalState(goal="Migrate me")
        state.created_at = time.time()
        rec = repo.set_active_goal("sess-old", state)

        result = repo.migrate_session("sess-old", "sess-new")

        assert result is not None
        assert result.goal_id == rec.goal_id
        assert result.session_id == "sess-new"

        # Old binding ended
        old_bind = db._conn.execute(
            "SELECT ended_at, end_reason FROM goal_sessions "
            "WHERE session_id='sess-old' AND ended_at IS NOT NULL"
        ).fetchone()
        assert old_bind is not None
        assert old_bind["end_reason"] == "session rollover"

        # New canonical binding exists
        new_bind = db._conn.execute(
            "SELECT session_id FROM goal_sessions "
            "WHERE session_id='sess-new' AND ended_at IS NULL AND redirect_to IS NULL"
        ).fetchone()
        assert new_bind is not None

        # session_rollover event
        events = db._conn.execute(
            "SELECT event_type FROM goal_events WHERE goal_id=? AND event_type='session_rollover'",
            (rec.goal_id,),
        ).fetchall()
        assert len(events) == 1
    finally:
        close(db, path)


def test_migrate_session_idempotent_when_old_has_no_binding():
    """migrate_session is idempotent: calling with an old_session_id that has no
    canonical active binding returns None (no error, no side effects)."""
    db, path = mkdb()
    try:
        repo = GoalRepository(db)
        state = GoalState(goal="Idempotent migrate")
        state.created_at = time.time()
        rec1 = repo.set_active_goal("sess-old", state)

        # First migrate ends the old binding
        rec2 = repo.migrate_session("sess-old", "sess-new")
        assert rec2.session_id == "sess-new"

        # Second migrate — old binding is already ended, returns None (not an error)
        rec3 = repo.migrate_session("sess-old", "sess-new")
        assert rec3 is None
    finally:
        close(db, path)


def test_migrate_session_conflict_does_not_overwrite():
    """If new_session has canonical binding for a different goal, migrate_session raises GoalSessionConflict."""
    db, path = mkdb()
    try:
        repo = GoalRepository(db)

        # Goal A on sess-a
        state_a = GoalState(goal="Goal A")
        state_a.created_at = time.time()
        repo.set_active_goal("sess-a", state_a)

        # Goal B on sess-b
        state_b = GoalState(goal="Goal B")
        state_b.created_at = time.time()
        repo.set_active_goal("sess-b", state_b)

        # Try to migrate Goal A → sess-b (conflict)
        from hermes_cli.goal_core.repository import GoalSessionConflict
        with pytest.raises(GoalSessionConflict) as exc_info:
            repo.migrate_session("sess-a", "sess-b")
        assert exc_info.value.existing_goal_id != exc_info.value.attempted_goal_id

        # sess-b binding is unchanged
        sess_b_binding = db._conn.execute(
            "SELECT goal_id FROM goal_sessions WHERE session_id='sess-b' AND ended_at IS NULL AND redirect_to IS NULL"
        ).fetchone()
        assert sess_b_binding["goal_id"] == state_b.goal_id

        # session_conflict event emitted
        events = db._conn.execute(
            "SELECT event_type FROM goal_events WHERE event_type='goal_session_conflict'"
        ).fetchall()
        assert len(events) == 1
    finally:
        close(db, path)


def test_migrate_session_alias_window_resolves_old_session():
    """alias_window=True keeps old binding alive as an alias (redirect_to) and creates new canonical."""
    db, path = mkdb()
    try:
        repo = GoalRepository(db)
        state = GoalState(goal="Alias window test")
        state.created_at = time.time()
        rec = repo.set_active_goal("sess-old", state)

        result = repo.migrate_session("sess-old", "sess-new", alias_window=True)

        assert result.session_id == "sess-new"

        # Old binding is now an alias (ended_at=NULL, redirect_to set)
        old_alias = db._conn.execute(
            "SELECT redirect_to, ended_at FROM goal_sessions WHERE session_id='sess-old' AND redirect_to IS NOT NULL"
        ).fetchone()
        assert old_alias is not None
        assert old_alias["redirect_to"] == "sess-new"
        assert old_alias["ended_at"] is None

        # New canonical binding
        new_bind = db._conn.execute(
            "SELECT session_id FROM goal_sessions "
            "WHERE session_id='sess-new' AND ended_at IS NULL AND redirect_to IS NULL"
        ).fetchone()
        assert new_bind is not None
    finally:
        close(db, path)


# ---------------------------------------------------------------------------
# list_goals
# ---------------------------------------------------------------------------

def test_list_goals_filters_by_status():
    """list_goals with status= filters correctly; default returns all."""
    db, path = mkdb()
    try:
        repo = GoalRepository(db)

        state_a = GoalState(goal="Active goal")
        state_a.created_at = time.time()
        repo.set_active_goal("sess-a", state_a)

        state_b = GoalState(goal="Another active")
        state_b.created_at = time.time()
        repo.set_active_goal("sess-b", state_b)

        state_c = GoalState(goal="Archived goal")
        state_c.created_at = time.time()
        rec_c = repo.set_active_goal("sess-c", state_c)
        repo.archive_goal(rec_c.goal_id)

        all_goals = repo.list_goals()
        active_goals = repo.list_goals(status="active")
        archived_goals = repo.list_goals(status="archived")

        # Default (no filter) returns all non-archived goals; SQL has WHERE status != 'archived'
        assert len(all_goals) >= 2
        # archived goal must NOT appear in default list
        archived_ids = {g.goal_id for g in archived_goals}
        assert all(g.goal_id not in archived_ids for g in all_goals)
        assert all(g.status == "active" for g in active_goals)
        assert all(g.status == "archived" for g in archived_goals)
        # status filter must return at least the one archived goal
        assert len(archived_goals) >= 1
    finally:
        close(db, path)


# ---------------------------------------------------------------------------
# archive_goal
# ---------------------------------------------------------------------------

def test_archive_goal_ends_active_bindings():
    """archive_goal sets row status=archived, ends all canonical bindings, emits events."""
    db, path = mkdb()
    try:
        repo = GoalRepository(db)
        state = GoalState(goal="Archive me")
        state.created_at = time.time()
        rec = repo.set_active_goal("sess-archive", state)

        repo.archive_goal(rec.goal_id, reason="user_requested")

        # Row is archived
        row = db._conn.execute(
            "SELECT status, archived_at, archived_reason FROM goals WHERE goal_id=?",
            (rec.goal_id,),
        ).fetchone()
        assert row["status"] == "archived"
        assert row["archived_reason"] == "user_requested"
        assert row["archived_at"] is not None

        # Binding ended with reason matching the archive_reason parameter
        binding = db._conn.execute(
            "SELECT ended_at, end_reason FROM goal_sessions WHERE goal_id=?",
            (rec.goal_id,),
        ).fetchone()
        assert binding["ended_at"] is not None
        assert binding["end_reason"] == "user_requested"

        # Events
        events = db._conn.execute(
            "SELECT event_type FROM goal_events WHERE goal_id=?", (rec.goal_id,)
        ).fetchall()
        event_types = {r[0] for r in events}
        assert "goal_archived" in event_types
        assert "session_unbound" in event_types
    finally:
        close(db, path)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def test_repository_events_are_emitted():
    """Every mutation emits at least one goal_events row."""
    db, path = mkdb()
    try:
        repo = GoalRepository(db)
        state = GoalState(goal="Event test")
        state.created_at = time.time()
        rec = repo.set_active_goal("sess-evt", state)

        events = db._conn.execute(
            "SELECT event_type FROM goal_events WHERE goal_id=?", (rec.goal_id,)
        ).fetchall()
        assert len(events) >= 2  # goal_created + session_bound at minimum
    finally:
        close(db, path)


# ---------------------------------------------------------------------------
# Constraint validation
# ---------------------------------------------------------------------------

def test_duplicate_active_binding_constraints_hold():
    """idx_one_active_goal_per_session: two canonical bindings for same session must not both exist."""
    db, path = mkdb()
    try:
        repo = GoalRepository(db)
        state = GoalState(goal="First")
        state.created_at = time.time()
        rec1 = repo.set_active_goal("sess-constraint", state)

        # Try to create a second canonical binding directly via SQL (simulating a bug)
        import uuid
        binding_id = f"bind-{uuid.uuid4().hex[:12]}"
        with pytest.raises(sqlite3.IntegrityError):
            db._conn.execute(
                """
                INSERT INTO goal_sessions
                  (binding_id, goal_id, session_id, session_key, started_at, ended_at,
                   end_reason, redirect_to, updated_at)
                VALUES
                  (:binding_id, :goal_id, :session_id, :session_key, :started_at, NULL,
                   NULL, NULL, :updated_at)
                """,
                {
                    "binding_id": binding_id,
                    "goal_id": f"goal-{uuid.uuid4().hex[:12]}",
                    "session_id": "sess-constraint",
                    "session_key": "key",
                    "started_at": time.time(),
                    "updated_at": time.time(),
                },
            )
    finally:
        close(db, path)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2.5: Serialisation / metadata correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_set_active_goal_persists_goal_id_and_revision_in_state_json():
    """goals.goal_id, goals.revision, and state_json all agree on identity."""
    db, path = mkdb()
    try:
        repo = GoalRepository(db)
        state = GoalState(goal="Metadata test")
        state.created_at = time.time()
        object.__setattr__(state, "goal_id", None)   # will be assigned by repo
        object.__setattr__(state, "revision", None)  # will be assigned by repo
        rec = repo.set_active_goal("sess-x", state)

        # Row columns
        row = db._conn.execute(
            "SELECT goal_id, revision, state_json FROM goals WHERE goal_id=?",
            (rec.goal_id,),
        ).fetchone()
        assert row["goal_id"] == rec.goal_id
        assert row["revision"] == 0

        # state_json round-trip
        data = json.loads(row["state_json"])
        assert data["goal_id"] == rec.goal_id, \
            "goal_id missing from state_json"
        assert data["revision"] == 0, \
            "revision missing/wrong in state_json"
    finally:
        close(db, path)


def test_save_goal_state_json_contains_incremented_revision():
    """After save_goal, state_json.revision matches the goals row and goal_state object."""
    db, path = mkdb()
    try:
        repo = GoalRepository(db)
        state = GoalState(goal="Save revision test")
        state.created_at = time.time()
        rec = repo.set_active_goal("sess-x", state)

        # Load and mutate
        loaded = repo.get_active_goal("sess-x")
        assert loaded is not None
        assert loaded.revision == 0
        object.__setattr__(loaded.state, "goal_id", rec.goal_id)
        object.__setattr__(loaded.state, "revision", 0)
        saved = repo.save_goal(rec.goal_id, loaded.state)

        assert saved.revision == 1

        row = db._conn.execute(
            "SELECT revision, state_json FROM goals WHERE goal_id=?",
            (rec.goal_id,),
        ).fetchone()
        assert row["revision"] == 1
        data = json.loads(row["state_json"])
        assert data["revision"] == 1, \
            f"state_json.revision is {data['revision']}, expected 1"
    finally:
        close(db, path)


def test_loaded_goal_record_restores_goal_id_revision_to_state_object():
    """A GoalRecord's state object has goal_id and revision attached via object.__setattr__."""
    db, path = mkdb()
    try:
        repo = GoalRepository(db)
        state = GoalState(goal="Round-trip test")
        state.created_at = time.time()
        rec = repo.set_active_goal("sess-x", state)

        # Simulate a GoalManager re-loading from repository
        reloaded = repo.get_active_goal("sess-x")
        assert reloaded is not None
        assert getattr(reloaded.state, "goal_id", None) == rec.goal_id
        assert getattr(reloaded.state, "revision", None) == 0

        # Modify and re-save
        object.__setattr__(reloaded.state, "goal_id", rec.goal_id)
        object.__setattr__(reloaded.state, "revision", 0)
        saved = repo.save_goal(rec.goal_id, reloaded.state)
        assert saved.revision == 1
        assert getattr(saved.state, "revision", None) == 1
    finally:
        close(db, path)


def test_archived_row_status_differs_from_state_json_status():
    """Archive sets goals.status='archived' but state_json.status='cleared'."""
    db, path = mkdb()
    try:
        repo = GoalRepository(db)
        state = GoalState(goal="Archive invariant test")
        state.created_at = time.time()
        rec = repo.set_active_goal("sess-x", state)
        goal_id = rec.goal_id

        repo.archive_goal(goal_id, reason="test archive")

        row = db._conn.execute(
            "SELECT status, state_json, archived_at FROM goals WHERE goal_id=?",
            (goal_id,),
        ).fetchone()
        assert row["status"] == "archived", \
            f"goals.status is {row['status']}, expected 'archived'"
        assert row["archived_at"] is not None, \
            "archived_at must be set"

        data = json.loads(row["state_json"])
        # archive_goal uses json_set(state_json, '$.status', 'cleared'), so the
        # state_json status is always 'cleared' after archival (goals.status row
        # is 'archived' — a separate invariant; see archive_goal docstring).
        assert data.get("status") == "cleared", \
            f"state_json.status is {data.get('status')}, expected 'cleared'"
    finally:
        close(db, path)


def test_save_goal_uses_atomic_revision_update_not_precheck_only():
    """A stale save_goal with expected_revision fails with atomic rowcount=0,
    even when a concurrent writer has already updated the row."""
    db, path = mkdb()
    try:
        repo_a = GoalRepository(db)
        repo_b = GoalRepository(db)

        state = GoalState(goal="TOCTOU test")
        state.created_at = time.time()
        rec = repo_a.set_active_goal("sess-a", state)
        goal_id = rec.goal_id

        # Simulate stale writer using the initial revision=0
        stale_state = GoalState.from_json(rec.state.to_json())
        object.__setattr__(stale_state, "goal_id", goal_id)
        object.__setattr__(stale_state, "revision", 0)

        # Concurrent writer B updates revision to 1 (this also injects into JSON)
        concurrent_state = GoalState.from_json(rec.state.to_json())
        object.__setattr__(concurrent_state, "goal_id", goal_id)
        object.__setattr__(concurrent_state, "revision", 0)
        saved = repo_b.save_goal(goal_id, concurrent_state)
        assert saved.revision == 1

        # Writer A's stale revision=0 must be rejected
        with pytest.raises(GoalRevisionConflict) as exc_info:
            repo_a.save_goal(goal_id, stale_state, expected_revision=0)
        assert exc_info.value.expected == 0
        assert exc_info.value.actual == 1

        # No spurious goal_updated event from the rejected writer
        updated_events = db._conn.execute(
            "SELECT event_type FROM goal_events WHERE event_type='goal_updated'"
        ).fetchall()
        assert len(updated_events) == 1, \
            f"expected 1 goal_updated, got {len(updated_events)}"
    finally:
        close(db, path)


def test_save_goal_event_not_emitted_on_conflict():
    """When save_goal raises GoalRevisionConflict, no goal_updated event is emitted."""
    db, path = mkdb()
    try:
        repo_a = GoalRepository(db)
        repo_b = GoalRepository(db)

        state = GoalState(goal="No-event-on-conflict test")
        state.created_at = time.time()
        rec = repo_a.set_active_goal("sess-a", state)
        goal_id = rec.goal_id

        # Writer B moves revision to 1
        s_b = GoalState.from_json(rec.state.to_json())
        object.__setattr__(s_b, "goal_id", goal_id)
        object.__setattr__(s_b, "revision", 0)
        repo_b.save_goal(goal_id, s_b)

        # Writer A tries stale save — raises but must not emit goal_updated
        s_a = GoalState.from_json(rec.state.to_json())
        object.__setattr__(s_a, "goal_id", goal_id)
        object.__setattr__(s_a, "revision", 0)
        with pytest.raises(GoalRevisionConflict):
            repo_a.save_goal(goal_id, s_a, expected_revision=0)

        updated_events = db._conn.execute(
            "SELECT event_type FROM goal_events WHERE event_type='goal_updated'"
        ).fetchall()
        # Only the successful write from B
        assert len(updated_events) == 1
    finally:
        close(db, path)
