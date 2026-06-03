"""
Integration tests for GoalManager repository wiring.

Verifies that when goals.v2_repository=True, GoalManager uses the
GoalRepository-backed storage path, and when False, uses the legacy
state_meta path. Tests cover load/save, migrate-on-read, replacement,
parallel sessions, and migrate_session.

Isolation strategy:
  - Each test gets a fresh temp db file.
  - _get_session_db is monkeypatched to return that db.
  - set_v2_enabled_for_test / clear_v2_enabled_for_test control the flag.
  - _v2_repo_cache is Noneed before each test to avoid cross-test leakage.
"""

import os
import tempfile
import time

import pytest

from hermes_cli.goal_core.repository import GoalRepository
from hermes_cli.goal_core.repository import set_v2_enabled_for_test
from hermes_cli.goal_core.repository import clear_v2_enabled_for_test
from hermes_cli.goals import GoalManager, load_goal, save_goal, clear_goal
from hermes_cli.goals import migrate_goal_session
from hermes_cli.goals import GoalState, GoalStatus
from hermes_state import SessionDB
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_caches():
    """Clear all inter-test caches before and after each test."""
    import hermes_cli.goals as gm
    from hermes_cli.goal_core.repository import clear_v2_enabled_for_test

    # Reset before test
    gm._DB_CACHE.clear()
    clear_v2_enabled_for_test()  # Clear _v2_override so test_goals.py runs with config-based lookup

    yield

    # Reset after test
    gm._DB_CACHE.clear()
    clear_v2_enabled_for_test()


@pytest.fixture
def tmp_db(tmp_path):
    """A fresh temporary SessionDB isolated from the real hermes home.

    Returns (db, db_path) — caller must close and unlink db_path.
    """
    db_path = tmp_path / "test_state.db"
    # Remove if exists from prior run
    if db_path.exists():
        db_path.unlink()
    db = SessionDB(db_path)
    yield db, str(db_path)
    db.close()
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def repo_on_tmp_db(tmp_db):
    """A GoalRepository backed by tmp_db's SessionDB, with v2=true."""
    db, _ = tmp_db
    set_v2_enabled_for_test(True)
    repo = GoalRepository(db=db)
    yield repo, db
    clear_v2_enabled_for_test()


# ---------------------------------------------------------------------------
# Helper: patch _get_session_db to return a specific db
# ---------------------------------------------------------------------------

def _patch_get_session_db(monkeypatch, db, hermes_home_path):
    """Make _get_session_db return `db` for `hermes_home_path`."""
    import hermes_cli.goals as gm

    def fake_get_session_db():
        return db

    monkeypatch.setattr(gm, "_get_session_db", fake_get_session_db)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_goal_manager_uses_legacy_storage_when_flag_false(tmp_db, monkeypatch):
    """GoalManager.set() writes to state_meta when v2_repository=False."""
    db, db_path = tmp_db
    set_v2_enabled_for_test(False)
    _patch_get_session_db(monkeypatch, db, str(Path(db_path).parent))

    gm = GoalManager("sess-legacy")
    gm.set("Legacy stored goal")

    # state_meta row exists
    row = db._conn.execute(
        "SELECT value FROM state_meta WHERE key=?", ("goal:sess-legacy",)
    ).fetchone()
    assert row is not None
    assert "Legacy stored goal" in row[0]

    # No goals table row (v2 disabled)
    goals_count = db._conn.execute("SELECT COUNT(*) FROM goals").fetchone()[0]
    assert goals_count == 0

    clear_v2_enabled_for_test()


def test_goal_manager_loads_repository_goal_when_flag_true(tmp_db, monkeypatch):
    """GoalManager.load returns repository-backed goal when v2_repository=True."""
    db, db_path = tmp_db
    set_v2_enabled_for_test(True)
    _patch_get_session_db(monkeypatch, db, str(Path(db_path).parent))

    # Pre-populate repository directly
    repo = GoalRepository(db=db)
    state = GoalState(goal="Repo-backed goal")
    state.created_at = time.time()
    repo.set_active_goal("sess-v2", state)

    # load_goal reads via repository
    gm = GoalManager("sess-v2")
    assert gm._state is not None
    assert gm._state.goal == "Repo-backed goal"
    assert gm._state.status == GoalStatus.ACTIVE.value

    # state_meta not used
    row = db._conn.execute(
        "SELECT value FROM state_meta WHERE key=?", ("goal:sess-v2",)
    ).fetchone()
    assert row is None

    clear_v2_enabled_for_test()


def test_goal_manager_saves_repository_goal_when_flag_true(tmp_db, monkeypatch):
    """GoalManager.set() writes to repository when v2_repository=True."""
    db, db_path = tmp_db
    set_v2_enabled_for_test(True)
    _patch_get_session_db(monkeypatch, db, str(Path(db_path).parent))

    gm = GoalManager("sess-v2-save")
    gm.set("Saved via GoalManager")

    # Verify via repository
    repo = GoalRepository(db=db)
    rec = repo.get_active_goal("sess-v2-save")
    assert rec is not None
    assert rec.state.goal == "Saved via GoalManager"

    # state_meta not touched
    row = db._conn.execute(
        "SELECT value FROM state_meta WHERE key=?", ("goal:sess-v2-save",)
    ).fetchone()
    assert row is None

    clear_v2_enabled_for_test()


def test_goal_manager_migrates_legacy_state_meta_on_first_load_when_flag_true(
    tmp_db, monkeypatch
):
    """First GoalManager init with v2=true triggers migrate-on-read from state_meta."""
    db, db_path = tmp_db

    # Put a legacy state_meta goal directly in DB
    legacy_key = "goal:sess-legacy-migrate"
    legacy_value = '{"goal":"Migrated from legacy","status":"active","created_at":%s}' % time.time()
    db._conn.execute(
        "INSERT INTO state_meta (key, value) VALUES (?, ?)",
        (legacy_key, legacy_value),
    )
    db._conn.commit()

    set_v2_enabled_for_test(True)
    _patch_get_session_db(monkeypatch, db, str(Path(db_path).parent))

    # First GoalManager init should migrate-on-read
    gm = GoalManager("sess-legacy-migrate")
    assert gm._state is not None
    assert gm._state.goal == "Migrated from legacy"

    # Repository has the goal
    repo = GoalRepository(db=db)
    rec = repo.get_active_goal("sess-legacy-migrate")
    assert rec is not None

    # Sidecar recorded
    sidecar = db._conn.execute(
        "SELECT * FROM goal_legacy_migrations WHERE legacy_key=?",
        (legacy_key,),
    ).fetchone()
    assert sidecar is not None

    # Original state_meta not mutated
    orig = db._conn.execute(
        "SELECT value FROM state_meta WHERE key=?", (legacy_key,)
    ).fetchone()
    assert orig[0] == legacy_value

    clear_v2_enabled_for_test()


def test_goal_manager_parallel_sessions_isolated_with_repository(tmp_db, monkeypatch):
    """Two sessions with active goals do not interfere."""
    db, db_path = tmp_db
    set_v2_enabled_for_test(True)
    _patch_get_session_db(monkeypatch, db, str(Path(db_path).parent))

    gm_a = GoalManager("sess-a")
    gm_a.set("Session A goal")

    gm_b = GoalManager("sess-b")
    gm_b.set("Session B goal")

    # Re-load and verify isolation
    gm_a2 = GoalManager("sess-a")
    gm_b2 = GoalManager("sess-b")
    assert gm_a2._state.goal == "Session A goal"
    assert gm_b2._state.goal == "Session B goal"

    # Replace sess-a's goal — sess-b unchanged
    repo = GoalRepository(db=db)
    state_c = GoalState(goal="Session A replaced")
    state_c.created_at = time.time()
    repo.set_active_goal("sess-a", state_c)

    gm_b3 = GoalManager("sess-b")
    assert gm_b3._state.goal == "Session B goal"

    clear_v2_enabled_for_test()


def test_goal_manager_replacement_archives_previous_goal_with_repository(tmp_db, monkeypatch):
    """Calling set() on a session with an existing goal archives the old one."""
    db, db_path = tmp_db
    set_v2_enabled_for_test(True)
    _patch_get_session_db(monkeypatch, db, str(Path(db_path).parent))

    gm1 = GoalManager("sess-replace")
    gm1.set("First goal")

    gm2 = GoalManager("sess-replace")
    gm2.set("Second goal")

    # First goal is now archived
    repo = GoalRepository(db=db)
    archived = repo.list_goals(status="archived")
    assert len(archived) >= 1
    first_archived = any(g.goal_text == "First goal" for g in archived)
    assert first_archived

    # Second goal is active
    active = repo.list_goals(status="active")
    assert any(g.goal_text == "Second goal" for g in active)

    # Old binding end_reason is "replaced_by_user" or "replaced by set_active_goal"
    old_binding = db._conn.execute(
        "SELECT end_reason FROM goal_sessions gs "
        "JOIN goals g ON g.goal_id = gs.goal_id "
        "WHERE g.goal_text=? AND gs.ended_at IS NOT NULL",
        ("First goal",),
    ).fetchone()
    assert old_binding is not None
    assert old_binding["end_reason"] in (
        "replaced_by_user",
        "replaced by set_active_goal",
    )

    # goal_replaced event exists on the new goal
    new_rec = next(g for g in active if g.goal_text == "Second goal")
    events = db._conn.execute(
        "SELECT event_type FROM goal_events WHERE goal_id=?",
        (new_rec.goal_id,),
    ).fetchall()
    event_types = {r[0] for r in events}
    assert "goal_replaced" in event_types

    clear_v2_enabled_for_test()


def test_migrate_goal_session_uses_repository_when_flag_true(tmp_db, monkeypatch):
    """migrate_goal_session delegates to repository when v2=true."""
    db, db_path = tmp_db
    set_v2_enabled_for_test(True)
    _patch_get_session_db(monkeypatch, db, str(Path(db_path).parent))

    repo = GoalRepository(db=db)
    state = GoalState(goal="Migrate this goal")
    state.created_at = time.time()
    rec = repo.set_active_goal("old-session", state)

    result = migrate_goal_session("old-session", "new-session")
    assert result is True

    # New session has the goal
    new_rec = repo.get_active_goal("new-session")
    assert new_rec is not None
    assert new_rec.goal_id == rec.goal_id

    # Old binding ended
    old_bind = db._conn.execute(
        "SELECT ended_at, end_reason FROM goal_sessions "
        "WHERE session_id='old-session' AND ended_at IS NOT NULL"
    ).fetchone()
    assert old_bind is not None
    assert old_bind["ended_at"] is not None

    clear_v2_enabled_for_test()


def test_migrate_goal_session_legacy_path_when_flag_false(tmp_db, monkeypatch):
    """migrate_goal_session uses state_meta path when v2=false."""
    db, db_path = tmp_db
    set_v2_enabled_for_test(False)
    _patch_get_session_db(monkeypatch, db, str(Path(db_path).parent))

    gm = GoalManager("old-sess")
    gm.set("Legacy migrate goal")

    result = migrate_goal_session("old-sess", "new-sess")
    assert result is True

    # Goal under new session in state_meta
    new_row = db._conn.execute(
        "SELECT value FROM state_meta WHERE key=?", ("goal:new-sess",)
    ).fetchone()
    assert new_row is not None
    assert "Legacy migrate goal" in new_row[0]

    # Old session cleared
    old_row = db._conn.execute(
        "SELECT value FROM state_meta WHERE key=?", ("goal:old-sess",)
    ).fetchone()
    if old_row:
        old_state = GoalState.from_json(old_row[0])
        assert old_state.status == GoalStatus.CLEARED.value

    clear_v2_enabled_for_test()


# ---------------------------------------------------------------------------
# Tests: no silent fallback to legacy on v2 repository error
# ---------------------------------------------------------------------------


def test_goal_manager_v2_save_does_not_silently_fallback_to_legacy_on_repo_error(tmp_db, monkeypatch):
    """save_goal raises when v2 enabled and GoalRepository fails, no silent legacy write."""
    db, db_path = tmp_db
    set_v2_enabled_for_test(True)
    _patch_get_session_db(monkeypatch, db, str(Path(db_path).parent))

    # Inject a goal into the DB and let the repository fail on save
    gm = GoalManager("test-sess")
    gm.set("V2 save error goal")
    state = gm._state
    assert state is not None, "GoalManager.set() should populate _state"

    # Monkeypatch GoalRepository.save_goal to raise
    from hermes_cli.goal_core import repository as repo_mod
    orig_save = repo_mod.GoalRepository.save_goal

    def failing_save(self, goal_id, state):
        raise RuntimeError("Repo save failed for test")

    monkeypatch.setattr(repo_mod.GoalRepository, "save_goal", failing_save)

    # save_goal should raise because v2 path does not fallback
    with pytest.raises(RuntimeError, match="Repo save failed for test"):
        save_goal("test-sess", state)

    # Verify no state_meta was written (silent fallback did not happen)
    legacy_row = db._conn.execute(
        "SELECT value FROM state_meta WHERE key=?", ("goal:test-sess",)
    ).fetchone()
    assert legacy_row is None, "state_meta should not be written on v2 repo error"

    clear_v2_enabled_for_test()


def test_goal_manager_v2_load_does_not_silently_fallback_to_legacy_on_repo_error(tmp_db, monkeypatch):
    """load_goal returns None when v2 enabled and GoalRepository fails, no silent fallback."""
    db, db_path = tmp_db
    set_v2_enabled_for_test(True)
    _patch_get_session_db(monkeypatch, db, str(Path(db_path).parent))

    # Pre-write a legacy state_meta entry to ensure we don't silently fallback
    from hermes_cli.goals import GoalState, GoalStatus
    legacy_state = GoalState(goal="legacy preload", status=GoalStatus.ACTIVE.value)
    db.set_meta("goal:test-sess", legacy_state.to_json())

    # Monkeypatch GoalRepository.get_active_goal to raise
    from hermes_cli.goal_core import repository as repo_mod
    def failing_get_active_goal(self, session_id):
        raise RuntimeError("Repo load failed for test")

    monkeypatch.setattr(repo_mod.GoalRepository, "get_active_goal", failing_get_active_goal)

    # load_goal should NOT fallback to legacy on error
    result = load_goal("test-sess")
    assert result is None, "load_goal should return None on v2 repo error, not fallback to legacy"

    clear_v2_enabled_for_test()


def test_goal_manager_v2_migrate_does_not_silently_fallback_to_legacy_on_repo_error(tmp_db, monkeypatch):
    """migrate_goal_session returns False when v2 enabled and GoalRepository fails, no silent legacy write."""
    db, db_path = tmp_db
    set_v2_enabled_for_test(True)
    _patch_get_session_db(monkeypatch, db, str(Path(db_path).parent))

    # Set up an active goal in the repository
    gm = GoalManager("old-sess")
    gm.set("V2 migrate error goal")
    state = gm._state

    # Monkeypatch GoalRepository.migrate_session to raise
    from hermes_cli.goal_core import repository as repo_mod
    def failing_migrate(self, old_sid, new_sid, reason="session_rollover"):
        raise RuntimeError("Repo migrate failed for test")

    monkeypatch.setattr(repo_mod.GoalRepository, "migrate_session", failing_migrate)

    result = migrate_goal_session("old-sess", "new-sess")
    assert result is False, "migrate_goal_session should return False on v2 repo error"

    # Verify no state_meta was written under new session (silent fallback did not happen)
    new_legacy = db._conn.execute(
        "SELECT value FROM state_meta WHERE key=?", ("goal:new-sess",)
    ).fetchone()
    assert new_legacy is None, "state_meta should not be written on v2 repo error during migrate"

    clear_v2_enabled_for_test()
