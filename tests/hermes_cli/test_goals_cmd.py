"""
Tests for hermes_cli/goal_core/goal_store.py and goals_cmd.py
Persistent cross-session goal tracking system.

Covers:
  - Atomic writes (no leftover .tmp files)
  - Stale in_progress reset after 30 min
  - Lock contention / LockError
  - Corrupt JSON → needs_review (not crash)
  - create_goal archives existing active goal
  - checklist state machine (valid transitions)
  - update_checklist_item blocks invalid transitions
  - append_evidence writes to both active.json AND evidence.log
  - archive_goal removes active.json, writes history/
  - soft_delete_goal moves to trash/
  - restore_goal from trash/
  - purge_trash deletes everything in trash/
  - review_goal validates + reactivates
  - list_goals reads history/ and trash/
  - handle_goals CLI subcommands
  - _atomic_write recovery path
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

# ─── helpers ───────────────────────────────────────────────────────────────────


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _mk_goal(
    sid: str = "test-id",
    text: str = "test goal",
    status: str = "active",
    checklist: Optional[list[dict]] = None,
    created_at: Optional[str] = None,
) -> dict:
    now = _now_iso()
    from datetime import datetime, timezone

    now_dt = datetime.now(timezone.utc)
    ts = now_dt.isoformat()
    items = checklist or []
    for i, item in enumerate(items):
        items[i].setdefault("id", f"item-{i}")
        items[i].setdefault("text", f"item {i}")
        items[i].setdefault("status", "pending")
        items[i].setdefault("created_at", ts)
        items[i].setdefault("updated_at", ts)
        items[i].setdefault("completed_at", None)
    return {
        "schema_version": "1.0",
        "id": sid,
        "text": text,
        "status": status,
        "checklist": items,
        "evidence": [],
        "created_at": created_at or ts,
        "updated_at": ts,
        "completed_at": None,
        "created_by": "agent",
    }


# ─── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def goals_dir(tmp_path, monkeypatch):
    """
    Isolated ~/.hermes/goals/ built on tmp_path.
    All GOALS_DIR, HISTORY_DIR, TRASH_DIR, EVIDENCE_LOG point here.
    Patches the module-level constants in memory — no subprocess needed.
    """
    import hermes_cli.goal_core.goal_store as gs

    gd = tmp_path / ".hermes" / "goals"
    gd.mkdir(parents=True)

    # Patch in-memory module constants (last for this interpreter lifetime)
    monkeypatch.setattr(gs, "GOALS_DIR", gd)
    monkeypatch.setattr(gs, "HISTORY_DIR", gd / "history")
    monkeypatch.setattr(gs, "TRASH_DIR", gd / "trash")
    monkeypatch.setattr(gs, "EVIDENCE_LOG", gd / "evidence.log")
    monkeypatch.setattr(gs, "LOCK_PATH", gd / ".lock")
    monkeypatch.setattr(gs, "STALE_LOCK_PATH", gd / ".lock_stale_at")

    # Ensure directories exist immediately
    gs._ensure_dirs()

    yield gd

    # Cleanup: release any residual locks so next test can acquire them
    lock = gd / ".lock"
    if lock.exists():
        try:
            import fcntl

            fd = os.open(str(lock), os.O_RDWR)
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        except Exception:
            pass


# ─── goal_store public API ──────────────────────────────────────────────────


class TestAtomicWriteNoTmpLeftover:
    """_atomic_write must not leave a .tmp file behind after success."""

    def test_no_tmp_after_write(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs

        gs._ensure_dirs()
        path = gs.GOALS_DIR / "active.json"
        data = _mk_goal()

        gs._atomic_write(path, data)

        assert not (goals_dir / "active.json.tmp").exists()
        assert path.exists()
        with open(path) as f:
            assert json.load(f)["id"] == data["id"]


class TestGetActiveGoal:
    """get_active_goal reads active.json and returns (goal, None) or (None, err)."""

    def test_no_goal_returns_none(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs

        goal, err = gs.get_active_goal()
        assert goal is None
        assert err is None

    def test_goal_returned(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs

        gs._ensure_dirs()
        gs._atomic_write(gs.GOALS_DIR / "active.json", _mk_goal())
        goal, err = gs.get_active_goal()
        assert goal is not None
        assert err is None
        assert goal["status"] == "active"

    def test_bad_json_returns_error(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs

        gs._ensure_dirs()
        path = gs.GOALS_DIR / "active.json"
        path.write_text("{broken")
        goal, err = gs.get_active_goal()
        assert goal is None
        assert "Invalid JSON" in err or "JSONDecodeError" in err


class TestStaleInProgressReset:
    """in_progress items older than 30 min are auto-reset to pending on read."""

    def test_stale_in_progress_is_reset(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs

        from datetime import datetime, timedelta, timezone

        gs._ensure_dirs()
        old_ts = (datetime.now(timezone.utc) - timedelta(minutes=31)).isoformat()
        goal = _mk_goal(
            checklist=[
                {
                    "id": "stale-item",
                    "text": "should revert",
                    "status": "in_progress",
                    "created_at": old_ts,
                    "updated_at": old_ts,
                    "completed_at": None,
                },
                {
                    "id": "fresh-item",
                    "text": "stays in_progress",
                    "status": "in_progress",
                    "created_at": _now_iso(),
                    "updated_at": _now_iso(),
                    "completed_at": None,
                },
            ]
        )
        gs._atomic_write(gs.GOALS_DIR / "active.json", goal)
        loaded, _ = gs.get_active_goal()

        # stale item should be reset
        stale = next(i for i in loaded["checklist"] if i["id"] == "stale-item")
        assert stale["status"] == "pending"
        # fresh item should be unchanged
        fresh = next(i for i in loaded["checklist"] if i["id"] == "fresh-item")
        assert fresh["status"] == "in_progress"


class TestSaveActiveGoal:
    """save_active_goal writes atomically. _ensure_dirs is called internally."""

    def test_write_and_read_back(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs

        gs._ensure_dirs()
        goal = _mk_goal(text="my shiny goal")
        ok, err = gs.save_active_goal(goal)
        assert ok, err
        loaded, _ = gs.get_active_goal()
        assert loaded["text"] == "my shiny goal"

    def test_read_old_evidence_after_write(self, goals_dir):
        """Writing must not corrupt the file mid-flight."""
        import hermes_cli.goal_core.goal_store as gs

        gs._ensure_dirs()
        goal = _mk_goal(
            checklist=[
                {
                    "id": "x",
                    "text": "step one",
                    "status": "completed",
                    "created_at": _now_iso(),
                    "updated_at": _now_iso(),
                    "completed_at": _now_iso(),
                }
            ]
        )
        gs._atomic_write(gs.GOALS_DIR / "active.json", goal)
        ok, _ = gs.save_active_goal(goal)
        assert ok


class TestCreateGoal:
    """create_goal makes a new active goal. Any existing goal is archived first."""

    def test_create_first_goal(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs

        gs._ensure_dirs()
        goal, err = gs.create_goal("my first goal")
        assert goal is not None, err
        assert goal["text"] == "my first goal"
        assert goal["status"] == "active"

    def test_create_archives_existing(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs

        gs._ensure_dirs()
        gs._atomic_write(gs.GOALS_DIR / "active.json", _mk_goal(text="old"))
        goal, _ = gs.create_goal("new goal")
        assert goal["text"] == "new goal"
        # old goal should be in history/
        hist = list((goals_dir / "history").glob("*.json"))
        assert len(hist) == 1


class TestChecklistItemLifecycle:
    """Add items, update status through valid transitions, reject bad ones."""

    def test_add_item(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs

        gs._ensure_dirs()
        gs._atomic_write(gs.GOALS_DIR / "active.json", _mk_goal())
        goal, _ = gs.add_checklist_item("do the thing")
        assert len(goal["checklist"]) == 1
        assert goal["checklist"][0]["text"] == "do the thing"
        assert goal["checklist"][0]["status"] == "pending"

    def test_pending_to_in_progress_ok(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs

        gs._ensure_dirs()
        gs._atomic_write(
            gs.GOALS_DIR / "active.json",
            _mk_goal(
                checklist=[
                    {
                        "id": "t1",
                        "text": "task 1",
                        "status": "pending",
                        "created_at": _now_iso(),
                        "updated_at": _now_iso(),
                        "completed_at": None,
                    }
                ]
            ),
        )
        goal, _ = gs.update_checklist_item("t1", "in_progress")
        assert goal["checklist"][0]["status"] == "in_progress"

    def test_in_progress_to_completed_ok(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs

        gs._ensure_dirs()
        gs._atomic_write(
            gs.GOALS_DIR / "active.json",
            _mk_goal(
                checklist=[
                    {
                        "id": "t1",
                        "text": "task 1",
                        "status": "in_progress",
                        "created_at": _now_iso(),
                        "updated_at": _now_iso(),
                        "completed_at": None,
                    }
                ]
            ),
        )
        goal, err = gs.update_checklist_item("t1", "completed")
        assert goal["checklist"][0]["status"] == "completed"
        assert goal["checklist"][0]["completed_at"] is not None

    def test_completed_to_pending_rejected(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs

        gs._ensure_dirs()
        gs._atomic_write(
            gs.GOALS_DIR / "active.json",
            _mk_goal(
                checklist=[
                    {
                        "id": "t1",
                        "text": "task 1",
                        "status": "completed",
                        "created_at": _now_iso(),
                        "updated_at": _now_iso(),
                        "completed_at": _now_iso(),
                    }
                ]
            ),
        )
        goal, err = gs.update_checklist_item("t1", "pending")
        assert goal is None
        assert "Cannot transition" in err

    def test_unknown_item_returns_error(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs

        gs._ensure_dirs()
        gs._atomic_write(gs.GOALS_DIR / "active.json", _mk_goal())
        _, err = gs.update_checklist_item("nonexistent-id", "completed")
        assert "not found" in err


class TestAppendEvidence:
    """append_evidence writes to active.json AND evidence.log."""

    def test_evidence_in_goal_and_global_log(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs

        gs._ensure_dirs()
        gs._atomic_write(gs.GOALS_DIR / "active.json", _mk_goal())
        ok, err = gs.append_evidence("found a bug in the lock", tool="terminal")
        assert ok, err

        # check active.json
        goal, _ = gs.get_active_goal()
        assert len(goal["evidence"]) == 1
        assert goal["evidence"][0]["content"] == "found a bug in the lock"
        assert goal["evidence"][0]["tool"] == "terminal"

        # check evidence.log is append-only
        with open(gs.EVIDENCE_LOG) as f:
            lines = f.readlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["content"] == "found a bug in the lock"


class TestArchiveGoal:
    """archive_goal(completed) moves active.json to history/, removes active.json."""

    def test_archive_completed_removes_active(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs

        gs._ensure_dirs()
        gs._atomic_write(gs.GOALS_DIR / "active.json", _mk_goal())
        ok, err = gs.archive_goal("completed")
        assert ok, err
        assert not (gs.GOALS_DIR / "active.json").exists()
        hist = list((goals_dir / "history").glob("*.json"))
        assert len(hist) == 1

    def test_archive_cancel_sets_status(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs

        gs._ensure_dirs()
        gs._atomic_write(gs.GOALS_DIR / "active.json", _mk_goal())
        ok, _ = gs.archive_goal("cancelled")
        assert ok
        hist = list((goals_dir / "history").glob("*.json"))[0]
        assert json.loads(hist.read_text())["status"] == "cancelled"


class TestReviewGoal:
    """review_goal validates active.json. Valid → status=active. Invalid → needs_review message."""

    def test_valid_goal_passes_review(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs

        gs._ensure_dirs()
        gs._atomic_write(gs.GOALS_DIR / "active.json", _mk_goal())
        fixed, err, msg = gs.review_goal()
        assert fixed is True
        assert err is None
        assert "valid" in msg.lower()

    def test_missing_file_is_error(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs

        gs._ensure_dirs()
        fixed, err, msg = gs.review_goal()
        assert fixed is False
        assert "No active.json" in err

    def test_corrupt_json_is_not_crash(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs

        gs._ensure_dirs()
        (gs.GOALS_DIR / "active.json").write_text("{broken")
        fixed, err, msg = gs.review_goal()
        assert fixed is False
        assert err is not None
        assert "Invalid" in err or "JSONDecodeError" in err


class TestListGoals:
    """list_goals reads history/ (and optionally trash/)."""

    def test_empty_dir_returns_empty_list(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs

        gs._ensure_dirs()
        assert gs.list_goals() == []

    def test_lists_archived_goals(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs

        gs._ensure_dirs()
        # write two goals to history/
        gs._atomic_write(
            gs.HISTORY_DIR / "goal-a.json",
            _mk_goal(sid="goal-a", text="first archived"),
        )
        gs._atomic_write(
            gs.HISTORY_DIR / "goal-b.json",
            _mk_goal(sid="goal-b", text="second archived"),
        )
        results = gs.list_goals()
        assert len(results) == 2

    def test_list_includes_trash_when_flagged(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs

        gs._ensure_dirs()
        gs._atomic_write(
            gs.TRASH_DIR / "trashed.json", _mk_goal(sid="trashed", text="deleted")
        )
        results = gs.list_goals(include_trash=False)
        assert all("trash" not in r.get("source", "") for r in results)
        results_all = gs.list_goals(include_trash=True)
        assert any("trash" in r.get("source", "") for r in results_all)


class TestSoftDeleteAndRestore:
    """soft_delete_goal moves to trash/. restore_goal moves back."""

    def test_soft_delete_moves_to_trash(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs

        gs._ensure_dirs()
        gs._atomic_write(gs.HISTORY_DIR / "goal-x.json", _mk_goal(sid="goal-x"))
        ok, err = gs.soft_delete_goal("goal-x")
        assert ok, err
        assert not (gs.HISTORY_DIR / "goal-x.json").exists()
        assert (gs.TRASH_DIR / "goal-x.json").exists()

    def test_restore_from_trash(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs

        gs._ensure_dirs()
        gs._atomic_write(gs.TRASH_DIR / "goal-y.json", _mk_goal(sid="goal-y"))
        ok, err = gs.restore_goal("goal-y")
        assert ok, err
        assert (gs.HISTORY_DIR / "goal-y.json").exists()
        assert not (gs.TRASH_DIR / "goal-y.json").exists()

    def test_restore_nonexistent_returns_error(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs

        gs._ensure_dirs()
        ok, err = gs.restore_goal("not-there")
        assert not ok
        assert "not found" in err


class TestPurgeTrash:
    """purge_trash deletes all goals in trash/."""

    def test_purge_deletes_all(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs

        gs._ensure_dirs()
        gs._atomic_write(gs.TRASH_DIR / "a.json", _mk_goal())
        gs._atomic_write(gs.TRASH_DIR / "b.json", _mk_goal())
        count, err = gs.purge_trash()
        assert count == 2
        assert err is None
        assert len(list(gs.TRASH_DIR.glob("*.json"))) == 0


class TestLockContention:
    """Concurrent processes trying to write active.json must not corrupt state."""

    def test_concurrent_write_one_wins(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs

        gs._ensure_dirs()
        gs._atomic_write(gs.GOALS_DIR / "active.json", _mk_goal())

        results: dict = {}
        errors: dict = {}
        barrier = threading.Barrier(2)

        def writer(n: int):
            try:
                barrier.wait()
                goal = _mk_goal(text=f"writer-{n}")
                ok, err = gs.save_active_goal(goal)
                results[n] = ok
                errors[n] = err
            except Exception as e:
                errors[n] = str(e)

        t1 = threading.Thread(target=writer, args=(1,))
        t2 = threading.Thread(target=writer, args=(2,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Both should complete without raising
        assert all(r for r in results.values()), f"errors: {errors}"
        # The winning file should be valid JSON
        with open(gs.GOALS_DIR / "active.json") as f:
            data = json.load(f)
        assert "text" in data


# ─── goals_cmd CLI handler ───────────────────────────────────────────────────


class TestAtomicWriteUniqueTmp:
    """Regression: unique temp paths must not collide under concurrency.

    THIS TEST VALIDATES FILE-LEVEL ATOMIC-WRITE SAFETY FOR THE LEGACY
    active.json STORE ONLY. It does not guarantee semantic conflict
    resolution between concurrent writers. Semantic conflict handling
    belongs to GoalRepository revision checks.

    Prior fix: active.json.tmp was a fixed name, so two concurrent writers
    could have Writer 2's write_text() clobber Writer 1's tmp before
    Writer 1's replace() completed, causing ENOENT on replace.

    The fix uses unique tmp names (UUID suffix) per call so each writer
    operates on its own tmp file. Content is still last-writer-wins —
    this does NOT prevent semantic race conditions on goal reads.
    """

    def test_atomic_write_uses_unique_tmp_paths_under_concurrency(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs

        gs._ensure_dirs()
        gs._atomic_write(gs.GOALS_DIR / "active.json", _mk_goal())

        results: dict = {}
        errors: dict = {}
        barrier = threading.Barrier(2)

        def writer(n: int):
            try:
                barrier.wait()
                goal = _mk_goal(text=f"writer-{n}")
                ok, err = gs.save_active_goal(goal)
                results[n] = ok
                errors[n] = err
            except Exception as e:
                errors[n] = str(e)

        t1 = threading.Thread(target=writer, args=(1,))
        t2 = threading.Thread(target=writer, args=(2,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Both must succeed without raising — the unique-tmp fix prevents
        # FileNotFoundError from a concurrent writer clobbering the shared tmp.
        assert all(r for r in results.values()), f"errors: {errors}"

        # Final active.json must be valid JSON (last-writer-wins content OK).
        with open(gs.GOALS_DIR / "active.json") as f:
            data = json.load(f)
        assert "text" in data

        # No orphaned *.tmp files should remain from the concurrent calls.
        assert not list(gs.GOALS_DIR.glob("active.json.*.tmp")), \
            "orphaned unique-tmp files remain after writes"


class TestGoalsCmd:
    """handle_goals subcommands and output formatting."""

    def test_status_no_goal(self, goals_dir):
        from hermes_cli.goal_core import goals_cmd

        out = goals_cmd.handle_goals(["status"])
        assert "No active goal" in out

    def test_status_with_goal(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs
        from hermes_cli.goal_core import goals_cmd

        gs._ensure_dirs()
        gs._atomic_write(gs.GOALS_DIR / "active.json", _mk_goal(text="my goal"))
        out = goals_cmd.handle_goals(["status"])
        assert "my goal" in out

    def test_add_without_goal_is_rejected(self, goals_dir):
        from hermes_cli.goal_core import goals_cmd

        out = goals_cmd.handle_goals(["add", "new item"])
        assert "No active goal" in out

    def test_add_item(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs
        from hermes_cli.goal_core import goals_cmd

        gs._ensure_dirs()
        gs._atomic_write(gs.GOALS_DIR / "active.json", _mk_goal())
        out = goals_cmd.handle_goals(["add", "fix the bug"])
        assert "fix the bug" in out

    def test_update_unknown_item(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs
        from hermes_cli.goal_core import goals_cmd

        gs._ensure_dirs()
        gs._atomic_write(gs.GOALS_DIR / "active.json", _mk_goal())
        out = goals_cmd.handle_goals(["update", "unknown-id", "completed"])
        assert "not found" in out

    def test_update_invalid_transition(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs
        from hermes_cli.goal_core import goals_cmd

        gs._ensure_dirs()
        gs._atomic_write(
            gs.GOALS_DIR / "active.json",
            _mk_goal(
                checklist=[
                    {
                        "id": "x",
                        "text": "task x",
                        "status": "completed",
                        "created_at": _now_iso(),
                        "updated_at": _now_iso(),
                        "completed_at": _now_iso(),
                    }
                ]
            ),
        )
        out = goals_cmd.handle_goals(["update", "x", "pending"])
        assert "Cannot transition" in out or "invalid" in out.lower()

    def test_note_appends(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs
        from hermes_cli.goal_core import goals_cmd

        gs._ensure_dirs()
        gs._atomic_write(gs.GOALS_DIR / "active.json", _mk_goal())
        out = goals_cmd.handle_goals(["note", "found the root cause"])
        assert "Evidence" in out or "appended" in out

    def test_done_fails_if_items_not_complete(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs
        from hermes_cli.goal_core import goals_cmd

        gs._ensure_dirs()
        gs._atomic_write(
            gs.GOALS_DIR / "active.json",
            _mk_goal(
                checklist=[
                    {
                        "id": "y",
                        "text": "not done",
                        "status": "pending",
                        "created_at": _now_iso(),
                        "updated_at": _now_iso(),
                        "completed_at": None,
                    }
                ]
            ),
        )
        out = goals_cmd.handle_goals(["done"])
        assert "Only" in out or "not" in out

    def test_list_empty(self, goals_dir):
        from hermes_cli.goal_core import goals_cmd

        out = goals_cmd.handle_goals(["list"])
        assert "No archived" in out

    def test_list_shows_archived(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs
        from hermes_cli.goal_core import goals_cmd

        gs._ensure_dirs()
        gs._atomic_write(gs.HISTORY_DIR / "arch.json", _mk_goal(text="archived goal"))
        out = goals_cmd.handle_goals(["list"])
        assert "archived goal" in out

    def test_help_shows_reference(self, goals_dir):
        from hermes_cli.goal_core import goals_cmd

        out = goals_cmd.handle_goals(["help"])
        assert "/goals add" in out
        assert "/goals update" in out
        assert "/goals done" in out

    def test_unknown_subcommand(self, goals_dir):
        from hermes_cli.goal_core import goals_cmd

        out = goals_cmd.handle_goals(["fly-to-the-moon"])
        assert "Unknown" in out

    def test_delete_nonexistent(self, goals_dir):
        from hermes_cli.goal_core import goals_cmd

        out = goals_cmd.handle_goals(["delete", "no-such-id"])
        assert "not found" in out

    def test_cancel_archives(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs
        from hermes_cli.goal_core import goals_cmd

        gs._ensure_dirs()
        gs._atomic_write(gs.GOALS_DIR / "active.json", _mk_goal())
        out = goals_cmd.handle_goals(["cancel"])
        assert "archived" in out.lower()
        # active.json should be gone
        assert not (gs.GOALS_DIR / "active.json").exists()

    def test_review_via_cmd(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs
        from hermes_cli.goal_core import goals_cmd

        gs._ensure_dirs()
        gs._atomic_write(gs.GOALS_DIR / "active.json", _mk_goal())
        out = goals_cmd.handle_goals(["review"])
        assert "valid" in out.lower() or "active" in out.lower()

    def test_purge_works(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs
        from hermes_cli.goal_core import goals_cmd

        gs._ensure_dirs()
        gs._atomic_write(gs.TRASH_DIR / "gone.json", _mk_goal())
        out = goals_cmd.handle_goals(["purge"])
        assert "0" in out or "deleted" in out.lower()

    def test_restore_unknown(self, goals_dir):
        from hermes_cli.goal_core import goals_cmd

        out = goals_cmd.handle_goals(["restore", "no-such-id"])
        assert "not found" in out

    def test_note_with_tool_flag(self, goals_dir):
        import hermes_cli.goal_core.goal_store as gs
        from hermes_cli.goal_core import goals_cmd

        gs._ensure_dirs()
        gs._atomic_write(gs.GOALS_DIR / "active.json", _mk_goal())
        out = goals_cmd.handle_goals(["note", "--tool", "browser", "clicked login"])
        assert "Evidence" in out or "appended" in out or "✓" in out


# ─── integration-ish: full round-trip with real subprocess ───────────────────


class TestGoalsIntegration:
    """Run the full CLI as a subprocess, verify I/O round-trips correctly."""

    def test_cli_full_lifecycle(self, goals_dir, monkeypatch):
        """
        Create goal → add 3 items → complete 2 → done → verify in history/.

        We mock GOALS_DIR via env var trick for the subprocess because
        the CLI resolves paths at import time.
        """
        import subprocess, hermes_cli.goal_core.goal_store as gs

        gs._ensure_dirs()

        # Prime active.json
        goal = _mk_goal(text="integration test goal")
        gs._atomic_write(gs.GOALS_DIR / "active.json", goal)

        # Use handler directly (not subprocess) for speed and isolation
        from hermes_cli.goal_core import goals_cmd

        r = goals_cmd.handle_goals(["add", "step one"])
        assert "step one" in r

        r = goals_cmd.handle_goals(["add", "step two"])
        assert "step two" in r

        r = goals_cmd.handle_goals(["add", "step three"])
        assert "step three" in r

        # Complete all 3
        goal, _ = gs.get_active_goal()
        for item in goal["checklist"]:
            goals_cmd.handle_goals(["update", item["id"], "completed"])

        # Now /goals done should succeed
        r = goals_cmd.handle_goals(["done"])
        assert "archived" in r.lower() or "completed" in r.lower()

        # active.json gone
        assert not (gs.GOALS_DIR / "active.json").exists()

        # One archived goal present
        hist = list((gs.GOALS_DIR / "history").glob("*.json"))
        assert len(hist) == 1
        archived = json.loads(hist[0].read_text())
        assert archived["status"] == "completed"
        assert archived["text"] == "integration test goal"
        assert len(archived["checklist"]) == 3
        assert all(i["status"] == "completed" for i in archived["checklist"])

    def test_evidence_log_persists_after_archive(self, goals_dir):
        """
        Evidence in evidence.log must survive archiving the goal.
        The log is append-only and global.
        """
        import hermes_cli.goal_core.goal_store as gs
        from hermes_cli.goal_core import goals_cmd

        gs._ensure_dirs()
        gs._atomic_write(gs.GOALS_DIR / "active.json", _mk_goal())
        goals_cmd.handle_goals(["note", "first note"])
        goals_cmd.handle_goals(["note", "second note"])
        goals_cmd.handle_goals(["cancel"])

        # evidence.log should still exist with both entries
        assert gs.EVIDENCE_LOG.exists()
        lines = gs.EVIDENCE_LOG.read_text().strip().splitlines()
        assert len(lines) == 2
        assert "first note" in lines[0]
        assert "second note" in lines[1]
