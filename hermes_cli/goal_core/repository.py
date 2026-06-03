"""
GoalRepository — canonical goal persistence layer backed by SessionDB.

Phase 3: goals.v2_repository feature flag wires this repository as the primary
storage path behind GoalManager.load_goal / save_goal / migrate_goal_session.
Goals table is canonical; state_meta is migrated on first read.

Design invariants (enforced in every write):
  goals.goal_id       = state.goal_id
  goals.revision      = state.revision
  goals.status        = row_status (derived from state + operation)
  goals.goal_text      = state.goal
  goals.state_json    = state.to_json()

Single helper: _state_to_goal_columns(state, now, row_status=None)
  row_status=None  → derive from state.status (active|paused|done|cleared)
  row_status set   → use that value (e.g. "archived" for archive operations)

Archive note: "archived" is a row-level status only.
  GoalState.status does not include "archived" (GoalStatus enum only has
  active|paused|done|cleared).  The repository stores "archived" in
  goals.status; when reconstructing GoalState, state.status is left as
  the pre-archive value (typically "cleared").  The row is authoritative
  for archived goals — state_json is preserved for audit but its status
  field stays within the enum.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from hermes_state import SessionDB

# GoalState lives in hermes_cli/goals.py (sibling to goal_core/)
sys.path.insert(0, str(Path(__file__).parent.parent))
from goals import GoalState, GoalStatus


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GoalRecord:
    """A fully-loaded goal row with its current GoalState."""

    goal_id: str
    state: GoalState
    revision: int
    status: str                      # row-level status (may be "archived")
    session_id: Optional[str]
    session_key: Optional[str]
    created_at: float
    updated_at: float

    def to_dict(self) -> dict:
        return {
            "goal_id": self.goal_id,
            "revision": self.revision,
            "status": self.status,
            "session_id": self.session_id,
            "session_key": self.session_key,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "state": self.state.to_json(),
        }


@dataclass
class GoalSummary:
    """Lightweight goal listing row."""

    goal_id: str
    goal_text: str
    status: str
    revision: int
    session_id: Optional[str]
    session_key: Optional[str]
    updated_at: float


@dataclass
class MigrationReport:
    """Counters from a migrate_all_state_meta_goals() run."""

    scanned: int = 0
    migrated: int = 0
    already_migrated: int = 0
    repaired_goal_ids: int = 0
    collisions_repaired: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class GoalRevisionConflict(Exception):
    """save_goal called with expected_revision that doesn't match current row."""

    def __init__(self, goal_id: str, expected: int, actual: int):
        self.goal_id = goal_id
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"GoalRevisionConflict: goal_id={goal_id} expected_revision={expected} actual={actual}"
        )


class GoalSessionConflict(Exception):
    """migrate_session called but new_session_id already has a canonical binding for a different goal."""

    def __init__(self, new_session_id: str, existing_goal_id: str, attempted_goal_id: str):
        self.new_session_id = new_session_id
        self.existing_goal_id = existing_goal_id
        self.attempted_goal_id = attempted_goal_id
        super().__init__(
            f"GoalSessionConflict: session_id={new_session_id} already bound to goal_id={existing_goal_id}, "
            f"cannot bind to goal_id={attempted_goal_id}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Feature flag (goals.v2_repository)
# ─────────────────────────────────────────────────────────────────────────────

# Test-override flag — set by test helpers, not by production code.
_v2_override: Optional[bool] = None


def _v2_enabled() -> bool:
    """Return whether v2 repository path is active.

    Check order:
      1. Test override (set_v2_enabled_for_test / clear_v2_enabled_for_test)
      2. Config file goals.v2_repository value
      3. Default False
    """
    if _v2_override is not None:
        return _v2_override
    try:
        from hermes_cli.config import read_raw_config

        cfg = read_raw_config()
        return bool(cfg.get("goals", {}).get("v2_repository", False))
    except Exception:
        return False


def set_v2_enabled_for_test(value: bool) -> None:
    """Override feature flag for the current test process."""
    global _v2_override
    _v2_override = value


def clear_v2_enabled_for_test() -> None:
    """Remove test override, restoring config-based lookup."""
    global _v2_override
    _v2_override = None


# Re-export test helpers so callers can import from one place.
set_v2_enabled_for_test = set_v2_enabled_for_test
clear_v2_enabled_for_test = clear_v2_enabled_for_test


# ─────────────────────────────────────────────────────────────────────────────
# Repository
# ─────────────────────────────────────────────────────────────────────────────

class GoalRepository:
    """Canonical goal persistence layer backed by SessionDB.

    Phase 2: implemented and tested in isolation from GoalManager.
    Goals.v2_repository=false — not yet wired as the primary system.
    """

    _redirect_stack: List[str] = []   # shared across _get_by_binding_chain calls
    MAX_REDIRECTS = 10

    def __init__(self, db: Optional[SessionDB] = None) -> None:
        self.db = db if db is not None else SessionDB()

    # ── GoalRecord construction ───────────────────────────────────────────────

    def _load_record(self, row: dict, session_id: Optional[str] = None) -> GoalRecord:
        """Reconstruct a GoalRecord from a goals table row.

        Phase 2.5: state_json is patched to include goal_id and revision
        before INSERT (via _state_to_goal_columns), and _load_state_json
        restores both to the GoalState object so the three agree.

        session_id: optional override for the session_id field, used when
        loading from a binding context where the session is known (e.g.
        migrate_session return value). If omitted, reads from the row.
        """
        state = GoalState.from_json(row["state_json"])
        # Restore goal_id and revision from column values — they are the
        # authoritative source of truth for the canonical goal.
        object.__setattr__(state, "goal_id", row["goal_id"])
        object.__setattr__(state, "revision", row["revision"])
        return GoalRecord(
            goal_id=row["goal_id"],
            state=state,
            revision=row["revision"],
            status=row["status"],
            session_id=session_id if session_id is not None else (row["session_id"] if "session_id" in row.keys() else None),
            session_key=row["session_key"] if "session_key" in row.keys() else None,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # ── Canonical binding helpers ────────────────────────────────────────────

    def _canonical_binding(self, session_id: str) -> Optional[dict]:
        """Return the canonical (non-alias, non-ended) binding for a session, or None."""
        return self.db._conn.execute(
            """
            SELECT gb.*
              FROM goal_sessions gb
             WHERE gb.session_id  = :session_id
               AND gb.ended_at    IS NULL
               AND gb.redirect_to IS NULL
            """,
            {"session_id": session_id},
        ).fetchone()

    def _get_by_binding_chain(self, session_id: str) -> Optional[GoalRecord]:
        """Follow the alias chain for session_id, return the terminal canonical record."""
        self._redirect_stack = []
        current = session_id

        while len(self._redirect_stack) < self.MAX_REDIRECTS:
            binding = self._canonical_binding(current)
            if not binding:
                return None

            if binding["redirect_to"]:
                self._redirect_stack.append(current)
                current = binding["redirect_to"]
                continue

            # Canonical endpoint found — load the goal
            row = self.db._conn.execute(
                "SELECT * FROM goals WHERE goal_id=?", (binding["goal_id"],)
            ).fetchone()
            if not row:
                return None
            return self._load_record(row, session_id=current)

        # Max redirects exceeded — return the last known binding as a best-effort
        row = self.db._conn.execute(
            "SELECT * FROM goals WHERE goal_id=?", (binding["goal_id"],)
        ).fetchone()
        if row:
            return self._load_record(row, session_id=current)
        return None

    # ── Column derivation ───────────────────────────────────────────────────

    @staticmethod
    def _status_from_state(state: GoalState) -> str:
        s = getattr(state, "status", GoalStatus.ACTIVE.value)
        if s == "done":
            return "done"
        if s == "paused":
            return "paused"
        if s == "cleared":
            return "cleared"
        return "active"

    @staticmethod
    def _state_to_goal_columns(
        state: GoalState,
        now: float,
        row_status: Optional[str] = None,
    ) -> dict:
        """Derive all goals-table columns from a GoalState.

        row_status=None  → derive from state.status
        row_status set   → use that value (e.g. "archived")
        """
        if row_status is None:
            row_status = GoalRepository._status_from_state(state)

        goal_id = getattr(state, "goal_id", None) or f"goal-{uuid.uuid4().hex[:12]}"
        revision = getattr(state, "revision", 0) or 0

        # Phase 2.5: build state_json from the GoalState, then inject
        # goal_id and revision explicitly so they are always present in the
        # stored JSON even if the GoalState object had them missing.
        data = json.loads(state.to_json())
        data["goal_id"] = goal_id
        data["revision"] = revision
        state_json = json.dumps(data, ensure_ascii=False)

        return {
            "goal_id": goal_id,
            "schema_version": 1,
            "revision": revision,
            "status": row_status,
            "goal_text": getattr(state, "goal", ""),
            "state_json": state_json,
            "source": getattr(state, "source", None),
            "created_at": getattr(state, "created_at", now),
            "updated_at": now,
            "completed_at": now if row_status == "done" else None,
        }

    # ── Save ───────────────────────────────────────────────────────────────

    def save_goal(
        self,
        goal_id: str,
        state: GoalState,
        expected_revision: Optional[int] = None,
    ) -> GoalRecord:
        """Persist or update a goal.

        expected_revision (optional): if provided, the UPDATE is conditional on
        the row having this revision; a mismatch raises GoalRevisionConflict
        without modifying the row.  This prevents TOCTOU races in callers that
        read-then-write without holding a lock.
        """
        now = time.time()
        cols = self._state_to_goal_columns(state, now)

        existing = self.db._conn.execute(
            "SELECT goal_id, revision FROM goals WHERE goal_id=?",
            (goal_id,),
        ).fetchone()

        if existing is None:
            # ── Insert ──────────────────────────────────────────────────────
            cols["goal_id"] = goal_id
            self.db._conn.execute(
                """
                INSERT INTO goals
                  (goal_id, schema_version, revision, status, goal_text,
                   state_json, source, created_at, updated_at,
                   completed_at, archived_at, archived_reason, session_key, created_by)
                VALUES
                  (:goal_id, :schema_version, :revision, :status, :goal_text,
                   :state_json, :source, :created_at, :updated_at,
                   :completed_at, NULL, NULL, NULL, 'human')
                """,
                cols,
            )
            self.emit_event(goal_id, "goal_created", session_id=getattr(state, "session_id", None))
            row = self.db._conn.execute("SELECT * FROM goals WHERE goal_id=?", (goal_id,)).fetchone()
            return self._load_record(row)

        # ── Update ───────────────────────────────────────────────────────────
        if expected_revision is not None:
            # Atomic conditional UPDATE — no pre-check TOCTOU.
            # Use state.revision as the new value; caller is responsible for
            # incrementing it between calls (or passing the right expected_revision).
            updated = self.db._conn.execute(
                """
                UPDATE goals
                   SET revision     = :revision,
                       status       = :status,
                       goal_text    = :goal_text,
                       state_json   = :state_json,
                       source       = :source,
                       updated_at   = :updated_at,
                       completed_at = :completed_at
                 WHERE goal_id      = :goal_id
                   AND revision     = :expected_revision
                """,
                {**cols, "goal_id": goal_id, "expected_revision": expected_revision},
            ).rowcount

            if updated == 0:
                # Re-query to get the actual current revision
                current = self.db._conn.execute(
                    "SELECT revision FROM goals WHERE goal_id=?", (goal_id,)
                ).fetchone()
                actual = current["revision"] if current else "N/A"
                self.emit_event(
                    goal_id,
                    "revision_conflict",
                    session_id=getattr(state, "session_id", None),
                    event_data={
                        "expected": expected_revision,
                        "actual": actual,
                    },
                )
                raise GoalRevisionConflict(goal_id, expected_revision, actual)

        else:
            # No expected_revision: read current revision from DB and increment
            # in-memory so every UPDATE bumps the counter.
            current_row = self.db._conn.execute(
                "SELECT revision FROM goals WHERE goal_id=?", (goal_id,)
            ).fetchone()
            next_revision = (current_row["revision"] + 1) if current_row else 1
            object.__setattr__(state, "revision", next_revision)
            cols = self._state_to_goal_columns(state, now)

            self.db._conn.execute(
                """
                UPDATE goals
                   SET revision     = :revision,
                       status       = :status,
                       goal_text    = :goal_text,
                       state_json   = :state_json,
                       source       = :source,
                       updated_at   = :updated_at,
                       completed_at = :completed_at
                 WHERE goal_id      = :goal_id
                """,
                {**cols, "goal_id": goal_id},
            )

        self.emit_event(goal_id, "goal_updated", session_id=getattr(state, "session_id", None))
        row = self.db._conn.execute("SELECT * FROM goals WHERE goal_id=?", (goal_id,)).fetchone()
        return self._load_record(row)

    # ── Set active goal ────────────────────────────────────────────────────

    def set_active_goal(
        self,
        session_id: str,
        state: GoalState,
        *,
        session_key: Optional[str] = None,
        archive_reason: str = "replaced by set_active_goal",
    ) -> GoalRecord:
        """Set or replace the active goal for a session.

        - Archives any existing goal for this session and ends its binding.
        - Saves the new goal (revision=0 initial).
        - Creates a canonical goal_sessions binding.
        - Emits goal_created and session_bound events.

        archive_reason: passed to archive_goal when replacing an existing goal.
        """
        now = time.time()

        # End any existing canonical binding for this session
        existing = self._canonical_binding(session_id)
        if existing:
            self.archive_goal(existing["goal_id"], reason=archive_reason)

        # Assign goal_id if missing
        if not getattr(state, "goal_id", None):
            object.__setattr__(state, "goal_id", f"goal-{uuid.uuid4().hex[:12]}")
        object.__setattr__(state, "revision", 0)

        cols = self._state_to_goal_columns(state, now, row_status="active")

        self.db._conn.execute(
            """
            INSERT INTO goals
              (goal_id, schema_version, revision, status, goal_text,
               state_json, source, created_at, updated_at,
               completed_at, archived_at, archived_reason, session_key, created_by)
            VALUES
              (:goal_id, :schema_version, :revision, :status, :goal_text,
               :state_json, :source, :created_at, :updated_at,
               :completed_at, NULL, NULL, :session_key, 'human')
            """,
            {**cols, "session_key": session_key},
        )

        # Create canonical binding
        binding_id = f"bind-{uuid.uuid4().hex[:12]}"
        self.db._conn.execute(
            """
            INSERT INTO goal_sessions
              (binding_id, goal_id, session_id, session_key, started_at,
               ended_at, end_reason, redirect_to, updated_at)
            VALUES
              (:binding_id, :goal_id, :session_id, :session_key, :started_at,
               NULL, NULL, NULL, :updated_at)
            """,
            {
                "binding_id": binding_id,
                "goal_id": cols["goal_id"],
                "session_id": session_id,
                "session_key": session_key,
                "started_at": now,
                "updated_at": now,
            },
        )

        self.emit_event(cols["goal_id"], "goal_created", session_id=session_id)
        self.emit_event(cols["goal_id"], "session_bound", session_id=session_id)

        # Emit goal_replaced on the new goal if this replaced an existing goal
        if existing:
            self.emit_event(
                cols["goal_id"],
                "goal_replaced",
                session_id=session_id,
                event_data={
                    "old_goal_id": existing["goal_id"],
                    "reason": archive_reason,
                },
            )

        row = self.db._conn.execute(
            "SELECT * FROM goals WHERE goal_id=?",
            (cols["goal_id"],),
        ).fetchone()
        return self._load_record(row, session_id=session_id)

    # ── Active goal (canonical binding only — no legacy fallback here) ────────

    def get_active_goal(self, session_id: str) -> Optional[GoalRecord]:
        """Return the active GoalRecord for a session.

        Lookup order:
          1. Canonical binding in goal_sessions
          2. Alias redirect resolution (up to MAX_REDIRECTS)
          3. Legacy state_meta goal:{session_id} — migrated on read
        """
        rec = self._get_by_binding_chain(session_id)
        if rec:
            return rec
        return self._migrate_legacy_on_read(session_id)

    def _migrate_legacy_on_read(self, session_id: str) -> Optional[GoalRecord]:
        """Migrate a single legacy state_meta goal for session_id and return the record.

        Returns None if no legacy goal exists for session_id.
        Does NOT modify the original state_meta value.
        """
        legacy_key = f"goal:{session_id}"
        raw_row = self.db._conn.execute(
            "SELECT value FROM state_meta WHERE key=?",
            (legacy_key,),
        ).fetchone()
        if not raw_row:
            return None

        raw_value = raw_row[0]

        # Check idempotency first
        already = self.db._conn.execute(
            "SELECT goal_id FROM goal_legacy_migrations WHERE legacy_key=?",
            (legacy_key,),
        ).fetchone()
        if already:
            return self._get_by_binding_chain(session_id)

        # Do the migration for this single row
        report = MigrationReport()
        try:
            self._migrate_one_state_meta_goal(session_id, legacy_key, raw_value, report)
        except Exception:
            return None

        if report.migrated == 0:
            return None

        return self._get_by_binding_chain(session_id)

    # ── Archive ─────────────────────────────────────────────────────────────

    def archive_goal(self, goal_id: str, *, reason: Optional[str] = None) -> None:
        """Archive a goal.

        Sets row-level status fields:
          - goals.status = 'archived'              (SQL row marker — NOT GoalState.status)
          - goals.archived_at   = now
          - goals.archived_reason = reason
          - goals.state_json = json_set(state_json, '$.status', 'cleared')

        Invariants for archived rows:
          - goals.status = 'archived'
          - state_json.status = 'cleared'
          - goals.archived_at IS NOT NULL
          - All canonical goal_sessions rows have ended_at set

        The split is intentional: GoalState.status cannot hold 'archived' (not
        in the GoalStatus enum), so the row carries it instead.
        """
        now = time.time()

        self.db._conn.execute(
            """
            UPDATE goals
               SET status          = 'archived',
                   archived_at     = :archived_at,
                   archived_reason = :reason,
                   state_json      = json_set(state_json, '$.status', 'cleared'),
                   updated_at      = :updated_at
             WHERE goal_id = :goal_id
               AND status  != 'archived'
            """,
            {"goal_id": goal_id, "archived_at": now, "reason": reason, "updated_at": now},
        )

        # End all canonical active bindings for this goal
        self.db._conn.execute(
            """
            UPDATE goal_sessions
               SET ended_at    = :ended_at,
                   end_reason  = :end_reason,
                   updated_at  = :updated_at
             WHERE goal_id     = :goal_id
               AND ended_at   IS NULL
               AND redirect_to IS NULL
            """,
            {
                "goal_id": goal_id,
                "ended_at": now,
                "end_reason": reason if reason else "archived",
                "updated_at": now,
            },
        )

        self.emit_event(
            goal_id,
            "goal_archived",
            session_id=None,
            event_data={"reason": reason} if reason else None,
        )

        self.emit_event(goal_id, "session_unbound", session_id=None)

    # ── Migrate session ─────────────────────────────────────────────────────

    def migrate_session(
        self,
        old_session_id: str,
        new_session_id: str,
        *,
        reason: str = "session rollover",
        alias_window: bool = False,
    ) -> Optional[GoalRecord]:
        """Migrate a goal from one session to another.

        Creates an alias binding for the old session that redirects to the new
        canonical binding, so in-flight requests on the old session are not
        broken during the rollover window.

        alias_window=True: keeps the old canonical binding alive (ends it and
        creates a redirect_to alias pointing to the new session). The old session
        can still resolve to the goal via the alias chain.

        Safety: if new_session_id already has a canonical binding (for a different
        goal), raises GoalSessionConflict without modifying anything.

        Returns the new canonical binding's GoalRecord, or None if there was
        nothing to migrate.
        """
        now = time.time()

        # Pre-check: resolve goal_id for old_session, and check new_session availability
        old_canonical = self._canonical_binding(old_session_id)
        if old_canonical is None:
            return None   # Nothing to migrate — idempotent

        old_goal_id = old_canonical["goal_id"]

        new_existing = self._canonical_binding(new_session_id)
        if new_existing is not None and new_existing["goal_id"] != old_goal_id:
            # Emit outside the tx so the event persists even on conflict/rollback
            self.emit_event(
                old_goal_id,
                "goal_session_conflict",
                session_id=new_session_id,
                event_data={
                    "new_session_id": new_session_id,
                    "existing_goal_id": new_existing["goal_id"],
                    "attempted_goal_id": old_goal_id,
                    "reason": reason,
                },
            )
            raise GoalSessionConflict(new_session_id, new_existing["goal_id"], old_goal_id)

        # Write transaction
        binding_id = f"bind-{uuid.uuid4().hex[:12]}"
        alias_id = f"alias-{uuid.uuid4().hex[:12]}"

        try:
            if alias_window:
                # Keep old canonical alive as an alias (redirect_to -> new_session)
                self.db._conn.execute(
                    """
                    UPDATE goal_sessions
                       SET redirect_to = :redirect_to,
                           updated_at  = :updated_at
                     WHERE binding_id  = :binding_id
                       AND ended_at   IS NULL
                    """,
                    {
                        "binding_id": old_canonical["binding_id"],
                        "redirect_to": new_session_id,
                        "updated_at": now,
                    },
                )
            else:
                # End old canonical binding
                self.db._conn.execute(
                    """
                    UPDATE goal_sessions
                       SET ended_at    = :ended_at,
                           end_reason  = :reason,
                           updated_at  = :updated_at
                     WHERE binding_id  = :binding_id
                       AND ended_at   IS NULL
                    """,
                    {
                        "binding_id": old_canonical["binding_id"],
                        "ended_at": now,
                        "reason": reason,
                        "updated_at": now,
                    },
                )

            # Insert new canonical binding
            self.db._conn.execute(
                """
                INSERT INTO goal_sessions
                  (binding_id, goal_id, session_id, session_key, started_at,
                   ended_at, end_reason, redirect_to, updated_at)
                VALUES
                  (:binding_id, :goal_id, :session_id, NULL, :started_at,
                   NULL, NULL, NULL, :updated_at)
                """,
                {
                    "binding_id": binding_id,
                    "goal_id": old_goal_id,
                    "session_id": new_session_id,
                    "started_at": now,
                    "updated_at": now,
                },
            )

            self.db._conn.execute(
                "UPDATE goals SET session_key=NULL, updated_at=? WHERE goal_id=?",
                (now, old_goal_id),
            )

            self.db._conn.commit()

        except Exception:
            self.db._conn.rollback()
            raise

        self.emit_event(
            old_goal_id,
            "session_rollover",
            session_id=new_session_id,
            event_data={
                "from_session_id": old_session_id,
                "to_session_id": new_session_id,
                "reason": reason,
            },
        )
        self.emit_event(old_goal_id, "session_bound", session_id=new_session_id)

        # Return the new canonical record
        return self.get_active_goal(new_session_id)

    # ── Goal listing ────────────────────────────────────────────────────────

    def list_goals(
        self,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[GoalSummary]:
        """Return lightweight goal summaries, newest first."""
        sql = """
            SELECT g.goal_id, g.goal_text, g.status, g.revision,
                   gb.session_id, gb.session_key, g.updated_at
              FROM goals g
         LEFT JOIN goal_sessions gb
                ON gb.goal_id = g.goal_id
               AND gb.ended_at IS NULL
               AND gb.redirect_to IS NULL
        """
        params: dict = {}

        if status == "archived":
            # Archived goals are excluded from the base query but explicitly
            # included when filtering for them — use a separate path.
            return self._list_goals_archived(limit)
        elif status:
            sql += " WHERE g.status = :status"
            params["status"] = status
        else:
            sql += " WHERE g.status != 'archived'"

        sql += " ORDER BY g.updated_at DESC LIMIT :limit"
        params["limit"] = limit

        rows = self.db._conn.execute(sql, params).fetchall()
        return [
            GoalSummary(
                goal_id=r["goal_id"],
                goal_text=r["goal_text"],
                status=r["status"],
                revision=r["revision"],
                session_id=r["session_id"],
                session_key=r["session_key"],
                updated_at=r["updated_at"],
            )
            for r in rows
        ]

    def _list_goals_archived(self, limit: int) -> List[GoalSummary]:
        """Return archived goal summaries without requiring a session binding."""
        rows = self.db._conn.execute(
            """
            SELECT goal_id, goal_text, status, revision,
                   NULL AS session_id, NULL AS session_key, updated_at
              FROM goals
             WHERE status = 'archived'
             ORDER BY updated_at DESC LIMIT :limit
            """,
            {"limit": limit},
        ).fetchall()
        return [
            GoalSummary(
                goal_id=r["goal_id"],
                goal_text=r["goal_text"],
                status=r["status"],
                revision=r["revision"],
                session_id=r["session_id"],
                session_key=r["session_key"],
                updated_at=r["updated_at"],
            )
            for r in rows
        ]

    def get_goal(self, goal_id: str) -> Optional[GoalRecord]:
        """Load a goal by goal_id, or None if not found."""
        row = self.db._conn.execute(
            "SELECT * FROM goals WHERE goal_id=?",
            (goal_id,),
        ).fetchone()
        if not row:
            return None
        return self._load_record(row)

    # ── Legacy state_meta migration ─────────────────────────────────────────

    def migrate_all_state_meta_goals(self) -> MigrationReport:
        """Scan and migrate every legacy state_meta goal:* row into the goals table.

        Does NOT modify any state_meta value.

        Returns a MigrationReport with counts for each outcome.
        """
        report = MigrationReport()

        rows = self.db._conn.execute(
            "SELECT key, value FROM state_meta WHERE key LIKE 'goal:%'",
        ).fetchall()

        report.scanned = len(rows)

        for (legacy_key, raw_value) in rows:
            session_id = legacy_key.split(":", 1)[1]
            try:
                self._migrate_one_state_meta_goal(session_id, legacy_key, raw_value, report)
            except Exception as exc:
                report.failed += 1
                report.errors.append(f"{legacy_key}: {exc}")

        return report

    def _migrate_one_state_meta_goal(
        self,
        session_id: str,
        legacy_key: str,
        raw_value: str,
        report: MigrationReport,
    ) -> None:
        """Migrate a single legacy state_meta goal:{session_id} row."""
        # 1. Idempotency: skip if already migrated
        already = self.db._conn.execute(
            "SELECT goal_id FROM goal_legacy_migrations WHERE legacy_key=?",
            (legacy_key,),
        ).fetchone()
        if already:
            report.already_migrated += 1
            return

        # 2. Parse the legacy GoalState JSON
        try:
            state = GoalState.from_json(raw_value)
        except Exception as exc:
            raise ValueError(f"Could not parse GoalState JSON: {exc}") from exc

        # 3. Resolve / repair goal_id
        goal_id = getattr(state, "goal_id", None)
        if not goal_id or not goal_id.startswith("goal-"):
            goal_id = f"goal-{uuid.uuid4().hex[:12]}"
            object.__setattr__(state, "goal_id", goal_id)
            report.repaired_goal_ids += 1

        # 4. Check for conflicting goal_id in goals table
        collision = False
        existing = self.db._conn.execute(
            "SELECT goal_id, state_json FROM goals WHERE goal_id=?",
            (goal_id,),
        ).fetchone()
        if existing:
            try:
                existing_state = GoalState.from_json(existing["state_json"])
                if getattr(existing_state, "goal", None) != getattr(state, "goal", None):
                    collision = True
            except Exception:
                collision = True

        if collision:
            goal_id = f"goal-{uuid.uuid4().hex[:12]}"
            object.__setattr__(state, "goal_id", goal_id)
            report.collisions_repaired += 1

        # 5. Inject goal_id and revision=0 into state_json
        data = json.loads(state.to_json())
        data["goal_id"] = goal_id
        data["revision"] = 0
        state_json = json.dumps(data, ensure_ascii=False)

        now = time.time()

        # 6. Insert goal row
        self.db._conn.execute(
            """
            INSERT OR IGNORE INTO goals
              (goal_id, schema_version, revision, status, goal_text,
               state_json, source, created_at, updated_at,
               completed_at, archived_at, archived_reason, session_key, created_by)
            VALUES
              (:goal_id, 1, 0, :status, :goal_text,
               :state_json, :source, :created_at, :updated_at,
               NULL, NULL, NULL, NULL, 'human')
            """,
            {
                "goal_id": goal_id,
                "status": getattr(state, "status", GoalStatus.ACTIVE.value),
                "goal_text": getattr(state, "goal", ""),
                "state_json": state_json,
                "source": "legacy",
                "created_at": getattr(state, "created_at", now),
                "updated_at": now,
            },
        )

        # 7. Create canonical goal_sessions binding
        binding_id = f"bind-{uuid.uuid4().hex[:12]}"
        self.db._conn.execute(
            """
            INSERT OR IGNORE INTO goal_sessions
              (binding_id, goal_id, session_id, session_key, started_at,
               ended_at, end_reason, redirect_to, updated_at)
            VALUES
              (:binding_id, :goal_id, :session_id, NULL, :started_at,
               NULL, NULL, NULL, :updated_at)
            """,
            {
                "binding_id": binding_id,
                "goal_id": goal_id,
                "session_id": session_id,
                "started_at": now,
                "updated_at": now,
            },
        )

        # 8. Record sidecar row
        self.db._conn.execute(
            "INSERT INTO goal_legacy_migrations (legacy_key, goal_id, migrated_at, source) "
            "VALUES (?, ?, ?, ?)",
            (legacy_key, goal_id, now, "state_meta"),
        )

        # 9. Emit event
        self.emit_event(
            goal_id,
            "state_meta_migrated",
            session_id=session_id,
            event_data={
                "legacy_key": legacy_key,
                "goal_id": goal_id,
                "session_id": session_id,
                "goal_text": getattr(state, "goal", "")[:80],
            },
        )

        report.migrated += 1

    # ── Event emission ─────────────────────────────────────────────────────

    def emit_event(
        self,
        goal_id: str,
        event_type: str,
        *,
        session_id: Optional[str] = None,
        session_key: Optional[str] = None,
        event_data: Optional[dict] = None,
    ) -> None:
        """Write a goal_events row."""
        self.db._conn.execute(
            """
            INSERT INTO goal_events
              (goal_id, session_id, session_key, event_type, event_data, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                goal_id,
                session_id,
                session_key,
                event_type,
                json.dumps(event_data or {}, ensure_ascii=False),
                time.time(),
            ),
        )
