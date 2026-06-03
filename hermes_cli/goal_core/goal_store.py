"""
Persistent cross-session goal tracking store.

File-based, decoupled from Hermes state.db. Provides atomic reads/writes
with flock locking, schema validation, and graceful degradation.

File structure:
  ~/.hermes/goals/
    active.json          # current active goal
    .lock                # flock file
    history/             # archived completed/cancelled goals
    trash/               # soft-deleted goals
    evidence.log         # append-only JSONL, global across all goals
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Schema version for forward-compatibility reading
SCHEMA_VERSION = "1.0"
GOALS_DIR = Path.home() / ".hermes" / "goals"
HISTORY_DIR = GOALS_DIR / "history"
TRASH_DIR = GOALS_DIR / "trash"
EVIDENCE_LOG = GOALS_DIR / "evidence.log"
LOCK_PATH = GOALS_DIR / ".lock"
STALE_LOCK_PATH = GOALS_DIR / ".lock_stale_at"
LOCK_TIMEOUT_SECS = 30  # after this, lock is considered stale

VALID_STATUSES = frozenset({"pending", "in_progress", "completed", "cancelled"})
VALID_GOAL_STATUSES = frozenset({"active", "completed", "cancelled", "needs_review"})
VALID_TRANSITIONS = {
    "pending": {"in_progress", "completed", "cancelled"},
    "in_progress": {"completed", "cancelled"},
    "completed": set(),
    "cancelled": set(),
}

# ─────────────────────────────────────────────────────────────────────────────
# Initialization
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_dirs() -> None:
    for d in (GOALS_DIR, HISTORY_DIR, TRASH_DIR):
        d.mkdir(parents=True, exist_ok=True)
    if not EVIDENCE_LOG.exists():
        EVIDENCE_LOG.touch()

    # Best-effort cleanup of orphaned unique-tmp files from crashed writers.
    # Pattern: active.json.<uid>.tmp — unique per call so no fixed-name
    # collision risk, but crash between write_text and replace can leave them.
    # Limit to GOALS_DIR only, never follow symlinks, ignore errors.
    try:
        now = time.time()
        for tmp in GOALS_DIR.glob("active.json.*.tmp"):
            try:
                if now - tmp.stat().st_mtime > 24 * 3600:
                    tmp.unlink(missing_ok=True)
            except Exception:
                pass
    except Exception:
        pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# Lock management
# ─────────────────────────────────────────────────────────────────────────────

class LockError(Exception):
    """Could not acquire lock within timeout."""
    pass


def _acquire_lock(nonblocking: bool = False) -> tuple:
    """Acquire flock on GOALS_DIR/.lock. Returns (fd, locked)."""
    _ensure_dirs()
    fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_RDWR)
    try:
        if nonblocking:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            # Retry 3x with 2-second sleeps
            for attempt in range(3):
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if attempt == 2:
                        raise LockError("Goal is being modified by another process. Retry shortly.")
                    time.sleep(2)
    except Exception:
        os.close(fd)
        raise
    return fd, True


def _release_lock(fd: int) -> None:
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def _is_lock_stale() -> bool:
    """Check if lock file is older than LOCK_TIMEOUT_SECS and holder is dead."""
    if not LOCK_PATH.exists():
        return False
    try:
        mtime = LOCK_PATH.stat().st_mtime
        age = time.time() - mtime
        if age < LOCK_TIMEOUT_SECS:
            return False
        # Check if PID is still alive
        try:
            with open(LOCK_PATH) as f:
                pid_str = f.read().strip()
                if pid_str.isdigit():
                    pid = int(pid_str)
                    os.kill(pid, 0)  # signal 0 = check existence
                    return False  # still alive
        except (ValueError, ProcessLookupError, PermissionError):
            pass
        return True
    except Exception:
        return False


def _remove_stale_lock() -> None:
    """Forcibly remove a stale lock file."""
    try:
        os.remove(LOCK_PATH)
    except FileNotFoundError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Schema validation
# ─────────────────────────────────────────────────────────────────────────────

def _validate_goal(data: dict) -> list[str]:
    """Validate a goal dict. Returns list of error strings (empty = valid)."""
    errors = []
    if not isinstance(data, dict):
        return ["root must be a dict"]

    if "id" not in data:
        errors.append("missing required field: id")
    if "text" not in data:
        errors.append("missing required field: text")
    if "status" not in data:
        errors.append("missing required field: status")
    elif data["status"] not in VALID_GOAL_STATUSES:
        errors.append(f"invalid status: {data['status']} (allowed: {VALID_GOAL_STATUSES})")

    checklist = data.get("checklist", [])
    if not isinstance(checklist, list):
        errors.append("checklist must be a list")
    else:
        for i, item in enumerate(checklist):
            if not isinstance(item, dict):
                errors.append(f"checklist[{i}] must be a dict")
                continue
            if "id" not in item:
                errors.append(f"checklist[{i}] missing id")
            if "text" not in item:
                errors.append(f"checklist[{i}] missing text")
            if "status" not in item:
                errors.append(f"checklist[{i}] missing status")
            elif item["status"] not in VALID_STATUSES:
                errors.append(f"checklist[{i}] invalid status: {item['status']}")

    return errors


# ─────────────────────────────────────────────────────────────────────────────
# Core read/write (atomic via rename)
# ─────────────────────────────────────────────────────────────────────────────

def _atomic_write(path: Path, data: dict) -> None:
    """Write to temp file then rename (atomic on POSIX).
    Uses a unique tmp name per call (UUID suffix) to avoid concurrent
    writers on the same tmp path clobbering each other.
    """
    uid = uuid.uuid4().hex[:8]
    tmp = path.with_name(f"{path.name}.{uid}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


def _read_goal(path: Path) -> tuple[Optional[dict], Optional[str]]:
    """Read and validate a goal file. Returns (goal, None) or (None, error)."""
    if not path.exists():
        return None, None
    try:
        data = json.loads(path.read_text())
        errors = _validate_goal(data)
        if errors:
            return None, f"Schema errors: {'; '.join(errors)}"
        return data, None
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_active_goal() -> tuple[Optional[dict], Optional[str]]:
    """Read active.json. Returns (goal, None) or (None, error). 'needs_review' is returned as normal."""
    _ensure_dirs()
    path = GOALS_DIR / "active.json"
    goal, err = _read_goal(path)
    if err:
        return goal, err
    # Auto-reset stale in_progress items
    if goal:
        now = time.time()
        changed = False
        for item in goal.get("checklist", []):
            if item.get("status") == "in_progress":
                updated_at = item.get("updated_at", "")
                try:
                    ts = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                    age = now - ts.timestamp()
                    if age > 30 * 60:
                        item["status"] = "pending"
                        changed = True
                except Exception:
                    pass
        if changed:
            _atomic_write(path, goal)
    return goal, None


def save_active_goal(goal: dict) -> tuple[bool, Optional[str]]:
    """Atomically write active.json. Returns (success, error)."""
    _ensure_dirs()
    if not os.access(str(GOALS_DIR), os.W_OK):
        return False, "goals/ directory is not writable"
    path = GOALS_DIR / "active.json"
    try:
        _atomic_write(path, goal)
        return True, None
    except Exception as e:
        return False, str(e)


def create_goal(text: str, created_by: str = "human") -> tuple[Optional[dict], Optional[str]]:
    """Create a new active goal (archives any existing one first)."""
    _ensure_dirs()

    # Archive existing active goal if present
    existing, _ = get_active_goal()
    if existing:
        arch_id = existing.get("id", str(uuid.uuid4()))
        arch_path = HISTORY_DIR / f"{arch_id}.json"
        try:
            _atomic_write(arch_path, existing)
        except Exception:
            pass  # non-fatal

    goal = {
        "schema_version": SCHEMA_VERSION,
        "id": str(uuid.uuid4()),
        "text": text,
        "checklist": [],
        "evidence": [],
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "completed_at": None,
        "status": "active",
        "created_by": created_by,
    }
    ok, err = save_active_goal(goal)
    return goal if ok else None, err


def add_checklist_item(text: str) -> tuple[Optional[dict], Optional[str]]:
    """Append a checklist item to the active goal."""
    goal, err = get_active_goal()
    if err:
        return None, err
    if not goal:
        return None, "No active goal. Use /goal <text> to create one."

    if goal.get("status") != "active":
        return None, f"Goal is {goal['status']}, not active. Cannot modify."

    item = {
        "id": str(uuid.uuid4()),
        "text": text,
        "status": "pending",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "completed_at": None,
    }
    goal["checklist"].append(item)
    goal["updated_at"] = _now_iso()
    ok, err = save_active_goal(goal)
    return goal if ok else None, err


def update_checklist_item(item_id: str, new_status: str) -> tuple[Optional[dict], Optional[str]]:
    """Update a checklist item's status."""
    if new_status not in VALID_STATUSES:
        return None, f"Invalid status: {new_status} (allowed: {VALID_STATUSES})"

    goal, err = get_active_goal()
    if err:
        return None, err
    if not goal:
        return None, "No active goal."

    if goal.get("status") != "active":
        return None, f"Goal is {goal['status']}, not active."

    item = None
    for it in goal["checklist"]:
        if it["id"] == item_id:
            item = it
            break
    if not item:
        return None, f"Item {item_id} not found in checklist."

    allowed = VALID_TRANSITIONS.get(item["status"], set())
    if new_status not in allowed and not (
        item["status"] in ("pending", "in_progress") and new_status == "cancelled"
    ):
        return None, f"Cannot transition from '{item['status']}' to '{new_status}'."

    item["status"] = new_status
    item["updated_at"] = _now_iso()
    if new_status in ("completed", "cancelled"):
        item["completed_at"] = _now_iso()
    goal["updated_at"] = _now_iso()

    ok, err = save_active_goal(goal)
    return goal if ok else None, err


def append_evidence(content: str, tool: Optional[str] = None, meta: Optional[dict] = None) -> tuple[bool, Optional[str]]:
    """Append an evidence entry to active goal AND to global evidence.log."""
    goal, err = get_active_goal()
    if err:
        return False, err
    if not goal:
        return False, "No active goal."

    entry = {
        "goal_id": goal["id"],
        "timestamp": _now_iso(),
        "content": content,
        "tool": tool,
        "meta": meta or {},
    }

    # Append to goal's evidence array
    goal["evidence"].append({
        "timestamp": entry["timestamp"],
        "content": entry["content"],
        "tool": entry["tool"],
        "meta": entry["meta"],
    })
    goal["updated_at"] = _now_iso()
    ok, err = save_active_goal(goal)
    if not ok:
        return False, err

    # Append to global evidence.log
    try:
        with open(EVIDENCE_LOG, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        return False, f"Goal updated but evidence.log write failed: {e}"

    return True, None


def archive_goal(status: str) -> tuple[bool, Optional[str]]:
    """Archive active goal to history/ with given status (completed/cancelled)."""
    if status not in {"completed", "cancelled"}:
        return False, f"Invalid status: {status}"

    goal, err = get_active_goal()
    if err:
        return False, err
    if not goal:
        return False, "No active goal."

    goal["status"] = status
    goal["updated_at"] = _now_iso()
    if status == "completed":
        goal["completed_at"] = _now_iso()

    arch_path = HISTORY_DIR / f"{goal['id']}.json"
    try:
        _atomic_write(arch_path, goal)
        os.remove(GOALS_DIR / "active.json")
        return True, None
    except Exception as e:
        return False, f"Archive failed: {e}"


def review_goal() -> tuple[bool, Optional[str], Optional[str]]:
    """Validate active.json. If valid, set status=active. Returns (fixed, error, message)."""
    _ensure_dirs()
    path = GOALS_DIR / "active.json"
    if not path.exists():
        return False, "No active.json to review.", None

    goal, err = _read_goal(path)
    if err:
        # Try to save a broken copy
        broken_path = TRASH_DIR / f"broken-{uuid.uuid4().hex[:8]}.json"
        try:
            shutil.copy2(str(path), str(broken_path))
            msg = f"Schema errors in active.json. Broken copy saved to {broken_path.name}."
        except Exception:
            msg = f"Schema errors in active.json: {err}"
        return False, err, msg

    goal["status"] = "active"
    goal["updated_at"] = _now_iso()
    ok, err = save_active_goal(goal)
    if ok:
        return True, None, "Goal is valid and set to active."
    return False, err, None


def list_goals(include_trash: bool = False) -> list[dict]:
    """List all goals in history/ (and trash/ if include_trash=True)."""
    _ensure_dirs()
    results = []
    for directory, prefix in [(HISTORY_DIR, ""), (TRASH_DIR, "trash:")] if include_trash else [(HISTORY_DIR, "")]:
        if not directory.exists():
            continue
        for f in sorted(directory.glob("*.json"), reverse=True):
            try:
                data = json.loads(f.read_text())
                results.append({
                    "id": data.get("id", f.stem),
                    "text": data.get("text", "")[:80],
                    "status": data.get("status", "unknown"),
                    "created_at": data.get("created_at", ""),
                    "completed_at": data.get("completed_at"),
                    "checklist_summary": f"{sum(1 for i in data.get('checklist', []) if i.get('status') == 'completed')}/{len(data.get('checklist', []))}",
                    "source": f"{prefix}{f.name}" if prefix else f.name,
                })
            except Exception:
                continue
    return results


def soft_delete_goal(goal_id: str) -> tuple[bool, Optional[str]]:
    """Move goal from history/ to trash/."""
    _ensure_dirs()
    src = HISTORY_DIR / f"{goal_id}.json"
    if not src.exists():
        # Try as active goal
        active, _ = get_active_goal()
        if active and active.get("id") == goal_id:
            src = GOALS_DIR / "active.json"
        else:
            return False, f"Goal {goal_id} not found."

    dst = TRASH_DIR / src.name
    try:
        shutil.move(str(src), str(dst))
        return True, None
    except Exception as e:
        return False, str(e)


def restore_goal(goal_id: str) -> tuple[bool, Optional[str]]:
    """Move goal from trash/ back to history/."""
    _ensure_dirs()
    src = TRASH_DIR / f"{goal_id}.json"
    if not src.exists():
        return False, f"Goal {goal_id} not found in trash."
    dst = HISTORY_DIR / src.name
    try:
        shutil.move(str(src), str(dst))
        return True, None
    except Exception as e:
        return False, str(e)


def purge_trash() -> tuple[int, Optional[str]]:
    """Permanently delete all goals in trash/. Returns (count_deleted, error)."""
    _ensure_dirs()
    if not TRASH_DIR.exists():
        return 0, None
    count = 0
    for f in TRASH_DIR.glob("*.json"):
        try:
            f.unlink()
            count += 1
        except Exception:
            pass
    return count, None
# ----------------------------------------------------------------------
# SessionDB import utilities
# ----------------------------------------------------------------------

def _load_sessiondb_goal(session_id: str) -> Optional[dict]:
    """Load a goal from SessionDB for a given session_id. Returns None if not found."""
    try:
        import sys as _sys
        _sys.path.insert(0, "/home/opc/.hermes/hermes-agent")
        from hermes_cli.goals import load_goal
    except Exception:
        return None

    state = load_goal(session_id)
    if state is None:
        return None

    goal_id = getattr(state, "goal_id", session_id)
    checklist = getattr(state, "checklist", []) or []
    return {
        "id": goal_id,
        "text": getattr(state, "goal", ""),
        "status": getattr(state, "status", "unknown"),
        "created_at": getattr(state, "created_at", _now_iso()),
        "checklist": [
            {
                "id": str(getattr(c, "id", "")),
                "text": getattr(c, "text", ""),
                "status": getattr(c, "status", "pending"),
                "created_at": getattr(c, "created_at", ""),
            }
            for c in checklist
        ],
        "evidence": [],
        "source": "sessiondb",
        "session_id": session_id,
    }


def list_sessiondb_goals() -> list[dict]:
    """List all goals stored in SessionDB across all sessions.

    Returns a list of goal dicts compatible with the goals_cmd format.
    """
    try:
        from hermes_state import SessionDB
    except Exception:
        return []

    db = SessionDB()
    try:
        rows = db.db.execute(
            "SELECT key FROM state_meta WHERE key LIKE 'goal:%' LIMIT 100"
        ).fetchall()
    except Exception:
        return []

    results = []
    for row in rows:
        key = row[0]
        if not key.startswith("goal:"):
            continue
        session_id = key.split(":", 1)[1]
        goal = _load_sessiondb_goal(session_id)
        if goal:
            results.append(goal)
    return results


def import_from_sessiondb(session_id: str) -> tuple[Optional[dict], Optional[str]]:
    """Import a goal from SessionDB into the goals/ active system.

    Writes the goal to active.json and archives it in history/.
    After import the goal becomes the /goals active goal.
    """
    goal = _load_sessiondb_goal(session_id)
    if goal is None:
        return None, f"No goal found in SessionDB for session {session_id}"

    # Validate
    errors = _validate_goal(goal)
    if errors:
        return None, f"Invalid goal data: {', '.join(errors)}"

    # Archive to history/ first
    hist_id = f"sessiondb-{session_id}"
    hist_path = HISTORY_DIR / f"{hist_id}.json"
    try:
        _atomic_write(hist_path, goal)
    except Exception as exc:
        return None, f"Could not write to history/: {exc}"

    # Write to active.json
    ok, err = save_active_goal(goal)
    if not ok:
        return None, err

    return goal, None
