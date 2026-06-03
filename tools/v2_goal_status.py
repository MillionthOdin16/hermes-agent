#!/usr/bin/env python3
"""Quick v2 goal persistence status inspection.

Usage:
    python3 tools/v2_goal_status.py

Answers:
- v2 enabled?
- schema version?
- number of active / archived / total goals
- active goal sessions
- any legacy state_meta goal:* rows still present
"""

import os, sys
from pathlib import Path

# Ensure we're in the Hermes venv
venv = Path.home() / ".hermes" / "hermes-agent" / "venv"
python = venv / "bin" / "python3"
if python.exists() and sys.executable != str(python):
    os.execv(str(python), [str(python), __file__] + sys.argv[1:])

sys.path.insert(0, str(Path.home() / ".hermes" / "hermes-agent"))

from hermes_cli.goal_core.repository import _v2_enabled, GoalRepository
from hermes_state import SessionDB, SCHEMA_VERSION

v2 = _v2_enabled()
print(f"v2 enabled:             {v2}")
print(f"schema version:         {SCHEMA_VERSION}")

db = SessionDB()

if v2:
    repo = GoalRepository(db=db)
    active = repo.list_goals(status="active")
    archived = repo.list_goals(status="archived")
    total = repo.list_goals()
    print(f"active goals:           {len(active)}")
    print(f"archived goals:         {len(archived)}")
    print(f"total goals:            {len(total)}")
    print()
    if active:
        print("Active goal sessions:")
        for g in active:
            sid = g.session_id or "(no session)"
            print(f"  {sid:45s} {g.goal_text[:60]}")
    else:
        print("No active goals.")
else:
    print("Active goals:           N/A (v2 disabled)")
    print("Archived goals:         N/A (v2 disabled)")
    print()

# Legacy state_meta check
rows = db._conn.execute(
    "SELECT key FROM state_meta WHERE key LIKE 'goal:%'"
).fetchall()
if rows:
    print(f"\nLegacy state_meta goal:* rows: {len(rows)} present")
    # Show first 5
    for r in rows[:5]:
        print(f"  {r[0]}")
    if len(rows) > 5:
        print(f"  ... and {len(rows)-5} more")
else:
    print("Legacy state_meta goal:* rows: none")

db.close()
