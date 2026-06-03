"""
CLI command layer for persistent cross-session goal tracking.

Wires goal_store.py into the Hermes CLI as a `/goals` command family.
Subcommands: add, update, note, status, done, cancel, list, delete, restore, purge, review
"""

from __future__ import annotations

import sys
import os

# Resolve goal_store relative to this file's location
_goal_core_dir = os.path.dirname(os.path.abspath(__file__))
_hermes_cli_dir = os.path.dirname(_goal_core_dir)
sys.path.insert(0, _hermes_cli_dir)

from hermes_cli.goal_core.goal_store import (
    get_active_goal,
    create_goal,
    add_checklist_item,
    update_checklist_item,
    append_evidence,
    archive_goal,
    review_goal,
    list_goals,
    soft_delete_goal,
    restore_goal,
    purge_trash,
)


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ok(msg: str) -> str:
    return f"\033[92m✓\033[0m {msg}"


def _err(msg: str) -> str:
    return f"\033[91m✗\033[0m {msg}"


def _info(msg: str) -> str:
    return f"\033[94mℹ\033[0m {msg}"


def _fmt_status(goal: dict) -> str:
    """Format goal status with checklist table and recent evidence."""
    lines = []
    lines.append(f"\033[1mGoal\033[0m [{goal['id'][:8]}]: {goal['text']}")
    lines.append(f"Status: \033[93m{goal['status']}\033[0m | Created: {goal['created_at'][:19]}")

    if goal.get("checklist"):
        lines.append("")
        lines.append("  Checklist:")
        for item in goal["checklist"]:
            icon = {
                "pending": "☐",
                "in_progress": "◐",
                "completed": "☑",
                "cancelled": "✗",
            }.get(item["status"], "?")
            color = {
                "pending": "90",
                "in_progress": "93",
                "completed": "92",
                "cancelled": "90",
            }.get(item["status"], "90")
            lines.append(f"    \033[{color}m{icon}\033[0m [{item['id'][:8]}] {item['text']} \033[90m({item['status']})\033[0m")
    else:
        lines.append("  No checklist items yet. Use /goals add <text> to add items.")

    if goal.get("evidence"):
        lines.append("")
        lines.append("  Recent evidence:")
        for ev in goal["evidence"][-5:]:
            tool_tag = f" \033[90m[{ev.get('tool', 'note')}]\033[0m" if ev.get("tool") else ""
            lines.append(f"    \033[90m{ev['timestamp'][11:19]}\033[0m {ev['content'][:80]}{tool_tag}")

    return "\n".join(lines)


def _fmt_list(goals: list[dict]) -> str:
    if not goals:
        return "No archived goals."
    lines = []
    for g in goals:
        done = sum(1 for i in g.get("checklist", []) if i.get("status") == "completed")
        total = len(g.get("checklist", []))
        src = g.get("source", "")
        lines.append(f"  \033[94m{src or g['id'][:8]}\033[0m | {g.get('status', '?'):12} | {done}/{total} done | {g.get('text', '')[:60]}")
    return "\n".join(lines) if lines else "No archived goals."


# ─────────────────────────────────────────────────────────────────────────────
# Command handlers (each returns str output)
# ─────────────────────────────────────────────────────────────────────────────

def handle_goals(args: list[str]) -> str:
    """Main dispatcher for /goals command. args = everything after 'goals'."""
    if not args:
        return _show_status()

    sub = args[0].lower()
    if sub == "add":
        if len(args) < 2:
            return _err("Usage: /goals add <text>")
        text = " ".join(args[1:])
        goal, err = add_checklist_item(text)
        if err:
            return _err(err)
        item = goal["checklist"][-1]
        return _ok(f"Added: {item['text']} [{item['id'][:8]}]")

    elif sub == "update":
        if len(args) < 3:
            return _err("Usage: /goals update <item_id> <pending|in_progress|completed|cancelled>")
        item_id, new_status = args[1], args[2].lower()
        goal, err = update_checklist_item(item_id, new_status)
        if err:
            return _err(err)
        return _ok(f"Updated {item_id[:8]} → {new_status}")

    elif sub == "note":
        if len(args) < 2:
            return _err("Usage: /goals note <text> [--tool <name>] [--meta <json>]")
        # Parse optional flags
        text_parts = []
        tool = None
        meta = None
        i = 1
        while i < len(args):
            if args[i] == "--tool" and i + 1 < len(args):
                tool = args[i + 1]
                i += 2
            elif args[i] == "--meta" and i + 1 < len(args):
                import json as _json
                try:
                    meta = _json.loads(args[i + 1])
                except Exception:
                    return _err(f"Invalid JSON in --meta: {args[i+1]}")
                i += 2
            else:
                text_parts.append(args[i])
                i += 1
        content = " ".join(text_parts)
        ok, err = append_evidence(content, tool=tool, meta=meta)
        if not ok:
            return _err(err)
        return _ok(f"Evidence appended: {content[:60]}...")

    elif sub in ("status", "show"):
        return _show_status()

    elif sub in ("done", "complete"):
        goal, _ = get_active_goal()
        if not goal:
            return _err("No active goal.")
        completed = sum(1 for i in goal.get("checklist", []) if i.get("status") == "completed")
        total = len(goal.get("checklist", []))
        if total > 0 and completed < total:
            return _err(f"Only {completed}/{total} checklist items completed. Use /goals cancel to archive anyway, or complete remaining items first.")
        ok, err = archive_goal("completed")
        if not ok:
            return _err(err)
        return _ok("Goal archived as completed.")

    elif sub in ("cancel",):
        ok, err = archive_goal("cancelled")
        if not ok:
            return _err(err)
        return _ok("Goal archived and cancelled.")

    elif sub in ("list", "ls"):
        goals = list_goals(include_trash=False)
        if not goals:
            return "No archived goals. Use /goals list --all to include trash."
        return _fmt_list(goals)

    elif sub == "list" and "--all" in args:
        goals = list_goals(include_trash=True)
        return _fmt_list(goals)

    elif sub in ("delete", "trash"):
        if len(args) < 2:
            return _err("Usage: /goals delete <goal_id>")
        goal_id = args[1]
        ok, err = soft_delete_goal(goal_id)
        if not ok:
            return _err(err)
        return _ok(f"Goal moved to trash. Use /goals restore {goal_id} to recover.")

    elif sub in ("restore",):
        if len(args) < 2:
            return _err("Usage: /goals restore <goal_id>")
        goal_id = args[1]
        ok, err = restore_goal(goal_id)
        if not ok:
            return _err(err)
        return _ok(f"Goal restored from trash.")

    elif sub in ("purge",):
        count, err = purge_trash()
        if err:
            return _err(err)
        return _ok(f"Permanently deleted {count} goal(s) from trash.")

    elif sub in ("review",):
        fixed, err, msg = review_goal()
        if err:
            return f"\033[93m⚠ needs_review\033[0m\n  {err}\n  {msg or ''}"
        return _ok(msg or "Goal is valid and active.")

    elif sub in ("help", "--help", "-h"):
        return """\033[1m/goals command reference\033[0m

  /goals add <text>              Append checklist item
  /goals update <id> <status>    Update item status (pending/in_progress/completed/cancelled)
  /goals note <text> [--tool n]  Append evidence entry
  /goals status                  Show active goal + checklist + recent evidence
  /goals done                   Archive as completed (fails if items remain)
  /goals cancel                 Archive and cancel active goal
  /goals list                   List archived goals (history/)
  /goals list --all             Include trash/
  /goals delete <id>            Move goal to trash/ (recoverable)
  /goals restore <id>           Restore from trash/
  /goals purge                  Permanently delete all trash/
  /goals review                 Validate and reactivate needs_review goal
  /goals help                   Show this reference

Item IDs are the first 8 characters of the UUID shown in /goals status.
"""

    else:
        return _err(f"Unknown /goals subcommand: {sub}\nRun /goals help for reference.")


def _show_status() -> str:
    goal, err = get_active_goal()
    if err:
        return f"\033[93m⚠ active.json error:\033[0m {err}\nRun /goals review to diagnose."
    if not goal:
        return "No active goal. Use /goal <text> to create one (the existing goal system),\nor import one from SessionDB."
    return _fmt_status(goal)
