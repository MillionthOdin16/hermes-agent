"""Goal-system external evidence producer.

This plugin is deliberately passive: it reads the current turn context that
Hermes already has, extracts small summaries of obvious verification evidence,
and returns ledger payloads through the ``goal_external_evidence`` hook. It
does not run commands, inspect file contents, or decide completion.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List

_TEXT_CAP = 4000
_SUMMARY_CAP = 500
_PATH_CAP = 300
_MAX_PAYLOADS = 4
_MAX_PATHS = 8

_TEST_RESULT_RE = re.compile(
    r"\b(?P<count>\d+)\s+"
    r"(?P<status>passed|failed|errors?|skipped|xfailed|xpassed)\b",
    re.IGNORECASE,
)
_COMMAND_LINE_RE = re.compile(
    r"(?im)^\s*(?:command\s*:\s*)?"
    r"(?P<command>(?:[\w./-]*pytest|python\s+-m\s+pytest|ruff\s+check|"
    r"python\s+-m\s+py_compile|npm\s+(?:test|run\s+test)|pnpm\s+test|"
    r"yarn\s+test)[^\n\r]*)$"
)
_ARTIFACT_PATH_RE = re.compile(
    r"(?<![\w@:])("
    r"(?:/|~/)[A-Za-z0-9._@%+\-/]+?\.[A-Za-z0-9]{1,16}"
    r"|(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.@%+-]+?\.[A-Za-z0-9]{1,16}"
    r")"
)


def _truncate(value: str, limit: int) -> str:
    value = str(value or "")
    return value if len(value) <= limit else value[:limit] + "... [truncated]"


def _message_texts(messages: Any) -> List[str]:
    texts: List[str] = []
    if not isinstance(messages, list):
        return texts
    for msg in messages[-12:]:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "")
        if role not in {"assistant", "tool"}:
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            texts.append(_truncate(content, _TEXT_CAP))
    return texts


def _extract_command(text: str) -> str | None:
    for match in _COMMAND_LINE_RE.finditer(text or ""):
        command = match.group("command").strip()
        if command:
            return _truncate(command, 200)
    return None


def _extract_test_payload(text: str) -> Dict[str, Any] | None:
    match = _TEST_RESULT_RE.search(text or "")
    if not match:
        return None
    result = f"{match.group('count')} {match.group('status').lower()}"
    command = _extract_command(text)
    summary = f"Detected verification result in current turn: {result}"
    payload: Dict[str, Any] = {
        "evidence_type": "test_result",
        "summary": _truncate(summary, _SUMMARY_CAP),
        "result_summary": result,
        "status": "failed" if "fail" in result or "error" in result else "passed",
    }
    if command:
        payload["command"] = command
    return payload


def _extract_artifact_payload(text: str) -> Dict[str, Any] | None:
    paths: List[str] = []
    seen = set()
    for match in _ARTIFACT_PATH_RE.finditer(text or ""):
        path = _truncate(match.group(1), _PATH_CAP)
        if path in seen:
            continue
        seen.add(path)
        paths.append(path)
        if len(paths) >= _MAX_PATHS:
            break
    if not paths:
        return None
    return {
        "evidence_type": "file_artifact",
        "summary": f"Detected artifact/path references in current turn: {len(paths)} path(s)",
        "artifact_paths": paths,
    }


def _approved_verifier_roots() -> List[Path]:
    raw = os.environ.get("HERMES_GOAL_FILE_VERIFIER_ROOTS", "")
    roots: List[Path] = []
    for item in raw.split(os.pathsep):
        if not item.strip():
            continue
        try:
            root = Path(item).expanduser().resolve()
        except (OSError, RuntimeError):
            continue
        if root.exists() and root.is_dir():
            roots.append(root)
    return roots


def _is_under_root(path: Path, roots: List[Path]) -> bool:
    for root in roots:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _extract_verified_file_payload(text: str) -> Dict[str, Any] | None:
    roots = _approved_verifier_roots()
    if not roots:
        return None
    verified: List[str] = []
    seen = set()
    for match in _ARTIFACT_PATH_RE.finditer(text or ""):
        raw_path = match.group(1)
        try:
            candidate = Path(raw_path).expanduser()
            if not candidate.is_absolute():
                continue
            resolved = candidate.resolve()
        except (OSError, RuntimeError):
            continue
        if str(resolved) in seen or not _is_under_root(resolved, roots):
            continue
        if resolved.exists() and resolved.is_file():
            seen.add(str(resolved))
            verified.append(_truncate(str(resolved), _PATH_CAP))
        if len(verified) >= _MAX_PATHS:
            break
    if not verified:
        return None
    return {
        "evidence_type": "verification_summary",
        "summary": f"Verified artifact exists under approved root: {Path(verified[0]).name}",
        "artifact_paths": verified,
        "status": "passed",
        "result_summary": "file exists",
    }


def collect_goal_external_evidence(
    *,
    session_id: str = "",
    goal: str = "",
    final_response: str = "",
    messages: Any = None,
    **_: Any,
) -> List[Dict[str, Any]]:
    """Return bounded evidence payloads for the active goal ledger."""
    del session_id, goal
    texts = [_truncate(final_response or "", _TEXT_CAP)]
    texts.extend(_message_texts(messages))
    joined = "\n".join(t for t in texts if t.strip())
    if not joined.strip():
        return []

    payloads: List[Dict[str, Any]] = []
    test_payload = _extract_test_payload(joined)
    if test_payload:
        payloads.append(test_payload)
    artifact_payload = _extract_artifact_payload(joined)
    if artifact_payload:
        payloads.append(artifact_payload)
    verified_file_payload = _extract_verified_file_payload(joined)
    if verified_file_payload:
        payloads.append(verified_file_payload)
    return payloads[:_MAX_PAYLOADS]

