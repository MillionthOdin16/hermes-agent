"""Shared bounded sanitization helpers for goal state, trace, and evidence."""
from __future__ import annotations

import re
from typing import Any, Dict


MAX_EVENT_STRING = 200
MAX_EVENT_PAYLOAD_KEYS = 10
MAX_EVENT_DEPTH = 3

EVENT_BLOCKED_KEYS_LOWER = frozenset({
    "raw_response", "full_response", "tool_output", "fetched_content",
    "file_content", "content", "body", "message_content", "response_text",
    "tooloutput",
})
EVENT_BLOCKED_KEYS_NORM = EVENT_BLOCKED_KEYS_LOWER | frozenset(
    k.lower().replace("_", "") for k in EVENT_BLOCKED_KEYS_LOWER
)

SECRET_URL_KEYWORDS = ("api_key", "apikey", "token", "secret", "password", "auth=", "credential")
SENSITIVE_PATH_MARKERS = (
    "/.ssh/", ".ssh/", "/.env", ".env", "/credentials", "credentials",
    "/secrets", "secrets", "/id_rsa", "id_rsa", "/.netrc", ".netrc",
    "/.npmrc", ".npmrc", "/.pypirc", ".pypirc",
)

SENSITIVE_PATH_PATTERNS = [
    re.compile(r"(?:^|[/\\])\.ssh(?:[/\\.]|$)", re.IGNORECASE),
    re.compile(r"(?:^|[/\\])\.env(?:$|[/\\.]|\b)", re.IGNORECASE),
    re.compile(r"(?:^|[/\\])(?:credentials|secrets)(?:$|[/\\])", re.IGNORECASE),
    re.compile(r"(?:credentials|secrets|apikey|api_key)\.(?:json|yaml|yml|txt|conf|cfg)", re.IGNORECASE),
    re.compile(r"(?:^|[/\\])id_rsa(?:$|[/\\])", re.IGNORECASE),
    re.compile(r"(?:^|[/\\])\.netrc$", re.IGNORECASE),
    re.compile(r"(?:^|[/\\])\.pgpass$", re.IGNORECASE),
    re.compile(r"(?:^|[/\\])\.npmrc$", re.IGNORECASE),
    re.compile(r"(?:^|[/\\])\.pypirc$", re.IGNORECASE),
]


def truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "… [truncated]"


def is_blocked_key(key: str) -> bool:
    """Case-insensitive blocked key check with variant normalization."""
    return key.lower() in EVENT_BLOCKED_KEYS_NORM


def sanitize_event_string(value: str) -> str:
    """Sanitize a single event string."""
    v = value[:MAX_EVENT_STRING] if len(value) > MAX_EVENT_STRING else value
    if "://" in v and any(kw in v.lower() for kw in SECRET_URL_KEYWORDS):
        return "[redacted]"
    if "://" in v:
        v = re.sub(r"(https?://)([^@/]+)@", r"\1***@", v)
    lower_v = v.lower()
    if any(marker in lower_v for marker in SENSITIVE_PATH_MARKERS):
        return "[redacted sensitive path]"
    return v


def sanitize_goal_event_payload(data: Dict[str, Any], _depth: int = 0) -> Dict[str, Any]:
    """Recursively sanitize and bound event payload values."""
    if _depth >= MAX_EVENT_DEPTH:
        return {"_summary": f"truncated at depth {MAX_EVENT_DEPTH}"}

    sanitized: Dict[str, Any] = {}
    for key, value in data.items():
        if is_blocked_key(key):
            continue
        if isinstance(value, str):
            sanitized[key] = sanitize_event_string(value)
        elif isinstance(value, (int, float, bool)) or value is None:
            sanitized[key] = value
        elif isinstance(value, list):
            sanitized[key] = sanitize_event_list(value, _depth)
        elif isinstance(value, dict):
            sanitized[key] = sanitize_goal_event_payload(value, _depth + 1)
        else:
            sanitized[key] = sanitize_event_string(str(value))
        if len(sanitized) >= MAX_EVENT_PAYLOAD_KEYS:
            break
    return sanitized


def sanitize_event_list(items: list, _depth: int) -> list:
    """Sanitize a list recursively, bounded to five entries."""
    result = []
    for item in items[:5]:
        if isinstance(item, str):
            result.append(sanitize_event_string(item))
        elif isinstance(item, dict):
            result.append(sanitize_goal_event_payload(item, _depth + 1))
        elif isinstance(item, list):
            result.append(sanitize_event_list(item, _depth + 1))
        else:
            result.append(item)
    if len(items) > 5:
        result.append(f"... ({len(items)} total)")
    return result


def sanitize_evidence_packet_text(text: str) -> str:
    """Redact secrets and sensitive paths from evidence packet text."""
    text = re.sub(
        r"(https?://)([^/\s]*):([^/\s@]+)@",
        r"\1\2:***@",
        text,
    )
    text = re.sub(
        r"\b[A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD|CREDENTIALS)[A-Z0-9_]*\s*=\s*[^\s,;]+",
        "[redacted credential]",
        text,
        flags=re.IGNORECASE,
    )
    for secret_key in ("api_key", "apikey", "token", "secret", "password", "auth", "credential"):
        text = re.sub(
            rf"({secret_key}=)[^&\s]+",
            r"\1[redacted]",
            text,
            flags=re.IGNORECASE,
        )
    for pat in SENSITIVE_PATH_PATTERNS:
        text = pat.sub("[redacted sensitive path]", text)
    return text


_truncate = truncate
_is_blocked_key = is_blocked_key
_sanitize_event_string = sanitize_event_string
_sanitize_goal_event_payload = sanitize_goal_event_payload
_sanitize_event_list = sanitize_event_list
_sanitize_evidence_packet_text = sanitize_evidence_packet_text
