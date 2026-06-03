"""Dashboard-safe serialization helpers for goal service responses."""
from __future__ import annotations

from typing import Any


def bounded_str(value: object, limit: int = 240) -> str:
    """Return a null-free, bounded string for dashboard/API payloads."""
    text = str(value or "").replace("\x00", "")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def is_authoritative_evidence_source(source: object) -> bool:
    """True for evidence sources the core judge can treat as direct proof."""
    return str(source or "") in {"tool_output", "verifier"}


def evidence_entry_to_public_dict(entry: object, *, authoritative: bool) -> dict[str, Any]:
    """Serialize an evidence ledger entry without raw outputs or unbounded fields."""
    return {
        "created_at": float(getattr(entry, "created_at", 0.0) or 0.0),
        "source": bounded_str(getattr(entry, "source", ""), 80),
        "evidence_type": bounded_str(getattr(entry, "evidence_type", ""), 80),
        "summary": bounded_str(getattr(entry, "summary", ""), 400),
        "status": bounded_str(getattr(entry, "status", ""), 80),
        "command": bounded_str(getattr(entry, "command", ""), 240),
        "result_summary": bounded_str(getattr(entry, "result_summary", ""), 400),
        "artifact_paths": [
            bounded_str(path, 240)
            for path in list(getattr(entry, "artifact_paths", []) or [])[:10]
        ],
        "item_ids": [
            bounded_str(item_id, 100)
            for item_id in list(getattr(entry, "item_ids", []) or [])[:20]
        ],
        "authoritative": authoritative,
    }

