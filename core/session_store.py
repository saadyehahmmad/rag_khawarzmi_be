"""
In-memory ingestion state for session-scoped survey collections.

Tracks which surveys have been embedded into their own Chroma collection so the
chat endpoint can check availability without hitting the filesystem on every request.

Thread-safe: uses a threading.Lock because BackgroundTasks run in a thread-pool.
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from typing import Any, Literal

SurveyStatus = Literal["ingesting", "ready", "failed"]

_lock = threading.Lock()
_state: dict[str, dict[str, Any]] = {}
# { survey_id_str: { "status": ..., "version_hash": ..., "ingested_at": ..., "question_count": int } }


def set_status(survey_id: str | int, status: SurveyStatus, **kwargs: Any) -> None:
    """Update ingestion state for a survey. Extra kwargs are merged into the record."""
    key = str(survey_id)
    with _lock:
        entry = _state.setdefault(key, {})
        entry["status"] = status
        entry.update(kwargs)
        if status == "ready":
            entry.setdefault("ingested_at", datetime.now(timezone.utc).isoformat())


def get_status(survey_id: str | int) -> dict[str, Any] | None:
    """Return a copy of the state record, or None if survey was never ingested."""
    key = str(survey_id)
    with _lock:
        return dict(_state[key]) if key in _state else None


def is_ready(survey_id: str | int) -> bool:
    """Return True only when the survey collection is fully ingested and ready."""
    s = get_status(survey_id)
    return s is not None and s.get("status") == "ready"


def compute_hash(data: dict[str, Any]) -> str:
    """
    Compute a short content hash of the survey data dict.

    Used for change detection: if the hash matches the stored one, re-ingestion
    is skipped (returns status='skipped' at the API layer).
    """
    raw = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
