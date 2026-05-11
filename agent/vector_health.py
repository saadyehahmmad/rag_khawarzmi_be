"""
Lightweight checks for on-disk Chroma stores (no embedding model load).

Used by the API lifespan and /health/ready so operators can detect a missing ingest
before users hit opaque empty-retrieval behavior.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from agent.env_utils import EMBEDDING_MODEL
from agent.paths import vector_store_root

logger = logging.getLogger(__name__)

# Config-driven system list — operators can extend without code edits.
# E.g.  RAG_SYSTEMS=designer,runtime,callcenter,admin,sales
_RAG_SYSTEMS_RAW = os.getenv("RAG_SYSTEMS", "").strip()
SYSTEMS: tuple[str, ...] = (
    tuple(s.strip() for s in _RAG_SYSTEMS_RAW.split(",") if s.strip())
    if _RAG_SYSTEMS_RAW
    else ("designer", "runtime", "callcenter", "admin")
)


def collection_dir_ready(path: Path) -> bool:
    """
    True if the directory exists and looks like a Chroma persist folder.

    Newer Chroma versions use chroma.sqlite3; older layouts may only have UUID subdirs.
    """
    if not path.is_dir():
        return False
    if (path / "chroma.sqlite3").is_file():
        return True
    try:
        return any(path.iterdir())
    except OSError:
        return False


def read_store_metadata(path: Path) -> dict[str, Any]:
    """
    Read .metadata.json written by ingestion (contains embedding_model, ingest_time, etc.).

    Returns an empty dict when the file is absent (e.g. legacy stores ingested before versioning).
    """
    meta_file = path / ".metadata.json"
    if not meta_file.is_file():
        return {}
    try:
        return json.loads(meta_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def describe_vector_stores() -> dict[str, Any]:
    """Structured snapshot for logs and readiness JSON, including embedding model version check."""
    base = vector_store_root()
    current_model = EMBEDDING_MODEL
    cols: dict[str, Any] = {}
    for name in SYSTEMS:
        p = base / name
        meta = read_store_metadata(p)
        stored_model = meta.get("embedding_model", "")
        model_match = (stored_model == current_model) if stored_model else None
        cols[name] = {
            "path": str(p),
            "ready": collection_dir_ready(p),
            "embedding_model": stored_model or None,
            "embedding_model_match": model_match,
        }
        if model_match is False:
            logger.warning(
                "Embedding model mismatch for '%s': stored=%r current=%r — re-ingest required",
                name,
                stored_model,
                current_model,
            )
    return {"vector_store_root": str(base), "current_embedding_model": current_model, "collections": cols}


def all_vector_stores_ready() -> bool:
    """True only when every expected system collection directory is populated."""
    info = describe_vector_stores()
    return all(c.get("ready") for c in info["collections"].values())

