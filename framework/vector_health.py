"""
Lightweight checks for catalog vector stores (no embedding model load).

Chroma: on-disk persist folders under ``VECTOR_STORE_PATH``.
Qdrant: collection ``points_count`` via HTTP (``VECTOR_BACKEND=qdrant``).

Survey session stores are always Chroma and are not covered here.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.env_utils import embedding_model, notify_env_fallback_once, vector_backend
from core.paths import vector_store_root

logger = logging.getLogger(__name__)

# Config-driven system list — operators can extend without code edits.
# E.g.  RAG_SYSTEMS=designer,runtime,callcenter,admin,sales
_RAG_SYSTEMS_RAW = os.getenv("RAG_SYSTEMS", "").strip()
if not _RAG_SYSTEMS_RAW:
    notify_env_fallback_once("RAG_SYSTEMS", "designer,runtime,callcenter,admin")
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


def _qdrant_collection_ready(client: Any, name: str) -> tuple[bool, int]:
    """
    Return (ready, points_count) for a Qdrant collection.

    ``ready`` is true when the collection exists and has at least one point.
    """
    try:
        info = client.get_collection(name)
    except Exception:
        return False, 0
    try:
        cnt = int(getattr(info, "points_count", 0) or 0)
    except (TypeError, ValueError):
        cnt = 0
    return cnt > 0, cnt


def describe_vector_stores() -> dict[str, Any]:
    """Structured snapshot for logs and readiness JSON, including embedding model version check."""
    base = vector_store_root()
    current_model = embedding_model()
    cols: dict[str, Any] = {}
    backend = vector_backend()
    qdrant_client: Any = None
    if backend == "qdrant":
        try:
            from core.vector_stores import get_qdrant_client_public

            qdrant_client = get_qdrant_client_public()
        except Exception as exc:  # noqa: BLE001 — optional deps / network
            logger.warning("Qdrant client init failed for health check: %s", exc)

    for name in SYSTEMS:
        p = base / name
        meta = read_store_metadata(p)
        stored_model = meta.get("embedding_model", "")
        model_match = (stored_model == current_model) if stored_model else None
        if backend == "qdrant":
            ready, pts = (False, 0)
            if qdrant_client is not None:
                ready, pts = _qdrant_collection_ready(qdrant_client, name)
            cols[name] = {
                "path": str(p),
                "backend": "qdrant",
                "ready": ready,
                "points_count": pts,
                "embedding_model": stored_model or None,
                "embedding_model_match": model_match,
            }
        else:
            cols[name] = {
                "path": str(p),
                "backend": "chroma",
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
    return {
        "vector_backend": backend,
        "vector_store_root": str(base),
        "current_embedding_model": current_model,
        "collections": cols,
    }


def all_vector_stores_ready() -> bool:
    """True only when every expected catalog collection is populated (Chroma disk or Qdrant)."""
    info = describe_vector_stores()
    return all(c.get("ready") for c in info["collections"].values())

