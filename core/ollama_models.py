"""
Query the local Ollama HTTP API for installed model tags (``GET /api/tags``).

Used by ``GET /llm/ollama/models`` so clients can populate model pickers without shell access.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from core.env_utils import env_default_float, ollama_base_url

logger = logging.getLogger(__name__)


def _list_timeout_sec() -> float:
    try:
        v = env_default_float("OLLAMA_LIST_TIMEOUT_SEC", 5.0)
    except ValueError:
        msg = "Invalid OLLAMA_LIST_TIMEOUT_SEC - using 5.0"
        logger.warning(msg)
        print(f"[env] {msg}", flush=True)
        v = 5.0
    return max(0.5, min(60.0, v))


def list_local_ollama_models(
    *,
    base_url: str | None = None,
    timeout_sec: float | None = None,
) -> dict[str, Any]:
    """
    Call Ollama ``GET /api/tags`` and return a stable JSON shape for the API layer.

    Args:
        base_url: Override ``OLLAMA_BASE_URL`` (default: from env).
        timeout_sec: HTTP timeout in seconds (default: ``OLLAMA_LIST_TIMEOUT_SEC`` or 5).

    Returns:
        ``ok``, ``base_url``, ``models`` (list of dicts with at least ``name``), and optional ``error``.
    """
    base = (base_url or ollama_base_url()).strip().rstrip("/") or "http://localhost:11434"
    to = _list_timeout_sec() if timeout_sec is None else max(0.5, min(60.0, float(timeout_sec)))
    url = f"{base}/api/tags"
    try:
        with httpx.Client(timeout=to) as client:
            r = client.get(url)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError as exc:
        logger.warning("Ollama list models failed (%s): %s", url, exc)
        return {"ok": False, "base_url": base, "models": [], "error": str(exc)}
    except ValueError as exc:
        return {"ok": False, "base_url": base, "models": [], "error": f"invalid JSON: {exc}"}

    if not isinstance(data, dict):
        return {"ok": False, "base_url": base, "models": [], "error": "unexpected response: not an object"}

    raw_models = data.get("models")
    if not isinstance(raw_models, list):
        return {"ok": False, "base_url": base, "models": [], "error": "unexpected response: missing models array"}

    models: list[dict[str, Any]] = []
    for row in raw_models:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or row.get("model") or "").strip()
        if not name:
            continue
        entry: dict[str, Any] = {"name": name}
        if "size" in row and isinstance(row["size"], int):
            entry["size"] = row["size"]
        if "modified_at" in row and isinstance(row["modified_at"], str):
            entry["modified_at"] = row["modified_at"]
        details = row.get("details")
        if isinstance(details, dict):
            ps = details.get("parameter_size")
            q = details.get("quantization_level")
            if isinstance(ps, str) and ps:
                entry["parameter_size"] = ps
            if isinstance(q, str) and q:
                entry["quantization_level"] = q
        models.append(entry)

    return {"ok": True, "base_url": base, "models": models, "error": None}


__all__ = ["list_local_ollama_models"]
