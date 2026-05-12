"""
Shared environment-variable helpers used across agent, ingestion, and API modules.

Centralising the boolean parser eliminates the five near-identical inline
implementations that previously existed in nodes.py, retrieval.py,
observability.py, and ingestion/config.py.
"""

from __future__ import annotations

import os

_RE_TRUTH: frozenset[str] = frozenset({"1", "true", "yes"})

# Single source of truth for the embedding model name used by ingestion, agent, and health checks.
# Default is the *small* variant (~120 MB RAM) so the API fits inside Render's 512 MB free tier.
# Override with EMBEDDING_MODEL=intfloat/multilingual-e5-large on instances with ≥2 GB RAM.
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")


def env_bool(name: str, default: bool = False) -> bool:
    """
    Parse a common true/false environment variable.

    Returns ``default`` when the variable is unset or blank.
    Recognised truthy values (case-insensitive): ``1``, ``true``, ``yes``.

    Args:
        name: Environment variable name.
        default: Value to return when the variable is absent or empty.

    Returns:
        Parsed boolean value.
    """
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in _RE_TRUTH
