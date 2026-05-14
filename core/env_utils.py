"""
Shared environment-variable helpers for core, ingestion, framework, and API code.

All accessors read ``os.environ`` when called (not at import time), so ``load_dotenv()``
can run before first use and values stay consistent with the process environment.

When a variable is unset or blank and a built-in default is used, the process logs
and prints a one-line notice (once per variable/default pair) so operators see it
in both structured logs and the console.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_RE_TRUTH: frozenset[str] = frozenset({"1", "true", "yes"})

_DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
_DEFAULT_OLLAMA_URL = "http://localhost:11434"
_DEFAULT_QDRANT_URL = "http://localhost:6333"

# Log at most once per distinct invalid VECTOR_BACKEND value (avoids log spam).
_vector_backend_warned_for: str | None = None

# One notice per (name, repr(default)) when a configured default replaces missing/empty env.
_env_fallback_notified: set[str] = set()


def notify_env_fallback_once(name: str, default: object) -> None:
    """
    Log and print once per process when ``name`` is unset/empty and ``default`` is used.

    Use from modules that read ``os.environ`` directly so fallback behaviour matches
    ``env_default_*`` helpers.

    Args:
        name: Environment variable name.
        default: Value substituted (shown in the message).
    """
    key = f"{name}\x00{default!r}"
    if key in _env_fallback_notified:
        return
    _env_fallback_notified.add(key)
    msg = f"[env] {name!r} is unset or empty - using default {default!r}"
    logger.info(msg)
    print(msg, flush=True)


def env_default_str(name: str, default: str) -> str:
    """
    Return stripped ``name`` from the environment, or ``default`` when missing/blank.

    Notifies once (log + print) when ``default`` is used.
    """
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        notify_env_fallback_once(name, default)
        return default
    return str(raw).strip()


def env_default_int(name: str, default: int) -> int:
    """
    Parse ``name`` as int, or return ``default`` when missing/blank.

    Notifies once when ``default`` is used. Malformed non-empty values still raise ``ValueError``.
    """
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        notify_env_fallback_once(name, default)
        return default
    return int(str(raw).strip(), 10)


def env_default_float(name: str, default: float) -> float:
    """Parse ``name`` as float, or ``default`` when missing/blank; notifies once when defaulted."""
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        notify_env_fallback_once(name, default)
        return default
    return float(str(raw).strip())

def embedding_model() -> str:
    """
    HuggingFace embedding model id (CPU, L2-normalised in ``get_embeddings``).

    Returns:
        Non-empty model name; defaults to ``intfloat/multilingual-e5-large``.
    """
    raw = os.environ.get("EMBEDDING_MODEL")
    if raw is None or not str(raw).strip():
        notify_env_fallback_once("EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL)
        return _DEFAULT_EMBEDDING_MODEL
    return str(raw).strip()


def vector_backend() -> str:
    """
    Which backend backs **catalog** collections (designer, runtime, …).

    Survey session stores remain Chroma-only regardless of this setting.

    Returns:
        ``chroma`` or ``qdrant``. Unknown values fall back to ``chroma``; typo ``qudrant`` maps to ``qdrant``.
    """
    global _vector_backend_warned_for
    raw_src = os.environ.get("VECTOR_BACKEND")
    if raw_src is None or not str(raw_src).strip():
        notify_env_fallback_once("VECTOR_BACKEND", "chroma")
        return "chroma"
    raw = str(raw_src).strip().lower()
    normalized = "qdrant" if raw in ("qdrant", "qudrant") else raw
    if normalized == "qdrant":
        return "qdrant"
    if normalized == "chroma":
        return "chroma"
    if _vector_backend_warned_for != raw:
        _vector_backend_warned_for = raw
        msg = f"Unknown VECTOR_BACKEND={raw!r} - using chroma"
        logger.warning(msg)
        print(f"[env] {msg}", flush=True)
    return "chroma"


def qdrant_url() -> str:
    """Qdrant HTTP(S) base URL (default ``http://localhost:6333``)."""
    raw = os.environ.get("QDRANT_URL")
    if raw is None or not str(raw).strip():
        notify_env_fallback_once("QDRANT_URL", _DEFAULT_QDRANT_URL)
        return _DEFAULT_QDRANT_URL
    return str(raw).strip()


def qdrant_api_key() -> str | None:
    """Optional Qdrant API key (cloud / secured instances)."""
    key = os.getenv("QDRANT_API_KEY", "").strip()
    return key or None


def ollama_base_url() -> str:
    """Ollama HTTP API base URL (no trailing slash)."""
    raw = os.environ.get("OLLAMA_BASE_URL")
    if raw is None or not str(raw).strip():
        notify_env_fallback_once("OLLAMA_BASE_URL", _DEFAULT_OLLAMA_URL)
        return _DEFAULT_OLLAMA_URL
    return str(raw).strip().rstrip("/") or _DEFAULT_OLLAMA_URL


def llm_provider() -> str:
    """
    Active chat LLM vendor for RAG answer + rewrite nodes.

    Returns:
        ``anthropic``, ``openai``, ``gemma``, or ``ollama``. Unknown values fall back to ``anthropic``.
    """
    raw = os.environ.get("LLM_PROVIDER")
    if raw is None or not str(raw).strip():
        notify_env_fallback_once("LLM_PROVIDER", "anthropic")
        return "anthropic"
    p = str(raw).strip().lower()
    if p in ("gemma", "google", "google_gemini", "google-gemini"):
        return "gemma"
    if p == "openai":
        return "openai"
    if p in ("ollama", "local"):
        return "ollama"
    if p == "anthropic":
        return "anthropic"
    msg = f"Unknown LLM_PROVIDER={p!r} - using anthropic"
    logger.warning(msg)
    print(f"[env] {msg}", flush=True)
    return "anthropic"


def env_bool(name: str, default: bool = False) -> bool:
    """
    Parse a common true/false environment variable.

    Returns ``default`` when the variable is unset or blank (notifies once when ``default`` is used).
    Recognised truthy values (case-insensitive): ``1``, ``true``, ``yes``.

    Args:
        name: Environment variable name.
        default: Value to return when the variable is absent or empty.

    Returns:
        Parsed boolean value.
    """
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        notify_env_fallback_once(name, default)
        return default
    return str(raw).strip().lower() in _RE_TRUTH


__all__ = [
    "embedding_model",
    "env_bool",
    "env_default_float",
    "env_default_int",
    "env_default_str",
    "llm_provider",
    "notify_env_fallback_once",
    "ollama_base_url",
    "qdrant_api_key",
    "qdrant_url",
    "vector_backend",
]
