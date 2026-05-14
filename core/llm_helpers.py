"""
Lazy singletons and helpers for LLM and embedding model access.

Separates infrastructure concerns (model loading, retry logic) from the pipeline
node logic. Survey-scoped Chroma is provided by ``framework.survey_store``.
Catalog vector stores (Chroma or Qdrant) are provided by ``core.vector_stores``.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings

from core.env_utils import embedding_model, env_default_float, env_default_int, llm_provider, ollama_base_url
from core.ollama_models import list_local_ollama_models
from core.vector_stores import get_vector_store

logger = logging.getLogger(__name__)

LLM_MAX_TOKENS = max(512, env_default_int("LLM_MAX_TOKENS", 4096))
LLM_MAX_RETRIES = max(1, env_default_int("LLM_MAX_RETRIES", 3))
LLM_RETRY_BASE_SEC = env_default_float("LLM_RETRY_BASE_SEC", 0.6)

HF_TOKEN = os.getenv("HF_TOKEN")

_llm: BaseChatModel | None = None
_embeddings: HuggingFaceEmbeddings | None = None


def _resolved_llm_model() -> str:
    """Model id for the active ``LLM_PROVIDER`` (``LLM_MODEL`` overrides per-provider defaults)."""
    raw = os.getenv("LLM_MODEL", "").strip()
    if raw:
        return raw
    prov = llm_provider()
    if prov == "openai":
        return "gpt-4o-mini"
    if prov == "gemma":
        return "gemma-2-9b-it"
    if prov == "ollama":
        return "gemma2:9b"
    return "claude-haiku-4-5-20251001"


def _build_chat_model() -> BaseChatModel:
    """Construct the chat model for the configured ``LLM_PROVIDER``."""
    provider = llm_provider()
    model = _resolved_llm_model()
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, max_tokens=LLM_MAX_TOKENS)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, max_tokens=LLM_MAX_TOKENS)
    if provider == "gemma":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "LLM_PROVIDER=gemma requires langchain-google-genai and GOOGLE_API_KEY "
                "(see requirements.txt)."
            ) from exc

        return ChatGoogleGenerativeAI(model=model, max_output_tokens=LLM_MAX_TOKENS)
    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "LLM_PROVIDER=ollama requires langchain-ollama (see requirements.txt)."
            ) from exc

        return ChatOllama(
            model=model,
            base_url=ollama_base_url(),
            num_predict=LLM_MAX_TOKENS,
        )
    raise RuntimeError(f"unsupported LLM_PROVIDER: {provider!r}")


def get_llm() -> BaseChatModel:
    """Lazy singleton for the configured chat model (Anthropic, OpenAI, Google Gemma, or Ollama)."""
    global _llm
    if _llm is None:
        _llm = _build_chat_model()
    return _llm


def llm_config_snapshot() -> dict[str, Any]:
    """
    Non-secret configuration for clients (e.g. Angular) to align UI with the running API.

    Returns:
        Provider id, resolved ``LLM_MODEL`` (or default for that provider), optional raw env override,
        and the configured Ollama base URL (for ``GET /llm/ollama/models`` probes).
    """
    raw_override = os.getenv("LLM_MODEL", "").strip()
    return {
        "llm_provider": llm_provider(),
        "llm_model": _resolved_llm_model(),
        "llm_model_env": raw_override or None,
        "ollama_base_url": ollama_base_url(),
    }


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Lazy singleton for the local HuggingFace embedding model (CPU, L2-normalised).

    ``normalize_embeddings=True`` is required for dot-product == cosine similarity.
    """
    global _embeddings
    if _embeddings is None:
        model_kwargs: dict[str, Any] = {"device": "cpu"}
        if HF_TOKEN:
            model_kwargs["token"] = HF_TOKEN
        _embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model(),
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def chunk_content_to_text(chunk: Any) -> str:
    """Normalize an AIMessage / chunk .content (str or list of content blocks) to plain text."""
    c = getattr(chunk, "content", None)
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for p in c:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("text", "")))
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts)
    return ""


def invoke_llm_text(llm: BaseChatModel, prompt: str) -> str:
    """
    Invoke the chat model and return the plain-text response.

    Retries up to ``LLM_MAX_RETRIES`` times with exponential back-off for
    transient network and I/O errors.
    """
    messages = [HumanMessage(content=prompt)]
    last_exc: BaseException | None = None
    for attempt in range(LLM_MAX_RETRIES):
        try:
            msg = llm.invoke(messages)
            return chunk_content_to_text(msg).strip()
        except (TimeoutError, ConnectionError, OSError) as exc:
            last_exc = exc
        except Exception as exc:  # noqa: BLE001 — LangChain wraps provider errors
            last_exc = exc
        if attempt >= LLM_MAX_RETRIES - 1:
            logger.exception("LLM invoke failed after %s attempts", LLM_MAX_RETRIES)
            break
        delay = LLM_RETRY_BASE_SEC * (2**attempt)
        logger.warning(
            "LLM invoke failed (attempt %s/%s): %s; sleeping %.1fs",
            attempt + 1,
            LLM_MAX_RETRIES,
            last_exc,
            delay,
        )
        time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def _ollama_configured_model_locally_installed(model_names: list[str], configured: str) -> bool:
    """
    Return True if ``configured`` matches an Ollama tag from ``GET /api/tags`` (exact or base name).

    Args:
        model_names: ``name`` fields from Ollama (e.g. ``gemma2:9b``, ``llama3:latest``).
        configured: Resolved ``LLM_MODEL`` (e.g. ``gemma2:9b`` or ``llama3``).

    Returns:
        Whether a pull/run is likely unnecessary for the configured id.
    """
    want = configured.strip()
    if not want:
        return True
    if want in model_names:
        return True
    for n in model_names:
        if n.startswith(want + ":"):
            return True
        base = n.split(":", 1)[0]
        if base == want:
            return True
    return False


def log_llm_startup_warnings() -> None:
    """
    Log one-line warnings when ``LLM_PROVIDER`` is misconfigured (missing API keys or Ollama issues).

    Does not raise — intended for API startup so operators notice misconfiguration early.
    """
    p = llm_provider()
    if p == "anthropic" and not os.getenv("ANTHROPIC_API_KEY", "").strip():
        logger.warning("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is unset — LLM calls will fail.")
    if p == "openai" and not os.getenv("OPENAI_API_KEY", "").strip():
        logger.warning("LLM_PROVIDER=openai but OPENAI_API_KEY is unset — LLM calls will fail.")
    if p == "gemma" and not os.getenv("GOOGLE_API_KEY", "").strip():
        logger.warning("LLM_PROVIDER=gemma but GOOGLE_API_KEY is unset — LLM calls will fail.")
    if p == "ollama":
        try:
            from langchain_ollama import ChatOllama  # noqa: F401
        except ImportError:
            logger.warning(
                "LLM_PROVIDER=ollama but langchain-ollama is not installed — LLM calls will fail."
            )
        base = ollama_base_url()
        probe = list_local_ollama_models(timeout_sec=3.0)
        if not probe.get("ok"):
            err = probe.get("error") or "unknown error"
            logger.warning(
                "LLM_PROVIDER=ollama but Ollama is not reachable at %s — LLM calls will fail (%s).",
                base,
                err,
            )
        else:
            names = [str(m.get("name", "")).strip() for m in probe.get("models") or []]
            names = [n for n in names if n]
            configured = _resolved_llm_model()
            if names and not _ollama_configured_model_locally_installed(names, configured):
                logger.warning(
                    "LLM_PROVIDER=ollama: model %r was not found in `ollama list` at %s — "
                    "run `ollama pull %s` or set LLM_MODEL to an installed tag.",
                    configured,
                    base,
                    configured,
                )


__all__ = [
    "chunk_content_to_text",
    "get_embeddings",
    "get_llm",
    "get_vector_store",
    "invoke_llm_text",
    "llm_config_snapshot",
    "log_llm_startup_warnings",
]
