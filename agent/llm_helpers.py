"""
Lazy singletons and helpers for LLM and embedding model access.

Separates infrastructure concerns (model loading, retry logic) from the pipeline
node logic in agent/nodes.py.  All callers import from here rather than
instantiating models directly.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage

from agent.env_utils import EMBEDDING_MODEL
from agent.paths import vector_store_root

logger = logging.getLogger(__name__)

VECTOR_STORE_PATH = vector_store_root()
LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
LLM_MAX_TOKENS = max(512, int(os.getenv("LLM_MAX_TOKENS", "4096")))
LLM_MAX_RETRIES = max(1, int(os.getenv("LLM_MAX_RETRIES", "3")))
LLM_RETRY_BASE_SEC = float(os.getenv("LLM_RETRY_BASE_SEC", "0.6"))

_llm: ChatAnthropic | None = None
_embeddings: HuggingFaceEmbeddings | None = None
_vector_stores: dict[str, Chroma] = {}


def get_llm() -> ChatAnthropic:
    """Lazy singleton for the Anthropic chat model."""
    global _llm
    if _llm is None:
        _llm = ChatAnthropic(model=LLM_MODEL, max_tokens=LLM_MAX_TOKENS)
    return _llm


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Lazy singleton for the local HuggingFace embedding model (CPU, L2-normalised).

    ``local_files_only=True`` prevents accidental re-downloads in production.
    ``normalize_embeddings=True`` is required for dot-product == cosine similarity.
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_vector_store(system: str) -> Chroma:
    """Return a cached Chroma client for the given system collection."""
    global _vector_stores
    if system not in _vector_stores:
        store_path = str(VECTOR_STORE_PATH / system)
        _vector_stores[system] = Chroma(
            persist_directory=store_path,
            embedding_function=get_embeddings(),
            collection_name=system,
        )
    return _vector_stores[system]


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


def invoke_llm_text(llm: ChatAnthropic, prompt: str) -> str:
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
        except Exception as exc:  # noqa: BLE001 — LangChain wraps Anthropic errors
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
