"""
Catalog vector stores (designer, runtime, …): Chroma on disk or remote Qdrant.

Survey-scoped stores remain Chroma-only (see ``framework.survey_store``).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from core.env_utils import qdrant_api_key, qdrant_url, vector_backend
from core.paths import vector_store_root

logger = logging.getLogger(__name__)

_chroma_stores: dict[str, Any] = {}
_qdrant_stores: dict[str, Any] = {}
_qdrant_client: Any = None


def _get_qdrant_client() -> Any:
    """Lazy singleton Qdrant HTTP client (used by ingestion and health checks)."""
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient

        url = qdrant_url()
        api_key = qdrant_api_key()
        kwargs: dict[str, Any] = {"url": url}
        if api_key:
            kwargs["api_key"] = api_key
        prefer_grpc = os.getenv("QDRANT_PREFER_GRPC", "").strip().lower() in ("1", "true", "yes")
        kwargs["prefer_grpc"] = prefer_grpc
        _qdrant_client = QdrantClient(**kwargs)
    return _qdrant_client


def get_qdrant_client_public() -> Any:
    """
    Expose the shared Qdrant client for ingestion / health (same process singleton).

    Returns:
        ``qdrant_client.QdrantClient`` instance.
    """
    return _get_qdrant_client()


def get_vector_store(system: str) -> Any:
    """
    Return a cached LangChain vector store for the given catalog system name.

    Backend is selected by ``VECTOR_BACKEND`` (``chroma`` default, ``qdrant``;
    typo ``qudrant`` is accepted).

    Args:
        system: Collection name (e.g. ``designer``).

    Returns:
        ``Chroma`` or ``QdrantVectorStore`` with ``similarity_search_with_score`` /
        ``max_marginal_relevance_search`` compatible with ``hybrid_retrieve``.
    """
    backend = vector_backend()
    if backend == "chroma":
        return _get_chroma_store(system)
    return _get_qdrant_store(system)


def _get_chroma_store(system: str) -> Any:
    global _chroma_stores
    if system not in _chroma_stores:
        from langchain_chroma import Chroma

        from core.llm_helpers import get_embeddings

        store_path = str(vector_store_root() / system)
        _chroma_stores[system] = Chroma(
            persist_directory=store_path,
            embedding_function=get_embeddings(),
            collection_name=system,
        )
    return _chroma_stores[system]


def _get_qdrant_store(system: str) -> Any:
    global _qdrant_stores
    if system not in _qdrant_stores:
        try:
            from langchain_qdrant import QdrantVectorStore
        except ImportError as exc:  # pragma: no cover - env guard
            raise ImportError(
                "VECTOR_BACKEND=qdrant requires langchain-qdrant and qdrant-client "
                "(see requirements.txt)."
            ) from exc

        from core.llm_helpers import get_embeddings

        client = _get_qdrant_client()
        _qdrant_stores[system] = QdrantVectorStore(
            client=client,
            collection_name=system,
            embedding=get_embeddings(),
        )
        logger.debug("Opened QdrantVectorStore collection=%r", system)
    return _qdrant_stores[system]


def reset_vector_store_caches() -> None:
    """Clear in-process store singletons (tests / reload hooks)."""
    global _chroma_stores, _qdrant_stores, _qdrant_client
    _chroma_stores = {}
    _qdrant_stores = {}
    _qdrant_client = None
