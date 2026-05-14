"""
Load survey layout chunks from Chroma (metadata filter, not embedding top-k).

Designer / embedded-survey path: full-catalog and overview-only ``get`` calls bypass hybrid
``top_k`` limits when the user asks for all questions/rules, a survey summary, or every layout doc.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _documents_from_chroma_get(raw: dict[str, Any]) -> list[Document]:
    """Build LangChain Documents from a Chroma ``get`` payload."""
    ids = raw.get("ids") or []
    texts = raw.get("documents") or []
    metas = raw.get("metadatas") or []
    out: list[Document] = []
    for i, _cid in enumerate(ids):
        content = texts[i] if i < len(texts) else ""
        meta = dict(metas[i]) if i < len(metas) and metas[i] else {}
        out.append(Document(page_content=content or "", metadata=meta))
    return out


def _page_sort_key(doc: Document) -> tuple[int, str]:
    """Sort page summaries: numeric page_id first, else lexicographic."""
    pid = doc.metadata.get("page_id", "")
    s = str(pid)
    try:
        return (0, f"{int(s):012d}")
    except (TypeError, ValueError):
        return (1, s)


def fetch_survey_overview_and_pages(store: Chroma, *, limit: int = 500) -> list[Document]:
    """
    Return every ``survey_overview`` and ``page_summary`` document for this collection.

    Args:
        store: Survey-scoped Chroma client.
        limit: Max rows from ``get`` (safety cap for very large forms).

    Returns:
        Overview first (if present), then page summaries ordered by ``page_id``.
        Empty list when the collection has no such metadata (legacy ingest) or on error.
    """
    try:
        raw = store.get(
            where={"chunk_type": {"$in": ["survey_overview", "page_summary"]}},
            limit=limit,
            include=["documents", "metadatas"],
        )
    except Exception as exc:  # noqa: BLE001 — Chroma / filter differences across versions
        logger.warning("fetch_survey_overview_and_pages: get() failed: %s", exc)
        return []

    docs = _documents_from_chroma_get(raw)
    if not docs:
        return []

    overview = [d for d in docs if d.metadata.get("chunk_type") == "survey_overview"]
    pages = [d for d in docs if d.metadata.get("chunk_type") == "page_summary"]
    pages_sorted = sorted(pages, key=_page_sort_key)
    ordered = overview[:1] + pages_sorted
    logger.info(
        "Survey layout catalog: overview=%s page_summaries=%s total_docs=%s",
        len(overview[:1]),
        len(pages_sorted),
        len(ordered),
    )
    return ordered


def fetch_survey_overview_only(store: Chroma, *, limit: int = 5) -> list[Document]:
    """
    Return only the ``survey_overview`` document(s) for this collection.

    Used when the user asks "what is this survey?" or requests a summary/description of the
    form — lighter than a full catalog fetch and puts the top-level survey description first.

    Args:
        store: Survey-scoped Chroma client.
        limit: Safety cap (normally 1 overview doc exists per survey).

    Returns:
        List of overview documents, or empty list on error / absent metadata.
    """
    try:
        raw = store.get(
            where={"chunk_type": "survey_overview"},
            limit=limit,
            include=["documents", "metadatas"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("fetch_survey_overview_only: get() failed: %s", exc)
        return []

    docs = _documents_from_chroma_get(raw)
    logger.info("Survey overview fetch: overview_chunks=%s", len(docs))
    return docs


def _rule_sort_key(doc: Document) -> tuple[int, str]:
    """Sort rule chunks: numeric ``rule_id`` first, else lexicographic."""
    rid = doc.metadata.get("rule_id", "")
    s = str(rid)
    try:
        return (0, f"{int(s):012d}")
    except (TypeError, ValueError):
        return (1, s)


def fetch_survey_all_rules(store: Chroma, *, limit: int = 500) -> list[Document]:
    """
    Return every ``rule`` document for this survey collection, ordered by ``rule_id``.

    Args:
        store: Survey-scoped Chroma client.
        limit: Max rows from ``get`` (safety cap for very large logic graphs).

    Returns:
        Sorted rule documents, or empty list on error / no metadata.
    """
    try:
        raw = store.get(
            where={"chunk_type": "rule"},
            limit=limit,
            include=["documents", "metadatas"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("fetch_survey_all_rules: get() failed: %s", exc)
        return []

    docs = _documents_from_chroma_get(raw)
    if not docs:
        return []

    ordered = sorted(docs, key=_rule_sort_key)
    logger.info("Survey rules catalog: rule_chunks=%s", len(ordered))
    return ordered
