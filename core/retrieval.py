"""
Hybrid retrieval (dense vector store + lexical BM25) with optional cross-encoder re-ranking.

Why this module exists:
- Pure embedding search misses exact tokens (SKUs, UI labels, Arabic phrasing variants).
- BM25 over a dense candidate pool adds lexical signal without indexing the full corpus at query time.
- Reciprocal Rank Fusion (RRF) merges dense and BM25 orderings without brittle score normalization.
- Optional CrossEncoder re-ranking improves precision on the short shortlist (disabled unless env set).

Tune via .env (see .env.example): RETRIEVAL_HYBRID, HYBRID_DENSE_POOL, RRF_K, RERANK_*.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from core.env_utils import env_bool, env_default_int
from core.text_ar import (
    apply_gulf_vocab_bridge,
    arabic_script_ratio,
    camel_get_lemmas,
    detect_language_for_rag,
    is_arabic_stopword,
)

logger = logging.getLogger(__name__)

# Pre-compiled regexes — reused across every _tokenize_for_bm25 call (hot path).
_RE_WS_SPLIT = re.compile(r"[\s\u200c\u200f]+")
_RE_STRIP_NON_WORD = re.compile(r"[^\w\u0600-\u06ff]+", re.UNICODE)

RETRIEVAL_HYBRID = env_bool("RETRIEVAL_HYBRID", True)
RRF_K = max(1, env_default_int("RRF_K", 60))
RERANK_MODEL = os.getenv("RERANK_CROSS_ENCODER_MODEL", "").strip()
RERANK_POOL = max(1, env_default_int("RERANK_POOL", 16))
RERANK_MAX_CHARS = max(256, env_default_int("RERANK_MAX_CHARS", 1024))
# When true, chunks whose metadata language tag doesn't match the query language are down-weighted.
# Set RETRIEVAL_LANG_FILTER=false to disable (e.g. cross-lingual retrieval experiments).
RETRIEVAL_LANG_FILTER = env_bool("RETRIEVAL_LANG_FILTER", True)

_cross_encoder: Any = None


def _tokenize_for_bm25(text: str) -> list[str]:
    """
    Tokenize Arabic/English text for BM25.

    Arabic token pipeline:
    1. Stopword removal  — pure function words discarded to sharpen BM25 precision.
    2. Gulf vocabulary bridge  — cross-dialect vocabulary mapped to MSA equivalents
       (e.g. ``وين`` → ``أين``) before lemmatization, because these are different
       lexemes that morphological analysis alone cannot bridge.
    3. CAMeL multi-lemma expansion  — all unique lemmas returned by Gulf then MSA
       analyzers are added as separate tokens, giving the query a wider match
       surface without noise (soft query expansion).

    Graceful degradation: if CAMeL data is not installed, ``camel_get_lemmas``
    returns the token unchanged and BM25 still operates normally.
    """
    if not text:
        return []
    lowered = text.lower()
    parts = _RE_WS_SPLIT.split(lowered)
    out: list[str] = []
    for p in parts:
        t = _RE_STRIP_NON_WORD.sub("", p)
        if len(t) < 2:
            continue
        if arabic_script_ratio(t) >= 0.5:
            if is_arabic_stopword(t):
                continue
            t = apply_gulf_vocab_bridge(t)
            out.extend(l for l in camel_get_lemmas(t) if len(l) >= 2)
        else:
            out.append(t)
    return out if out else ["empty"]


def _doc_id(d: Document) -> int:
    """Stable identity for ranking maps (object id is fine within one request)."""
    return id(d)


def _rrf_fuse_rankings(dense_order: list[Document], bm25_order: list[Document], *, k: int) -> list[Document]:
    """RRF over two ordered lists (same document universe; bm25_order is a reorder of the pool)."""
    scores: dict[int, float] = {}
    for r, d in enumerate(dense_order, start=1):
        did = _doc_id(d)
        scores[did] = scores.get(did, 0.0) + 1.0 / (k + r)
    for r, d in enumerate(bm25_order, start=1):
        did = _doc_id(d)
        scores[did] = scores.get(did, 0.0) + 1.0 / (k + r)
    # dense_order holds each document once; sort that list by fused RRF score.
    return sorted(dense_order, key=lambda d: scores.get(_doc_id(d), 0.0), reverse=True)


def _get_cross_encoder():
    """Lazy CrossEncoder (sentence-transformers); None when RERANK_CROSS_ENCODER_MODEL unset."""
    global _cross_encoder
    if not RERANK_MODEL:
        return None
    if _cross_encoder is False:
        return None
    if _cross_encoder is not None:
        return _cross_encoder
    try:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(RERANK_MODEL, trust_remote_code=True)
        logger.info("Cross-encoder reranker loaded: %s", RERANK_MODEL)
    except ImportError as exc:
        logger.warning("sentence-transformers not installed (%s); continuing without rerank.", exc)
        _cross_encoder = False
        return None
    except (OSError, RuntimeError) as exc:
        logger.warning("Cross-encoder load failed (%s); continuing without rerank.", exc)
        _cross_encoder = False
        return None
    return _cross_encoder


def _cross_encoder_rerank(query: str, docs: list[Document], top_k: int) -> list[Document]:
    model = _get_cross_encoder()
    if model is None or not docs:
        return docs[:top_k]
    pairs: list[list[str]] = []
    for d in docs:
        text = (d.page_content or "")[:RERANK_MAX_CHARS]
        pairs.append([query, text])
    try:
        raw_scores = model.predict(pairs, show_progress_bar=False)
    except (RuntimeError, ValueError, TypeError) as exc:
        logger.warning("CrossEncoder.predict failed (%s); using pre-rerank order.", exc)
        return docs[:top_k]
    scored = sorted(
        zip(docs, (float(s) for s in raw_scores)),
        key=lambda x: x[1],
        reverse=True,
    )
    return [d for d, _ in scored[:top_k]]


def hybrid_retrieve(
    store: Any,
    rq: str,
    *,
    top_k: int,
    dense_pool_size: int,
    use_mmr_dense_pool: bool,
    mmr_fetch_k: int,
    mmr_lambda: float,
) -> tuple[list[Document], float | None]:
    """
    Return (docs, best_l2_distance): up to top_k Documents plus the top-1 dense L2 score.

    Merging the distance measurement into this call avoids a second round-trip to the vector store.

    Hybrid path:
    1) Dense candidate pool via similarity_search_with_score (or MMR fallback).
    2) BM25 scores within that pool; RRF merges dense and BM25 rankings.
    3) Optional cross-encoder on the top RERANK_POOL fused docs -> final top_k.
    """
    pool = max(top_k, dense_pool_size)
    best_d: float | None = None

    if not RETRIEVAL_HYBRID:
        if use_mmr_dense_pool:
            try:
                raw = store.max_marginal_relevance_search(
                    rq, k=top_k, fetch_k=max(mmr_fetch_k, top_k), lambda_mult=mmr_lambda
                )
                return _deduplicate_docs(raw), None
            except (RuntimeError, ValueError) as exc:
                logger.warning("MMR retrieval failed (%s); using similarity_search", exc)
        scored = store.similarity_search_with_score(rq, k=top_k)
        if scored:
            best_d = float(scored[0][1])
        return _deduplicate_docs([d for d, _ in scored]), best_d

    dense_pool: list[Document]
    if use_mmr_dense_pool:
        try:
            dense_pool = store.max_marginal_relevance_search(
                rq,
                k=pool,
                fetch_k=max(mmr_fetch_k, pool),
                lambda_mult=mmr_lambda,
            )
        except (RuntimeError, ValueError) as exc:
            logger.warning("MMR pool failed (%s); similarity_search for hybrid pool", exc)
            scored = store.similarity_search_with_score(rq, k=pool)
            dense_pool = [d for d, _ in scored]
            if scored:
                best_d = float(scored[0][1])
    else:
        scored = store.similarity_search_with_score(rq, k=pool)
        dense_pool = [d for d, _ in scored]
        if scored:
            best_d = float(scored[0][1])

    if not dense_pool:
        return [], None

    # Language-aware filtering: prefer chunks whose language tag matches the query.
    # Chunks without a language tag (legacy ingest) are always included.
    if RETRIEVAL_LANG_FILTER:
        _ql = detect_language_for_rag(rq)
        q_lang = _ql if _ql != "mixed" else None
        if q_lang is not None:
            matched = [d for d in dense_pool if d.metadata.get("language", q_lang) == q_lang]
            # Only apply filter when it leaves at least top_k candidates.
            if len(matched) >= top_k:
                dense_pool = matched

    corpus_tokens = [_tokenize_for_bm25(d.page_content or "") for d in dense_pool]
    bm25 = BM25Okapi(corpus_tokens)
    q_tokens = _tokenize_for_bm25(rq)
    bm25_scores = bm25.get_scores(q_tokens)
    order_idx = sorted(range(len(dense_pool)), key=lambda i: bm25_scores[i], reverse=True)
    bm25_order = [dense_pool[i] for i in order_idx]

    fused = _rrf_fuse_rankings(dense_pool, bm25_order, k=RRF_K)
    shortlist = fused[: max(top_k, RERANK_POOL)]

    if RERANK_MODEL:
        reranked = _cross_encoder_rerank(rq, shortlist, top_k)
        return _deduplicate_docs(reranked), best_d

    return _deduplicate_docs(fused[:top_k]), best_d


def _deduplicate_docs(docs: list[Document]) -> list[Document]:
    """
    Remove near-duplicate chunks that share the same source file and first-200-char prefix.

    Overlapping splitter windows frequently produce chunks whose leading text is identical;
    keeping both wastes context window tokens and inflates the same fact's apparent importance.
    The first occurrence (highest-ranked) is kept; subsequent duplicates are dropped.
    """
    seen: set[tuple[str, str]] = set()
    result: list[Document] = []
    for doc in docs:
        key = (
            doc.metadata.get("source_file", ""),
            (doc.page_content or "")[:200].strip(),
        )
        if key not in seen:
            seen.add(key)
            result.append(doc)
    return result




def documents_to_chunks_and_refs(docs: list[Document]) -> tuple[list[str], list[dict[str, Any]]]:
    """
    Split LangChain Documents into plain chunk strings and parallel citation metadata.

    Each ref includes a 1-based `ref` index matching the [n] labels in the answer prompt.
    section_path and system_label (added by enrich_with_section_context) are preserved
    so format_numbered_context can re-inject heading context on every chunk.
    """
    chunks: list[str] = []
    refs: list[dict[str, Any]] = []
    for d in docs:
        text = d.page_content or ""
        if not text.strip():
            continue
        meta = d.metadata or {}
        chunks.append(text)
        # Parse image_refs from Chroma metadata (stored as JSON string by ingestion).
        raw_image_refs = meta.get("image_refs", "")
        try:
            image_refs: list[str] = json.loads(raw_image_refs) if raw_image_refs else []
        except (json.JSONDecodeError, TypeError):
            image_refs = []
        refs.append(
            {
                "ref": 0,
                "source_file": str(meta.get("source_file", "")),
                "source_rel": str(meta.get("source_rel", "")),
                "system": str(meta.get("system", "")),
                "system_label": str(meta.get("system_label", "")),
                "section_path": str(meta.get("section_path", "")),
                "ingest_source": str(meta.get("ingest_source", "")),
                "image_refs": image_refs,
            }
        )
    for j, r in enumerate(refs, start=1):
        r["ref"] = j
    return chunks, refs


def format_numbered_context(chunks: list[str], refs: list[dict[str, Any]]) -> str:
    """
    Build the context block shown to the answer model.

    Each chunk is prefixed with a Markdown-formatted reference header that
    carries the source system and section breadcrumb so the LLM can cite them
    cleanly.  Headers use bold + › arrow to distinguish them from body text:

        **[1] System Label** › Section Path

    Legacy ``[System | section]`` headers already embedded inside the chunk
    text are stripped to avoid duplication.
    """
    parts: list[str] = []
    for text, ref in zip(chunks, refs):
        if not text.strip():
            continue
        idx: int = ref.get("ref", 0)
        section_path: str = ref.get("section_path", "")
        system_label: str = ref.get("system_label", "")

        # Build clean Markdown header.
        if system_label and section_path:
            header = f"**[{idx}] {system_label}** › {section_path}"
        elif system_label:
            header = f"**[{idx}] {system_label}**"
        else:
            header = f"**[{idx}]**"

        # Strip legacy "[System | section]" header already embedded in chunk.
        clean = text.lstrip()
        if clean.startswith("[") and "|" in clean[:80] and "]" in clean[:80]:
            clean = clean[clean.find("]") + 1:].lstrip()

        parts.append(f"{header}\n\n{clean}")
    return "\n\n---\n\n".join(parts)
