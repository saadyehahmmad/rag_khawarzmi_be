"""
Append-only JSONL query logging for offline evaluation and coverage-gap mining.
"""

from __future__ import annotations

import json
from datetime import datetime

from framework.state import AgentState

from .config import LOG_PATH, logger

_COVERAGE_GAP_SIGNALS_EN = (
    "don't have enough information",
    "do not have enough information",
    "not covered in",
    "not available in",
    "context does not contain",
    "context doesn't contain",
    "cannot find",
    "no information",
    "not documented",
    "suggest contacting",
    "contact the realsoft",
    "contact realsoft",
)

_COVERAGE_GAP_SIGNALS_AR = (
    "لا تتوفر لديّ",
    "لا تتوفر لدي",
    "لا تتوفر معلومات",
    "غير موثق",
    "لا يوجد في",
    "لا تتوفر في",
    "التواصل مع فريق",
    "للتواصل مع",
)


def detect_coverage_gap(answer: str) -> bool:
    """Return True when the answer signals that the retrieved context was insufficient."""
    lower = answer.lower()
    for sig in _COVERAGE_GAP_SIGNALS_EN:
        if sig in lower:
            return True
    for sig in _COVERAGE_GAP_SIGNALS_AR:
        if sig in answer:
            return True
    return False


def append_query_log_entry(state: AgentState, answer: str, fallback: bool) -> None:
    """
    Append one JSON line per query for offline evaluation and documentation gap analysis.

    ``coverage_gap=True`` means the LLM signalled the docs didn't fully cover the
    question. Mine these log lines to discover missing documentation topics.
    """
    refs = state.get("retrieved_source_refs") or []
    coverage_gap = fallback or detect_coverage_gap(answer)
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "request_id": state.get("request_id", ""),
        "thread_id": state.get("thread_id", "unknown"),
        "question": state["question"],
        "language": state.get("language"),
        "system": state.get("system"),
        "rewritten_question": state.get("rewritten_question"),
        "chunks_count": len(state.get("retrieved_chunks") or []),
        "citations_count": len(refs),
        "citation_sources": [r.get("source_file", "") for r in refs[:12]],
        "relevance": state.get("relevance"),
        "retrieval_best_distance": state.get("retrieval_best_distance"),
        "fallback": fallback,
        "coverage_gap": coverage_gap,
        "answer_preview": answer[:500],
    }
    log_file = LOG_PATH / f"queries_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
    try:
        with open(log_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("Query log write failed (%s); skipping log entry.", exc)
