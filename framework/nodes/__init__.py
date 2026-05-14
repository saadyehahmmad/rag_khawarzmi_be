"""
LangGraph node package: language, rewrite+route, retrieval, answer/fallback.

``PRE_ANSWER_PIPELINE`` is the single ordered list used by both LangGraph and SSE ``/chat``.

Related:
  ``core.llm_helpers`` — models and retry
  ``core.retrieval`` — hybrid search
  ``framework.nodes.answer_prompt`` — grounded answer prompt assembly
"""

from __future__ import annotations

from .answer_prompt import build_answer_prompt
from core.nodes.config import NodeFn
from .fallback_text import fallback_answer_text
from .pipeline import (
    PRE_ANSWER_PIPELINE,
    answer_node,
    configure_pipeline,
    fallback_node,
    language_detect_node,
    payload_context_node,
    retrieval_node,
    rewrite_and_route_node,
    route_by_relevance,
)
from core.nodes.query_log import append_query_log_entry
from .streaming import astream_answer_tokens, astream_fallback_tokens

__all__ = [
    "PRE_ANSWER_PIPELINE",
    "NodeFn",
    "answer_node",
    "append_query_log_entry",
    "astream_answer_tokens",
    "astream_fallback_tokens",
    "build_answer_prompt",
    "configure_pipeline",
    "fallback_answer_text",
    "fallback_node",
    "language_detect_node",
    "payload_context_node",
    "retrieval_node",
    "rewrite_and_route_node",
    "route_by_relevance",
]
