"""
Shared LangGraph state schema for the agentic RAG pipeline.

Each node returns a partial update; LangGraph merges keys into this TypedDict.
"""

from typing import Any, Literal, NotRequired, Optional, TypedDict


class AgentState(TypedDict):
    """State passed between all nodes in the LangGraph graph."""

    question: str
    language: Literal["ar", "en", "mixed"]
    rewritten_question: str
    # UI / client-provided context (e.g. query param `?system=designer`).
    # This is used as a default hint for ambiguous follow-ups and for meta-questions
    # like "what system is this?", but should NOT force retrieval if the question
    # is clearly about another module.
    ui_system: NotRequired[Optional[str]]
    system: Optional[str]
    retrieved_chunks: list[str]
    # Parallel to retrieved_chunks after retrieval_node: citation targets for UI and eval.
    retrieved_source_refs: NotRequired[list[dict[str, Any]]]
    # Top-1 dense L2 distance from Chroma (lower is closer); set in retrieval_node for metrics / tuning.
    retrieval_best_distance: NotRequired[float | None]
    # Ordered image URLs to display alongside the answer (populated from screens.json metadata).
    image_urls: NotRequired[list[str]]
    relevance: Literal["relevant", "irrelevant", "unknown"]
    answer: str
    thread_id: str
    # Prior turns loaded from Redis or file (API) before invoke; not mutated by nodes.
    conversation_history: NotRequired[list[dict[str, str]]]
    # Correlation id from API middleware for logs and audit trails.
    request_id: NotRequired[str]
