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
    # ── Client context (sent per-request by the FE) ──────────────────────────
    # Current Designer screen id from the Angular app (see designer package).
    # ``None`` / ``""`` means Dashboard (/) per ``vector_stores/designer/ROUTES.md``.
    page_id: NotRequired[Optional[str]]
    # Active survey ID — only present when page_id is a designer wizard page.
    survey_id: NotRequired[Optional[str]]
    # Displayed name for the user; both empty means unauthenticated session.
    user_name_en: NotRequired[str]
    user_name_ar: NotRequired[str]
    is_authenticated: NotRequired[bool]
    # UI language preference ("en" | "ar"); separate from question language detection.
    system_language: NotRequired[Optional[str]]
    # True while the wait-path applies: survey is embedding (paired with ``survey_ingesting``).
    survey_context_missing: NotRequired[bool]
    # True when the survey is currently being embedded (``session_store`` status ``ingesting``).
    survey_ingesting: NotRequired[bool]
    # True when at least one retrieved chunk came from ``vector_stores/surveys/<id>/`` (embedded survey index).
    survey_vector_context_used: NotRequired[bool]
    # True when ``survey_id`` was sent for Designer but no per-survey Chroma index exists yet (designer-manual retrieval only).
    survey_index_absent: NotRequired[bool]
    # When set, answer_node streams this text and skips the LLM (payload prescripts only).
    prescripted_answer: NotRequired[Optional[str]]
