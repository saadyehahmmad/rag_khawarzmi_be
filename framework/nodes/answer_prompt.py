"""
Answer LLM prompt assembly: history, active system note, numbered chunk context.

Profile injection
-----------------
``configure_answer_prompt(profile)`` is called by ``configure_pipeline`` in ``pipeline.py``
and sets ``_answer_prompt_template``.  The default is the existing Al-Khawarzmi
``ANSWER_PROMPT`` constant so this module is fully functional without a profile.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from alkawarzmi.designer import describe_designer_location_for_prompt, wants_survey_overview
from alkawarzmi.prompt_templates import ANSWER_PROMPT
from core.client_locale import ui_reply_language
from core.governance import redact_prompt_injection_spans
from core.retrieval import format_numbered_context
from core.thread_memory import format_history_for_prompt
from framework.state import AgentState

if TYPE_CHECKING:
    from framework.profile import RAGProfile

# ---------------------------------------------------------------------------
# Profile-driven module state
# ---------------------------------------------------------------------------

_answer_prompt_template: str = ANSWER_PROMPT


def configure_answer_prompt(profile: "RAGProfile") -> None:
    """Set the answer prompt template from the active profile. Called by configure_pipeline."""
    global _answer_prompt_template
    _answer_prompt_template = profile.prompts.answer()


def history_block(state: AgentState) -> str:
    """Prior turns formatted for prompts (empty thread yields placeholder string)."""
    return format_history_for_prompt(state.get("conversation_history") or [])


def effective_retrieval_query(state: AgentState) -> str:
    """Effective search query: rewritten text when present, else original question."""
    return state.get("rewritten_question") or state["question"]


def active_system_note(state: AgentState) -> str:
    """
    Short instruction for the answer model when the product area is known.

    Improves answers to meta questions (e.g. "what system is this?") using the UI
    context (query param) or the routed retrieval collection.

    Returns:
        Empty string when no active area, otherwise a single paragraph for the prompt.
    """
    sys_key = (state.get("ui_system") or state.get("system") or "").strip().lower()
    if not sys_key or sys_key == "none":
        return ""
    base = (
        f"The user is currently in the **{sys_key}** product area (retrieval collection). "
        "If they ask which system or module they are using, treat this as their context "
        "and answer accordingly, still grounded in the retrieved context below.\n\n"
    )
    if sys_key == "designer":
        base += describe_designer_location_for_prompt(
            state.get("page_id"),
            lang=ui_reply_language(state),
        )
    return base


def survey_session_note(state: AgentState) -> str:
    """
    Short LLM hint when the client attached a survey_id (retrieval may include survey chunks).

    Returns:
        Empty string when no ``survey_id``. Otherwise a short paragraph: when survey-index
        chunks were merged into context (optionally extra guidance for overview/summary asks),
        when the per-survey index is absent, or when only Designer manuals matched.
    """
    sid = (state.get("survey_id") or "").strip()
    if not sid:
        return ""

    used = bool(state.get("survey_vector_context_used"))
    index_absent = bool(state.get("survey_index_absent"))

    if used:
        base = (
            f"The request includes **survey_id** `{sid}` and the numbered context includes at least one "
            "**embedded survey-index** passage (pages/questions/rules from the per-survey vector store). "
            "Use those passages as ground truth for this form.\n\n"
        )
        if wants_survey_overview(state):
            base += (
                "The user is asking for a **high-level overview or summary** of this survey — lead with "
                "the **survey_overview** / top-level description in context, then optional page themes if present.\n\n"
            )
        return base

    if index_absent:
        return (
            f"The request includes **survey_id** `{sid}`, but **no per-survey index is on the server yet** "
            "(nothing was ingested for this id). The numbered context is **Designer product manuals only**. "
            "Do **not** invent or guess this form's questions, pages, or rules. If the user needs form-specific "
            "detail, say briefly that they should save the survey from Designer so the backend can index it, "
            "then answer what the manuals *do* support.\n\n"
        )

    return (
        f"The request includes **survey_id** `{sid}`, but the retrieved passages may be **Designer manuals only** "
        "(no embedded survey-index lines in this turn). Do **not** imply the answer reflects an indexed copy of "
        "this form unless you cite an embedded survey chunk.\n\n"
    )


def build_answer_prompt(state: AgentState) -> str:
    """Shared prompt string for sync answer_node and SSE streaming (numbered context + citations)."""
    raw_chunks = state.get("retrieved_chunks") or []
    chunks = [redact_prompt_injection_spans(str(c)) for c in raw_chunks]
    refs = state.get("retrieved_source_refs") or []
    chunks_text = format_numbered_context(chunks, refs)
    q_ans = state.get("rewritten_question") or state["question"]
    question_for_answer = redact_prompt_injection_spans(str(q_ans))
    hist = redact_prompt_injection_spans(history_block(state))
    return _answer_prompt_template.format(
        language=state.get("language", "en"),
        active_system_note=active_system_note(state),
        survey_note=survey_session_note(state),
        chunks=chunks_text,
        question=question_for_answer,
        conversation_history=hist,
    )
