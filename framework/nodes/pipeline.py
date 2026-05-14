"""
LangGraph node callables: language, payload prescripts, rewrite+route, survey, retrieval, answer.

Typical cost: 2 LLM calls per question (rewrite+route, answer). Retrieval has no LLM grader.

Profile injection
-----------------
Call ``configure_pipeline(profile)`` once at application startup (inside ``build_graph``),
which always runs before the graph is compiled in normal use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from alkawarzmi.payload_context import payload_context_step
from alkawarzmi.prompt_templates import REWRITE_AND_ROUTE_JSON_PROMPT
from core.llm_helpers import get_llm, invoke_llm_text
from core.output_safety import SAFETY_STRICT, check_answer
from core.governance import redact_prompt_injection_spans
from core.query_script_gate import UNSUPPORTED_SCRIPT_REPLY, question_uses_only_supported_scripts
from core.query_typo_normalize import normalize_query_typo
from core.text_ar import detect_language_for_rag, normalize_arabic_answer_text
from framework.state import AgentState
from framework.vector_health import SYSTEMS as VECTOR_SYSTEMS

from core.nodes.config import NodeFn, logger
from core.nodes.query_log import append_query_log_entry
from core.nodes.rewrite_parse import parse_rewrite_route_json

from .answer_prompt import build_answer_prompt, configure_answer_prompt, history_block
from .fallback_text import fallback_answer_text
from .intent import is_greeting
from .retrieval_step import configure_retrieval_hooks, retrieval_node

if TYPE_CHECKING:
    from framework.profile import RAGProfile

# ---------------------------------------------------------------------------
# Profile-driven module state (set once at startup via configure_pipeline)
# ---------------------------------------------------------------------------

_pipeline_rewrite_prompt: str = REWRITE_AND_ROUTE_JSON_PROMPT
_pipeline_systems: tuple[str, ...] = tuple(VECTOR_SYSTEMS)
_pipeline_prescripts_fn = payload_context_step
_pipeline_is_overview_fn = None  # set lazily below via property-like access
_pipeline_fallback_fn = fallback_answer_text


def _default_is_overview(question: str) -> bool:
    from .intent import is_platform_overview
    return is_platform_overview(question)


_pipeline_is_overview_fn = _default_is_overview


def configure_pipeline(profile: "RAGProfile") -> None:
    """
    Inject business-layer dependencies from *profile* into this module's nodes.

    Must be called once before the LangGraph is compiled (i.e. inside
    ``build_graph(profile)``). Thread-safe for read-after-write since FastAPI
    starts a single worker process before accepting requests.
    """
    global _pipeline_rewrite_prompt, _pipeline_systems
    global _pipeline_prescripts_fn, _pipeline_is_overview_fn, _pipeline_fallback_fn

    _pipeline_rewrite_prompt = profile.prompts.rewrite_and_route()
    _pipeline_systems = tuple(profile.systems)
    _pipeline_prescripts_fn = profile.prescripts.run
    _pipeline_is_overview_fn = profile.intent_detector.is_platform_overview
    _pipeline_fallback_fn = profile.fallback.get

    configure_answer_prompt(profile)
    configure_retrieval_hooks(profile.survey_retrieval)
    logger.info(
        "configure_pipeline: profile=%r systems=%s",
        profile.platform_name,
        _pipeline_systems,
    )


def _normalize_arabic_answer_if_needed(text: str, lang: str) -> str:
    """Apply light Arabic normalization for display/search when language is Arabic or mixed."""
    if lang in ("ar", "mixed"):
        return normalize_arabic_answer_text(text)
    return text


def query_script_gate_node(state: AgentState) -> dict[str, Any]:
    """Reject questions with characters outside English / Arabic scripts — no LLM."""
    if state.get("prescripted_answer"):
        return {}
    q = state.get("question") or ""
    if question_uses_only_supported_scripts(q):
        return {}
    logger.info("Node: query_script_gate — unsupported script, prescripted reply")
    return {"prescripted_answer": UNSUPPORTED_SCRIPT_REPLY}


def query_typo_normalize_node(state: AgentState) -> dict[str, Any]:
    """Fix obvious typos (EN distance-1, AR ي/ى map) before language_detect — no LLM."""
    if state.get("prescripted_answer"):
        return {}
    q = state.get("question") or ""
    fixed = normalize_query_typo(q)
    if fixed == q:
        return {}
    logger.info("Node: query_typo_normalize %r -> %r", q, fixed)
    return {"question": fixed}


def language_detect_node(state: AgentState) -> dict[str, Any]:
    """Detect Arabic / English / mixed from script ratios (no LLM)."""
    logger.info("Node: language_detect (heuristic)")
    language = detect_language_for_rag(state["question"])
    logger.info("  Detected language: %s", language)
    return {"language": language}


def payload_context_node(state: AgentState) -> dict[str, Any]:
    """
    Apply greeting-with-name, where-am-I, UI-language hints, and designer navigation
    hints from the client payload before routing (no LLM).
    """
    return _pipeline_prescripts_fn(state)


def rewrite_and_route_node(state: AgentState) -> dict[str, Any]:
    """Single LLM call: clarify query for search + pick the product collection."""
    if state.get("prescripted_answer"):
        logger.info("Node: rewrite_and_route — skipped (prescripted payload context)")
        return {}

    ui_system = (state.get("ui_system") or "").strip().lower()
    if ui_system and ui_system not in _pipeline_systems:
        ui_system = ""

    original_q = state.get("question", "")
    if is_greeting(original_q):
        logger.info("Node: rewrite_and_route — greeting detected, routing to none")
        return {"rewritten_question": original_q, "system": "none"}

    logger.info("Node: rewrite_and_route (1 LLM)")
    llm = get_llm()
    prompt = _pipeline_rewrite_prompt.format(
        question=redact_prompt_injection_spans(state["question"]),
        conversation_history=redact_prompt_injection_spans(history_block(state)),
    )
    raw = invoke_llm_text(llm, prompt)
    rewritten, system = parse_rewrite_route_json(raw, state["question"])
    lang = state.get("language", "en")
    rewritten = _normalize_arabic_answer_if_needed(rewritten, lang)
    if ui_system and system == "none":
        system = ui_system
        logger.info("  system=%s (default from ui_system) rewritten=%s", system, rewritten[:200])
    else:
        logger.info("  system=%s rewritten=%s", system, rewritten[:200])
    return {"rewritten_question": rewritten, "system": system}


def answer_node(state: AgentState) -> dict[str, Any]:
    """Generate the final grounded answer from retrieved chunks."""
    prescripted = state.get("prescripted_answer")
    if prescripted:
        logger.info("Node: answer (prescripted payload context)")
        ans = str(prescripted)
        append_query_log_entry(state, ans, fallback=False)
        return {"answer": ans}

    logger.info("Node: answer")
    llm = get_llm()
    answer = invoke_llm_text(llm, build_answer_prompt(state))
    lang = state.get("language", "en")
    answer = _normalize_arabic_answer_if_needed(answer, lang)

    safety = check_answer(
        answer,
        system=state.get("system"),
        chunks=state.get("retrieved_chunks") or [],
        request_id=state.get("request_id", ""),
    )
    if not safety.safe and SAFETY_STRICT:
        logger.warning("answer_node: safety check failed — substituting fallback (SAFETY_STRICT)")
        answer = _pipeline_fallback_fn(state)

    append_query_log_entry(state, answer, fallback=False)
    return {"answer": answer}


def fallback_node(state: AgentState) -> dict[str, Any]:
    """Return a safe bilingual fallback when retrieval is not relevant."""
    logger.info("Node: fallback")
    answer = _pipeline_fallback_fn(state)
    append_query_log_entry(state, answer, fallback=True)
    return {"answer": answer}


def route_by_relevance(state: AgentState) -> str:
    """Conditional edge function: returns 'answer' or 'fallback'."""
    if state.get("prescripted_answer"):
        return "answer"
    return "answer" if state.get("relevance") == "relevant" else "fallback"


PRE_ANSWER_PIPELINE: tuple[tuple[str, NodeFn], ...] = (
    ("query_script_gate", query_script_gate_node),
    ("query_typo_normalize", query_typo_normalize_node),
    ("language_detect", language_detect_node),
    ("payload_context", payload_context_node),
    ("rewrite_and_route", rewrite_and_route_node),
    ("retrieval", retrieval_node),
)
