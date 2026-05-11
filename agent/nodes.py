"""
LangGraph node functions: language (heuristic), rewrite+route (1 LLM), retrieval + distance
relevance, then answer or fallback (1 LLM).

Typical cost: 2 LLM calls per question (was 5: language, rewrite, route, relevance grader, answer).
PRE_ANSWER_PIPELINE is the single ordered list used by both LangGraph and the SSE /chat path.

Related modules:
  agent.llm_helpers      — model singletons and retry logic
  agent.image_selection  — semantic UI screenshot matching
  agent.output_safety    — post-LLM answer validation
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent.env_utils import env_bool
from agent.image_selection import select_images_for_question
from agent.llm_helpers import get_llm, get_vector_store, invoke_llm_text
from agent.output_safety import SAFETY_STRICT, check_answer
from agent.prompts import (
    ANSWER_PROMPT,
    FALLBACK_MESSAGE_AR,
    FALLBACK_MESSAGE_EN,
    GREETING_MESSAGE_AR,
    GREETING_MESSAGE_EN,
    PLATFORM_OVERVIEW_MESSAGE_AR,
    PLATFORM_OVERVIEW_MESSAGE_EN,
    REWRITE_AND_ROUTE_JSON_PROMPT,
)
from agent.retrieval import (
    RETRIEVAL_HYBRID,
    documents_to_chunks_and_refs,
    format_numbered_context,
    hybrid_retrieve,
)
from agent.state import AgentState
from agent.text_ar import detect_language_for_rag, normalize_arabic_answer_text, normalize_arabic_question
from agent.thread_memory import format_history_for_prompt
from agent.vector_health import SYSTEMS as VECTOR_SYSTEMS

load_dotenv()

logger = logging.getLogger(__name__)

LOG_PATH = Path(os.getenv("LOG_PATH", "./logs"))
LOG_PATH.mkdir(parents=True, exist_ok=True)

RETRIEVAL_TOP_K = max(1, int(os.getenv("RETRIEVAL_TOP_K", "8")))
RETRIEVAL_FETCH_K = max(
    RETRIEVAL_TOP_K,
    int(os.getenv("RETRIEVAL_FETCH_K", str(max(RETRIEVAL_TOP_K * 3, 12)))),
)
RETRIEVAL_MMR_LAMBDA = float(os.getenv("RETRIEVAL_MMR_LAMBDA", "0.5"))
RETRIEVAL_USE_MMR = env_bool("RETRIEVAL_USE_MMR", False)
RETRIEVAL_NORMALIZE_AR = env_bool("RETRIEVAL_NORMALIZE_AR", True)
# Dense candidate pool size for hybrid path (BM25 + RRF over this pool). Should be >= 3x RETRIEVAL_TOP_K.
HYBRID_DENSE_POOL = max(RETRIEVAL_TOP_K, int(os.getenv("HYBRID_DENSE_POOL", "24")))
_RELEVANCE_MAX_L2_RAW = os.getenv("RETRIEVAL_RELEVANCE_MAX_L2", "").strip()
# Minimum chunk count to consider retrieval relevant regardless of L2 distance.
_RELEVANCE_MIN_CHUNKS = max(1, int(os.getenv("RETRIEVAL_MIN_RELEVANT_CHUNKS", "3")))

# Pre-compiled regexes for JSON-fence stripping in _parse_rewrite_route_json.
_RE_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL | re.IGNORECASE)
_RE_FENCE_OPEN = re.compile(r"^```[^\n]*\n?")
_RE_FENCE_CLOSE = re.compile(r"\n?```\s*$")

# Type alias for node callables (used by PRE_ANSWER_PIPELINE and graph wiring).
NodeFn = Callable[[AgentState], dict[str, Any]]


def _parse_l2_cap(raw: str) -> float | None:
    """Parse RETRIEVAL_RELEVANCE_MAX_L2 once at startup; returns None to disable the gate."""
    if not raw:
        return None
    try:
        v = float(raw)
        return v if v > 0 else None
    except ValueError:
        logger.warning("Invalid RETRIEVAL_RELEVANCE_MAX_L2=%r; ignoring gate", raw)
        return None


_RELEVANCE_MAX_L2_CAP: float | None = _parse_l2_cap(_RELEVANCE_MAX_L2_RAW)


def _history_block(state: AgentState) -> str:
    """Prior turns formatted for prompts (empty thread yields placeholder string)."""
    return format_history_for_prompt(state.get("conversation_history") or [])


def _rq(state: AgentState) -> str:
    """Effective search query: rewritten text when present, else original question."""
    return state.get("rewritten_question") or state["question"]


# ── Greeting and platform-overview detection ──────────────────────────────────

_PLATFORM_NAMES = (
    "خوارزمي", "الخوارزمي", "al-khawarzmi", "al khawarzmi", "khawarzmi",
    "al-khwarzmi", "al khwarzmi", "khwarzmi",
)
_OVERVIEW_QUESTION_WORDS_AR = (
    "شو", "ما", "ايش", "إيش", "وش", "عرفني", "عرّفني", "احكيلي", "شرح",
    "شرحلي", "اخبرني", "أخبرني", "وضحلي", "وضّح", "فسرلي", "فسّر",
    "what", "tell me", "explain", "describe", "about",
)
_SYSTEM_KEYWORDS = (
    "مصمم", "designer", "builder", "بلدر",
    "ادمن", "admin", "field management", "ميداني",
    "callcenter", "call center", "كول سنتر", "مركز الاتصال",
    "runtime", "مشغل", "رن تايم",
)

_GREETING_PATTERNS_EN = frozenset({
    "hi", "hello", "hey", "howdy", "good morning", "good afternoon", "good evening",
    "good day", "greetings", "how are you", "how are you doing", "how do you do",
    "what's up", "whats up", "sup", "yo", "hiya",
})
_GREETING_PATTERNS_AR = frozenset({
    "مرحبا", "مرحباً", "السلام عليكم", "أهلا", "أهلاً", "اهلا", "اهلاً",
    "صباح الخير", "مساء الورد", "مساء الخير", "كيف حالك", "شخبارك", "مساء النور",
    "كيف الحال", "كيفك", "شلونك", "هاي", "هلا", "هلو", "سلام",
})


def _is_platform_overview(question: str) -> bool:
    """Return True when the user is asking what Al-Khawarzmi is (no specific system)."""
    q = question.strip().lower()
    if not any(name in q for name in _PLATFORM_NAMES):
        return False
    if any(kw in q for kw in _SYSTEM_KEYWORDS):
        return False
    words = set(re.split(r"[\s\W]+", q))
    return any(w in words for w in _OVERVIEW_QUESTION_WORDS_AR)


def _is_greeting(question: str) -> bool:
    """Return True when the question is a standalone greeting with no product intent."""
    q = question.strip().rstrip("?.!,،؟").strip().lower()
    if q in _GREETING_PATTERNS_EN or q in _GREETING_PATTERNS_AR:
        return True
    for pat in _GREETING_PATTERNS_EN:
        if q.startswith(pat) and len(q) <= len(pat) + 8:
            return True
    for pat in _GREETING_PATTERNS_AR:
        if q.startswith(pat) and len(q) <= len(pat) + 8:
            return True
    return False


def fallback_answer_text(state: AgentState) -> str:
    """Static bilingual message when retrieval is not usable (sync + SSE paths)."""
    lang = state.get("language", "en")
    original_q = state.get("question", "")
    if _is_greeting(original_q):
        return GREETING_MESSAGE_AR if lang == "ar" else GREETING_MESSAGE_EN
    if _is_platform_overview(original_q):
        return PLATFORM_OVERVIEW_MESSAGE_AR if lang == "ar" else PLATFORM_OVERVIEW_MESSAGE_EN
    return FALLBACK_MESSAGE_AR if lang == "ar" else FALLBACK_MESSAGE_EN


def _active_system_note(state: AgentState) -> str:
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
    return (
        f"The user is currently in the **{sys_key}** product area (retrieval collection). "
        "If they ask which system or module they are using, treat this as their context "
        "and answer accordingly, still grounded in the retrieved context below.\n\n"
    )


def build_answer_prompt(state: AgentState) -> str:
    """Shared prompt string for sync answer_node and SSE streaming (numbered context + citations)."""
    chunks = state.get("retrieved_chunks") or []
    refs = state.get("retrieved_source_refs") or []
    chunks_text = format_numbered_context(chunks, refs)
    # Always use the fully-resolved rewritten_question so the LLM answers the actual
    # topic and not a meta-instruction like "بالعربي" or "explain more".
    question_for_answer = state.get("rewritten_question") or state["question"]
    return ANSWER_PROMPT.format(
        language=state.get("language", "en"),
        active_system_note=_active_system_note(state),
        chunks=chunks_text,
        question=question_for_answer,
        conversation_history=_history_block(state),
    )


async def astream_answer_tokens(state: AgentState) -> AsyncIterator[str]:
    """Stream model token deltas for the grounded answer (SSE /chat)."""
    llm = get_llm()
    messages = [HumanMessage(content=build_answer_prompt(state))]
    lang = state.get("language", "en")
    # For Arabic outputs, apply deterministic typo fixes while streaming.
    # We normalize the growing buffer and emit only the newly-added suffix so
    # token boundaries cannot "hide" a substring replacement.
    raw_buf = ""
    emitted_len = 0

    async for chunk in llm.astream(messages):
        c = getattr(chunk, "content", None)
        parts: list[str] = []
        if isinstance(c, str) and c:
            parts = [c]
        elif isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and p.get("type") == "text":
                    t = str(p.get("text", ""))
                    if t:
                        parts.append(t)

        for part in parts:
            if not part:
                continue
            if lang in ("ar", "mixed"):
                raw_buf += part
                normalized = normalize_arabic_answer_text(raw_buf)
                delta = normalized[emitted_len:]
                if delta:
                    emitted_len = len(normalized)
                    yield delta
            else:
                yield part


async def astream_fallback_tokens(state: AgentState) -> AsyncIterator[str]:
    """Stream the fallback / greeting message line by line (SSE /chat)."""
    text = fallback_answer_text(state)
    for line in text.splitlines(keepends=True):
        if line.strip():
            yield line
            await asyncio.sleep(0.04)


# ── Relevance gate ────────────────────────────────────────────────────────────

def _relevance_from_dense_distance(
    chunks: list[str],
    best_distance: float | None,
) -> str:
    """
    Determine retrieval relevance using two complementary signals.

    Signal 1 — chunk count: if the hybrid pipeline returned at least
    ``_RELEVANCE_MIN_CHUNKS`` non-empty chunks, the corpus has enough signal
    to attempt an answer regardless of the top-1 L2 distance.

    Signal 2 — top-1 L2 distance (optional; only applied when chunk count is
    below the minimum threshold).  Disabled when RETRIEVAL_RELEVANCE_MAX_L2 is unset.
    """
    if not chunks:
        return "irrelevant"
    if len(chunks) >= _RELEVANCE_MIN_CHUNKS:
        return "relevant"
    if _RELEVANCE_MAX_L2_CAP is None:
        return "relevant"
    if best_distance is None:
        return "relevant"
    return "relevant" if best_distance <= _RELEVANCE_MAX_L2_CAP else "irrelevant"


# ── JSON parsing for rewrite+route ───────────────────────────────────────────

def _parse_rewrite_route_json(raw: str, fallback_question: str) -> tuple[str, str]:
    """Parse one JSON object from the model output; tolerant of markdown fences."""
    systems = ("designer", "runtime", "callcenter", "admin", "none")
    text = raw.strip()
    try:
        if "```" in text:
            m = _RE_JSON_FENCE.search(text)
            if m:
                text = m.group(1).strip()
            else:
                text = _RE_FENCE_OPEN.sub("", text)
                text = _RE_FENCE_CLOSE.sub("", text).strip()
        i = text.find("{")
        j = text.rfind("}")
        if i < 0 or j <= i:
            raise ValueError(f"no JSON object braces found in: {text[:80]!r}")
        obj = json.loads(text[i : j + 1])
        rw = str(obj.get("rewritten_question", "")).strip()
        sys = str(obj.get("system", "")).strip().lower()
        if sys not in systems:
            sys = (
                "none"
                if not rw or "unrelated" in rw.lower() or "unable to route" in rw.lower()
                else "designer"
            )
        if not rw:
            rw = fallback_question
        return rw, sys
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning("rewrite_and_route JSON parse failed (%s); using original question", exc)
        return fallback_question, "designer"


# ── Node functions ────────────────────────────────────────────────────────────

def language_detect_node(state: AgentState) -> dict[str, Any]:
    """Detect Arabic / English / mixed from script ratios (no LLM)."""
    logger.info("Node: language_detect (heuristic)")
    language = detect_language_for_rag(state["question"])
    logger.info("  Detected language: %s", language)
    return {"language": language}


def rewrite_and_route_node(state: AgentState) -> dict[str, Any]:
    """Single LLM call: clarify query for search + pick the product collection."""
    ui_system = (state.get("ui_system") or "").strip().lower()
    if ui_system and ui_system not in VECTOR_SYSTEMS:
        ui_system = ""

    # Pure greetings ("مرحبا", "hello", ...) carry no product intent. Short-circuit
    # to system="none" so the LLM cannot occasionally rewrite them into a platform
    # question and mis-route them to a specific system.
    original_q = state.get("question", "")
    if _is_greeting(original_q):
        logger.info("Node: rewrite_and_route — greeting detected, routing to none")
        return {"rewritten_question": original_q, "system": "none"}

    logger.info("Node: rewrite_and_route (1 LLM)")
    llm = get_llm()
    prompt = REWRITE_AND_ROUTE_JSON_PROMPT.format(
        question=state["question"],
        conversation_history=_history_block(state),
    )
    raw = invoke_llm_text(llm, prompt)
    rewritten, system = _parse_rewrite_route_json(raw, state["question"])
    # Apply minimal typo fixes for Arabic terms that the LLM occasionally misspells.
    # This is deterministic and does not affect retrieval embeddings (only the displayed text/query).
    lang = state.get("language", "en")
    if lang in ("ar", "mixed"):
        rewritten = normalize_arabic_answer_text(rewritten)
    # UI context is a DEFAULT, not a lock: if the user asks about another module
    # (e.g. callcenter) we should still retrieve from that module.
    # Only fall back to the UI system when routing gives "none".
    if ui_system and system == "none":
        system = ui_system
        logger.info("  system=%s (default from ui_system) rewritten=%s", system, rewritten[:200])
    else:
        logger.info("  system=%s rewritten=%s", system, rewritten[:200])
    return {"rewritten_question": rewritten, "system": system}


def retrieval_node(state: AgentState) -> dict[str, Any]:
    """Hybrid retrieve + top-1 dense distance for relevance gating (no LLM grader)."""
    system = state.get("system") or "designer"
    if system == "none":
        logger.info("Node: retrieval skipped — question routed as off-topic")
        return {
            "retrieved_chunks": [],
            "retrieved_source_refs": [],
            "retrieval_best_distance": None,
            "relevance": "irrelevant",
            "image_urls": [],
        }

    logger.info("Node: retrieval + relevance (%s)", system)
    store = get_vector_store(system)
    rq = _rq(state)
    lang = state.get("language", "en")
    if RETRIEVAL_NORMALIZE_AR and lang in ("ar", "mixed"):
        rq = normalize_arabic_question(rq)

    docs, best_d = hybrid_retrieve(
        store,
        rq,
        top_k=RETRIEVAL_TOP_K,
        dense_pool_size=HYBRID_DENSE_POOL,
        use_mmr_dense_pool=RETRIEVAL_USE_MMR,
        mmr_fetch_k=RETRIEVAL_FETCH_K,
        mmr_lambda=RETRIEVAL_MMR_LAMBDA,
    )
    chunks, refs = documents_to_chunks_and_refs(docs)
    relevance = _relevance_from_dense_distance(chunks, best_d)

    # Screen image selection should be driven by the retrieval query (user intent),
    # not by the generated answer. Use the same normalized query used for retrieval
    # to reduce accidental matches on UI labels like "Survey Language — اختر لغة الاستمارة".
    image_query = rq or (state.get("rewritten_question") or "")
    selected_fnames = select_images_for_question(image_query, system, language=lang)
    image_urls = [f"/images/{fname}" for fname in selected_fnames]

    logger.info(
        "  Retrieved %s chunks hybrid=%s best_l2=%s relevance=%s images=%s",
        len(chunks),
        RETRIEVAL_HYBRID,
        best_d,
        relevance,
        len(image_urls),
    )
    return {
        "retrieved_chunks": chunks,
        "retrieved_source_refs": refs,
        "retrieval_best_distance": best_d,
        "relevance": relevance,  # type: ignore[dict-item]
        "image_urls": image_urls,
    }


def answer_node(state: AgentState) -> dict[str, Any]:
    """Generate the final grounded answer from retrieved chunks."""
    logger.info("Node: answer")
    llm = get_llm()
    answer = invoke_llm_text(llm, build_answer_prompt(state))
    # Apply minimal typo fixes for Arabic terms in the final answer.
    lang = state.get("language", "en")
    if lang in ("ar", "mixed"):
        answer = normalize_arabic_answer_text(answer)

    # Post-LLM safety check — flags are logged; answer is substituted only in strict mode.
    safety = check_answer(
        answer,
        system=state.get("system"),
        chunks=state.get("retrieved_chunks") or [],
        request_id=state.get("request_id", ""),
    )
    if not safety.safe and SAFETY_STRICT:
        logger.warning("answer_node: safety check failed — substituting fallback (SAFETY_STRICT)")
        answer = fallback_answer_text(state)

    append_query_log_entry(state, answer, fallback=False)
    return {"answer": answer}


def fallback_node(state: AgentState) -> dict[str, Any]:
    """Return a safe bilingual fallback when retrieval is not relevant."""
    logger.info("Node: fallback")
    answer = fallback_answer_text(state)
    append_query_log_entry(state, answer, fallback=True)
    return {"answer": answer}


def route_by_relevance(state: AgentState) -> str:
    """Conditional edge function: returns 'answer' or 'fallback'."""
    return "answer" if state.get("relevance") == "relevant" else "fallback"


# ── Query logging ─────────────────────────────────────────────────────────────

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


def _detect_coverage_gap(answer: str) -> bool:
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
    question.  Mine these log lines to discover missing documentation topics.
    """
    refs = state.get("retrieved_source_refs") or []
    coverage_gap = fallback or _detect_coverage_gap(answer)
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


# Single source of truth for linear steps before answer/fallback (order matters).
# Both the LangGraph compilation in agent/graph.py and the SSE path in api/main.py
# iterate this tuple — keeping them in sync without code duplication.
PRE_ANSWER_PIPELINE: tuple[tuple[str, NodeFn], ...] = (
    ("language_detect", language_detect_node),
    ("rewrite_and_route", rewrite_and_route_node),
    ("retrieval", retrieval_node),
)
