"""
Chat routes: synchronous and SSE streaming.

Exposes:
  POST /chat/sync  — run the full graph once and return the final state
  POST /chat       — SSE stream: node progress events then real LLM tokens
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Optional, cast

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sse_starlette.sse import EventSourceResponse

from core.governance import (
    enforce_rate_limit,
    evaluate_question,
    log_blocked_attempt,
    refusal_message_for_outcome,
)
from framework.nodes import (
    PRE_ANSWER_PIPELINE,
    append_query_log_entry,
    astream_answer_tokens,
    astream_fallback_tokens,
    route_by_relevance,
)
from core.observability import log_rag_summary, start_span, thread_scope
from framework.state import AgentState
from core.text_ar import normalize_arabic_answer_text
from core.thread_memory import append_turn, load_conversation, resolve_thread_id
from framework.vector_health import SYSTEMS
from api.deps import (
    ChatRequest,
    dumps,
    get_compiled_graph,
    get_request_id,
    raise_http_for_governance,
    verify_rag_api_key,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_api_system(raw: Optional[str]) -> Optional[str]:
    """
    Map the optional ``system`` query parameter to a configured Chroma collection name.

    Unknown values are ignored (routing LLM will choose the system). Valid names
    follow ``RAG_SYSTEMS`` / :data:`framework.vector_health.SYSTEMS`.

    Args:
        raw: Raw query string from the client (e.g. ``"designer"``).

    Returns:
        Lowercase collection slug, or ``None`` if missing or invalid.
    """
    if raw is None:
        return None
    s = raw.strip().lower()
    if not s:
        return None
    if s not in SYSTEMS:
        logger.warning(
            "Ignoring unknown system query param %r (expected one of %s)",
            raw,
            ",".join(SYSTEMS),
        )
        return None
    return s


def _build_initial_state(
    request: ChatRequest,
    request_id: str,
    question: str,
    *,
    thread_id: str,
    api_system: Optional[str] = None,
) -> AgentState:
    """Load thread memory and build the LangGraph input state."""
    history = load_conversation(thread_id)

    user_name = request.user_name
    user_name_en = (user_name.en or "") if user_name else ""
    user_name_ar = (user_name.ar or "") if user_name else ""
    is_authenticated = bool(user_name_en or user_name_ar)

    return {
        "question": question,
        "language": "en",
        "rewritten_question": "",
        "ui_system": api_system,
        "system": None,
        "retrieved_chunks": [],
        "retrieved_source_refs": [],
        "relevance": "unknown",
        "answer": "",
        "thread_id": thread_id,
        "conversation_history": history,
        "request_id": request_id,
        # Client context
        "page_id": request.page_id or None,
        "survey_id": request.survey_id or None,
        "user_name_en": user_name_en,
        "user_name_ar": user_name_ar,
        "is_authenticated": is_authenticated,
        "system_language": request.system_language or None,
        "survey_context_missing": False,
        "survey_ingesting": False,
        "survey_index_absent": False,
        "survey_vector_context_used": False,
    }


def _persist_thread_memory(request: ChatRequest, result: dict[str, Any]) -> None:
    """Append this user question and assistant answer for cumulative threads."""
    tid = result.get("thread_id")
    if not isinstance(tid, str) or not tid:
        return
    append_turn(tid, request.question.strip(), str(result.get("answer", "")))


# ---------------------------------------------------------------------------
# POST /chat/sync
# ---------------------------------------------------------------------------


@router.post("/chat/sync")
def chat_sync(
    request: ChatRequest,
    req: Request,
    system: Optional[str] = Query(
        None,
        description=(
            "Product area / Chroma collection (e.g. designer). "
            "When set, retrieval uses this system."
        ),
    ),
    _auth: None = Depends(verify_rag_api_key),
) -> dict[str, Any]:
    """Run the full graph once and return the final state (for tests and simple clients)."""
    enforce_rate_limit(req)
    rid = get_request_id(req)

    try:
        tid = resolve_thread_id(request.thread_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    gov = evaluate_question(request.question)
    raise_http_for_governance(gov, thread_id=tid, request_id=rid, question_raw=request.question.strip())

    initial = _build_initial_state(
        request,
        rid,
        gov.sanitized_question,
        thread_id=tid,
        api_system=_normalize_api_system(system),
    )

    t0 = time.perf_counter()
    qchars = len(request.question.strip())

    with thread_scope(tid):
        try:
            with start_span("rag.graph.invoke", request_id=rid, thread_id=tid):
                result = get_compiled_graph().invoke(initial)
        except Exception as exc:  # noqa: BLE001
            log_rag_summary(
                request_id=rid,
                thread_id=tid,
                route="POST /chat/sync",
                duration_ms=(time.perf_counter() - t0) * 1000,
                system=None,
                relevance=None,
                fallback=False,
                question_chars=qchars,
                answer_chars=0,
                error=str(exc),
            )
            raise HTTPException(
                status_code=500, detail="An internal error occurred. Please try again."
            ) from exc

        try:
            _persist_thread_memory(request, result)
        except OSError:
            pass

        ans = str(result.get("answer", ""))
        log_rag_summary(
            request_id=rid,
            thread_id=tid,
            route="POST /chat/sync",
            duration_ms=(time.perf_counter() - t0) * 1000,
            system=str(result.get("system") or ""),
            relevance=str(result.get("relevance") or ""),
            fallback=result.get("relevance") != "relevant",
            question_chars=qchars,
            answer_chars=len(ans),
        )

    return {
        "thread_id": result.get("thread_id"),
        "language": result.get("language"),
        "system": result.get("system"),
        "rewritten_question": result.get("rewritten_question"),
        "relevance": result.get("relevance"),
        "answer": ans,
        "images": result.get("image_urls") or [],
    }


# ---------------------------------------------------------------------------
# POST /chat  (SSE)
# ---------------------------------------------------------------------------


async def _sse_stream(
    request: ChatRequest,
    http: Request,
    api_system: Optional[str],
) -> AsyncIterator[dict[str, str]]:
    """
    Generate SSE events for a single chat turn.

    Emits ``node_start`` as each retrieval step finishes, then streams real
    LLM tokens for answers. The fallback path emits one token event with the
    full static message (same client shape as streaming).
    """
    try:
        enforce_rate_limit(http)
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, str) else dumps(exc.detail)
        yield {"event": "error", "data": dumps({"message": detail})}
        return

    rid = get_request_id(http)

    try:
        tid = resolve_thread_id(request.thread_id)
    except ValueError as exc:
        yield {"event": "error", "data": dumps({"message": str(exc)})}
        return

    qchars = len(request.question.strip())
    merged: dict[str, Any] = {}

    with thread_scope(tid):
        gov = evaluate_question(request.question)
        if not gov.allowed:
            log_blocked_attempt(
                reason_codes=gov.reason_codes,
                thread_id=tid,
                request_id=rid,
                question_full=request.question.strip(),
            )
            qraw = request.question.strip()
            payload: dict[str, Any] = {
                "code": "policy_violation",
                "reasons": list(gov.reason_codes),
                "thread_id": tid,
                "message": refusal_message_for_outcome(gov, question_text=qraw),
            }
            if "question_too_long" in gov.reason_codes:
                payload["code"] = "question_too_long"
            yield {"event": "blocked", "data": dumps(payload)}
            yield {"event": "done", "data": dumps({"thread_id": tid, "blocked": True})}
            return

        try:
            initial = _build_initial_state(
                request,
                rid,
                gov.sanitized_question,
                thread_id=tid,
                api_system=api_system,
            )
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else dumps(exc.detail)
            yield {"event": "error", "data": dumps({"message": detail})}
            return

        thread_id = str(initial.get("thread_id", ""))
        merged = dict(initial)
        t0 = time.perf_counter()

        try:
            for name, fn in PRE_ANSWER_PIPELINE:
                yield {"event": "node_start", "data": dumps({"node": name})}
                with start_span(f"rag.node.{name}", request_id=rid, thread_id=tid):
                    partial = await asyncio.to_thread(fn, cast(AgentState, merged))
                merged.update(partial)

            branch = route_by_relevance(cast(AgentState, merged))
            last_node = "answer" if branch == "answer" else "fallback"
            yield {"event": "node_start", "data": dumps({"node": last_node})}

            answer_parts: list[str] = []
            stream_fn = astream_answer_tokens if branch == "answer" else astream_fallback_tokens
            async for token in stream_fn(cast(AgentState, merged)):
                answer_parts.append(token)
                yield {"event": "token", "data": token}

            full_answer = normalize_arabic_answer_text("".join(answer_parts))
            merged["answer"] = full_answer
            append_query_log_entry(cast(AgentState, merged), full_answer, fallback=(branch != "answer"))

            try:
                _persist_thread_memory(request, merged)
            except OSError:
                pass

            log_rag_summary(
                request_id=rid,
                thread_id=tid,
                route="POST /chat",
                duration_ms=(time.perf_counter() - t0) * 1000,
                system=str(merged.get("system") or ""),
                relevance=str(merged.get("relevance") or ""),
                fallback=branch != "answer",
                question_chars=qchars,
                answer_chars=len(full_answer),
            )
            yield {
                "event": "done",
                "data": dumps(
                    {
                        "thread_id": merged.get("thread_id") or thread_id,
                        "images": merged.get("image_urls") or [],
                    }
                ),
            }
        except Exception as exc:  # noqa: BLE001
            log_rag_summary(
                request_id=rid,
                thread_id=tid,
                route="POST /chat",
                duration_ms=(time.perf_counter() - t0) * 1000,
                system=str(merged.get("system") or "") or None,
                relevance=str(merged.get("relevance") or "") or None,
                fallback=False,
                question_chars=qchars,
                answer_chars=len(str(merged.get("answer", ""))),
                error=str(exc),
            )
            yield {"event": "error", "data": dumps({"message": str(exc)})}


@router.post("/chat")
async def chat(
    request: ChatRequest,
    req: Request,
    system: Optional[str] = Query(
        None,
        description=(
            "Product area / Chroma collection (e.g. designer). "
            "When set, retrieval uses this system."
        ),
    ),
    _auth: None = Depends(verify_rag_api_key),
) -> EventSourceResponse:
    """SSE endpoint compatible with the Angular EventSource client."""
    return EventSourceResponse(_sse_stream(request, req, _normalize_api_system(system)))
