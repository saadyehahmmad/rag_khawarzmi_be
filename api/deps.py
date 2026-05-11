"""
Shared FastAPI dependencies, models, and helpers used across all routers.

Provides:
- JSON serialization helper (_dumps)
- Compiled LangGraph singleton (get_compiled_graph)
- ChatRequest Pydantic model
- API-key authentication dependency (_verify_rag_api_key)
- Request-ID extraction helper (_request_id)
- Governance-to-HTTP mapping helper (_raise_http_for_governance)
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Optional

from fastapi import Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field

from agent.governance import (
    MAX_QUESTION_CHARS,
    GovernanceOutcome,
    log_blocked_attempt,
    refusal_message_for_outcome,
)
from agent.graph import build_graph

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled-graph singleton
# ---------------------------------------------------------------------------

_compiled_graph: Any = None


def get_compiled_graph() -> Any:
    """Return the singleton LangGraph app, building it on first call."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


# ---------------------------------------------------------------------------
# JSON helper
# ---------------------------------------------------------------------------


def dumps(data: Any) -> str:
    """Serialize *data* to JSON for SSE payloads (UTF-8, no ASCII escaping)."""
    return json.dumps(data, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Inbound chat payload."""

    question: str = Field(..., min_length=1)
    thread_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Authentication helpers
# ---------------------------------------------------------------------------


def _resolve_bearer_token(
    authorization: Optional[str],
    x_api_key: Optional[str],
    api_key: Optional[str] = None,
) -> Optional[str]:
    """Extract the first non-empty credential from Bearer header, X-API-Key, or query param."""
    if authorization and authorization.lower().startswith("bearer "):
        tok = authorization[7:].strip()
        if tok:
            return tok
    if x_api_key:
        tok = x_api_key.strip()
        if tok:
            return tok
    if api_key:
        tok = api_key.strip()
        if tok:
            return tok
    return None


def verify_rag_api_key(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    api_key: Optional[str] = Query(None),
) -> None:
    """
    Require a matching RAG_API_KEY credential on protected routes.

    Health stays unauthenticated so load balancers can probe /health.
    When RAG_API_KEY is not set, all requests are allowed through.
    """
    expected = os.getenv("RAG_API_KEY", "").strip()
    if not expected:
        return
    token = _resolve_bearer_token(authorization, x_api_key, api_key)
    if not token or token != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def resolve_bearer_for_ready(
    authorization: Optional[str],
    x_api_key: Optional[str],
) -> Optional[str]:
    """Thin wrapper used by the health-ready auth dependency."""
    return _resolve_bearer_token(authorization, x_api_key)


# ---------------------------------------------------------------------------
# Request-ID helper
# ---------------------------------------------------------------------------


def get_request_id(req: Request) -> str:
    """Return the request-scoped ID that the observability middleware assigned."""
    return getattr(req.state, "request_id", str(uuid.uuid4()))


# ---------------------------------------------------------------------------
# Governance helper
# ---------------------------------------------------------------------------


def raise_http_for_governance(
    outcome: GovernanceOutcome,
    *,
    thread_id: str,
    request_id: str,
    question_raw: str,
) -> None:
    """Map a blocked governance outcome to HTTP errors and write audit when configured."""
    if outcome.allowed:
        return
    log_blocked_attempt(
        reason_codes=outcome.reason_codes,
        thread_id=thread_id,
        request_id=request_id,
        question_full=question_raw,
    )
    detail: dict[str, Any] = {
        "code": "policy_violation",
        "reasons": list(outcome.reason_codes),
    }
    if "question_too_long" in outcome.reason_codes or "empty_question" in outcome.reason_codes:
        detail["message"] = (
            f"Question is empty or exceeds maximum length ({MAX_QUESTION_CHARS} characters)."
        )
        raise HTTPException(status_code=422, detail=detail)
    detail["message"] = refusal_message_for_outcome(outcome, question_text=question_raw)
    raise HTTPException(status_code=403, detail=detail)
