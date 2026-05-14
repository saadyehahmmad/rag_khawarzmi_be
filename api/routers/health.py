"""
Health and observability routes.

Exposes:
  GET /health        — basic liveness (unauthenticated)
  GET /health/ready  — readiness probe (optionally authenticated via RAG_HEALTH_READY_KEY)
  GET /metrics       — Prometheus metrics (unauthenticated)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import Response

from framework.vector_health import all_vector_stores_ready, describe_vector_stores
from api.deps import resolve_bearer_for_ready

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# /health — liveness
# ---------------------------------------------------------------------------


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness probe. Always responds 200 while the process is running."""
    body: dict[str, str] = {"status": "ok", "service": "Al-Khwarizmi AI Assistant"}
    ver = os.getenv("APP_VERSION", "").strip()
    if ver:
        body["version"] = ver
    return body


# ---------------------------------------------------------------------------
# /health/ready — readiness
# ---------------------------------------------------------------------------


def _verify_health_ready_key(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> None:
    """
    Optional authentication for the /health/ready endpoint.

    When RAG_HEALTH_READY_KEY is set callers must supply it via
    ``Authorization: Bearer <key>`` or ``X-API-Key`` header.
    Load-balancer probes that do not set this env var are unaffected.
    """
    expected = os.getenv("RAG_HEALTH_READY_KEY", "").strip()
    if not expected:
        return
    token = resolve_bearer_for_ready(authorization, x_api_key)
    if not token or token != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing health-ready API key")


@router.get("/health/ready")
def health_ready(_auth: None = Depends(_verify_health_ready_key)) -> dict[str, Any]:
    """
    Readiness probe: catalog vector stores exist and look populated (Chroma on disk or Qdrant).

    Returns 503 when not ready so orchestrators can keep the pod out of rotation.
    Optionally authenticated via RAG_HEALTH_READY_KEY.
    """
    body = describe_vector_stores()
    if not all_vector_stores_ready():
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "vector_stores": body},
        )
    return {"status": "ready", "vector_stores": body}


# ---------------------------------------------------------------------------
# /metrics — Prometheus
# ---------------------------------------------------------------------------


@router.get("/metrics")
def metrics() -> Response:
    """
    Prometheus metrics endpoint.

    Returns plain-text metrics in Prometheus exposition format.
    Responds 503 when ``prometheus_client`` is not installed (never raises
    so health checks stay clean).
    """
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest  # noqa: PLC0415

        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        return Response(
            content="# prometheus_client not installed\n",
            status_code=503,
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )
