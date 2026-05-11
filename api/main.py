"""
FastAPI application factory.

Responsibilities:
- Lifespan: validate vector stores, warm the graph and embeddings, handle SIGTERM.
- Middleware: CORS, request-ID + JSON access logging.
- Static files: docs/images/ mounted at /images.
- Routers: health (+ metrics) and chat (sync + SSE) registered here.

Optional environment variables:
  RAG_API_KEY              — shared secret for chat routes (Bearer / X-API-Key / ?api_key=)
  RAG_REQUIRE_VECTOR_STORES — fail fast at startup when Chroma folders are missing
  ALLOWED_ORIGINS          — comma-separated CORS origins (default: http://localhost:4200)
  DOCS_PATH                — root of the docs tree (default: docs)
  APP_VERSION              — injected into /health response
  REDIS_URL / MEMORY_PATH  — thread-memory backends (see agent/thread_memory.py)
"""

from __future__ import annotations

import logging
import os
import signal
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from agent.env_utils import env_bool
from agent.llm_helpers import get_embeddings
from agent.observability import (
    log_access_json,
    otel_init,
    prom_init,
    request_id_cv,
    setup_observability,
)
from agent.vector_health import all_vector_stores_ready, describe_vector_stores
from api.deps import dumps, get_compiled_graph
from api.routers.chat import router as chat_router
from api.routers.health import router as health_router

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """
    Validate optional vector-store strictness, log readiness, warm the graph.

    Warming loads embeddings once so the first user request is not a cold start.
    Registers a SIGTERM handler so long-running inference is allowed to drain before exit.
    """
    setup_observability()
    prom_init()  # Register Prometheus metrics at startup (no-op if library absent).
    otel_init()  # Initialize OTel tracing when OTLP_ENDPOINT is set (no-op otherwise).

    def _handle_sigterm(*_: Any) -> None:
        logger.info("SIGTERM received — draining in-flight requests and shutting down")

    try:
        signal.signal(signal.SIGTERM, _handle_sigterm)
    except (OSError, ValueError):
        # SIGTERM not available on Windows or inside sub-threads; safe to skip.
        pass

    info = describe_vector_stores()
    strict = env_bool("RAG_REQUIRE_VECTOR_STORES", False)
    if strict and not all_vector_stores_ready():
        raise RuntimeError(
            "RAG_REQUIRE_VECTOR_STORES is enabled but one or more Chroma collections are missing. "
            f"Details: {dumps(info)}"
        )
    if not all_vector_stores_ready():
        logger.warning(
            "Vector store readiness check failed — run ingestion before production. %s",
            dumps(info),
        )

    get_compiled_graph()
    get_embeddings()  # Pre-warm embeddings to avoid cold-start cost on the first request.
    logger.info("API startup complete — ready to serve requests")
    yield
    logger.info("API shutdown complete")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Al-Khwarizmi AI Assistant", version="1.0.0", lifespan=_lifespan)

# CORS
_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:4200")
_allow = [o.strip() for o in _origins.split(",") if o.strip()]
# Never allow an empty origin list — fall back to localhost to prevent implicit wildcard.
if not _allow:
    _allow = ["http://localhost:4200"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files — screenshot images from docs/images/
_images_dir = Path(os.getenv("DOCS_PATH", "docs")) / "images"
_images_dir.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(_images_dir)), name="images")


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def request_observability_middleware(request: Request, call_next):
    """
    Assign X-Request-ID, bind logging contextvars, emit JSON access log with duration.

    Runs for every HTTP call (including /health) so operators get uniform traces.
    """
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = rid
    tok = request_id_cv.set(rid)
    t0 = time.perf_counter()
    status = 500
    response: Any = None
    try:
        response = await call_next(request)
        status = response.status_code
    finally:
        duration_ms = (time.perf_counter() - t0) * 1000
        log_access_json(
            request_id=rid,
            method=request.method,
            path=request.url.path,
            status_code=status,
            duration_ms=duration_ms,
        )
        request_id_cv.reset(tok)
    if response is not None:
        response.headers["X-Request-ID"] = rid
        # On Windows, idle keep-alive sockets accumulate non-paged pool memory
        # and eventually trigger WinError 10055 (WSAENOBUFS).  Force non-SSE
        # responses to close their TCP connection immediately so the kernel
        # recycles the socket buffer right away.
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" not in content_type:
            response.headers["Connection"] = "close"
    return response


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(health_router)
app.include_router(chat_router)
