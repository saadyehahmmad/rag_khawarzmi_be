"""
Request-scoped context, structured logging, and optional OpenTelemetry tracing.

Goals:
- Correlate API, LangGraph node logs, and JSONL metrics with X-Request-ID and thread_id.
- Optional one-line JSON access logs and a dedicated metrics file for dashboards.
- Optional OTel distributed tracing when OTLP_ENDPOINT is set (graceful no-op otherwise).

OTel setup (optional):
  pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc
  OTLP_ENDPOINT=http://localhost:4317  # Jaeger / Grafana Tempo / OTEL Collector
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Optional OpenTelemetry import (graceful no-op when not installed) ─────────
try:
    from opentelemetry import trace as _otel_trace
    from opentelemetry.sdk.trace import TracerProvider as _TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor as _BatchSpanProcessor

    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False

_tracer: Any = None  # Set by otel_init() when OTLP_ENDPOINT is configured.

from agent.env_utils import env_bool

# Populated by HTTP middleware for the lifetime of one request.
request_id_cv: ContextVar[str | None] = ContextVar("request_id", default=None)
# Set in chat handlers after thread_id is resolved (body / session legacy).
thread_id_cv: ContextVar[str | None] = ContextVar("thread_id", default=None)

_METRICS_PATH = os.getenv("OBSERVABILITY_METRICS_LOG", "").strip()


class _RequestContextFilter(logging.Filter):
    """Attach request_id and thread_id from contextvars onto every LogRecord (safe for any formatter)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = request_id_cv.get() or "-"
        if not hasattr(record, "thread_id"):
            record.thread_id = thread_id_cv.get() or "-"
        return True


class _SafeTextFormatter(logging.Formatter):
    """
    Plain-text formatter that always injects request_id / thread_id from contextvars.

    Python's logging propagation sends records from child loggers directly to parent
    *handlers*, bypassing parent *logger*-level filters.  Injecting the fields inside
    the formatter makes it safe regardless of propagation path.
    """

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "request_id"):
            record.request_id = request_id_cv.get() or "-"
        if not hasattr(record, "thread_id"):
            record.thread_id = thread_id_cv.get() or "-"
        return super().format(record)


class _JsonStdoutFormatter(logging.Formatter):
    """One JSON object per line for log aggregators (Loki, CloudWatch, Datadog agent, etc.)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
            "thread_id": getattr(record, "thread_id", "-"),
        }
        if record.exc_info and record.exc_info[0] is not None:
            payload["exc_type"] = record.exc_info[0].__name__
        return json.dumps(payload, ensure_ascii=False)


def otel_init() -> None:
    """
    Initialize OTel tracing when ``OTLP_ENDPOINT`` is set in the environment.

    Uses the OTLP gRPC exporter (default port 4317) compatible with Jaeger,
    Grafana Tempo, and the OpenTelemetry Collector.  Safe to call multiple times
    (idempotent).  No-op when ``opentelemetry-sdk`` is not installed.
    """
    global _tracer
    if not _HAS_OTEL or _tracer is not None:
        return
    endpoint = os.getenv("OTLP_ENDPOINT", "").strip()
    if not endpoint:
        return
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # noqa: PLC0415
            OTLPSpanExporter,
        )

        provider = _TracerProvider()
        provider.add_span_processor(_BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        _otel_trace.set_tracer_provider(provider)
        _tracer = _otel_trace.get_tracer("rag")
        logging.getLogger(__name__).info("OTel tracing enabled → %s", endpoint)
    except Exception as exc:  # noqa: BLE001 — never crash startup on OTel failure
        logging.getLogger(__name__).warning("OTel init failed (%s); tracing disabled.", exc)


@contextmanager
def start_span(name: str, **attrs: Any) -> Iterator[Any]:
    """
    Context manager that wraps a named OTel span with optional string attributes.

    Falls back to a ``nullcontext`` no-op when OTel is not configured, so callers
    never need to guard with ``if _tracer``.

    Args:
        name: Span name (e.g. ``"rag.node.retrieval"``).
        **attrs: Arbitrary string/int/float attributes attached to the span.

    Yields:
        The active OTel ``Span`` object, or ``None`` when tracing is disabled.
    """
    if _tracer is None:
        with nullcontext() as ctx:
            yield ctx
        return
    with _tracer.start_as_current_span(name) as span:
        for k, v in attrs.items():
            span.set_attribute(k, str(v))
        yield span


def setup_observability() -> None:
    """
    Configure root logging: context filter, LOG_LEVEL, optional JSON stderr format.

    Safe to call once at API startup (lifespan). Idempotent enough for reload in dev.
    """
    root = logging.getLogger()
    if getattr(root, "_rag_obs_configured", False):
        return
    setattr(root, "_rag_obs_configured", True)

    filt = _RequestContextFilter()
    root.addFilter(filt)

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    root.setLevel(getattr(logging, level_name, logging.INFO))

    if env_bool("OBSERVABILITY_JSON_STDOUT", False):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JsonStdoutFormatter())
        for h in list(root.handlers):
            if isinstance(h, logging.StreamHandler) and h.stream in (sys.stdout, sys.stderr):
                root.removeHandler(h)
        root.addHandler(handler)
        return

    # Default: keep plain stream handlers but add request_id / thread_id columns when possible.
    text_fmt = _SafeTextFormatter(
        "%(asctime)s %(levelname)s [rid=%(request_id)s tid=%(thread_id)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for h in root.handlers:
        if isinstance(h, logging.StreamHandler) and h.stream in (sys.stdout, sys.stderr):
            h.setFormatter(text_fmt)


def append_metrics_event(event: dict[str, Any]) -> None:
    """
    Append one JSON line to OBSERVABILITY_METRICS_LOG when that env path is set.

    Intended for RAG completion summaries (not full prompts). Omit or redact PII in callers.
    """
    if not _METRICS_PATH:
        return
    try:
        path = Path(_METRICS_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = dict(event)
        line.setdefault("ts", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    except OSError as exc:
        logging.getLogger(__name__).warning("metrics log write failed: %s", exc)


@contextmanager
def thread_scope(thread_id: str) -> Iterator[None]:
    """Bind thread_id for LangGraph / nested logs for this block only."""
    tok: Token[str | None] = thread_id_cv.set(thread_id)
    try:
        yield
    finally:
        thread_id_cv.reset(tok)


def log_access_json(
    *,
    request_id: str,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
) -> None:
    """Structured HTTP access line (always JSON for easy grep / ingestion)."""
    logging.getLogger("rag.access").info(
        "%s",
        json.dumps(
            {
                "event": "http_access",
                "request_id": request_id,
                "method": method,
                "path": path,
                "status_code": status_code,
                "duration_ms": round(duration_ms, 2),
            },
            ensure_ascii=False,
        ),
    )


def log_rag_summary(
    *,
    request_id: str,
    thread_id: str,
    route: str,
    duration_ms: float,
    system: str | None,
    relevance: str | None,
    fallback: bool,
    question_chars: int,
    answer_chars: int,
    error: str | None = None,
) -> None:
    """Single high-signal line for search and metrics files."""
    sys = system if (system and str(system).strip()) else None
    rel = relevance if (relevance and str(relevance).strip()) else None
    payload = {
        "event": "rag_request",
        "request_id": request_id,
        "thread_id": thread_id,
        "route": route,
        "duration_ms": round(duration_ms, 2),
        "system": sys,
        "relevance": rel,
        "fallback": fallback,
        "question_chars": question_chars,
        "answer_chars": answer_chars,
    }
    if error:
        payload["error"] = error[:500]
    logging.getLogger("rag.rag").info("%s", json.dumps(payload, ensure_ascii=False))
    append_metrics_event(payload)
    # Feed Prometheus in-process counters (no-ops when prometheus_client not installed).
    _prom_record(
        route=route,
        system=system,
        relevance=relevance,
        fallback=fallback,
        duration_ms=duration_ms,
    )


# ─── Prometheus in-process metrics ───────────────────────────────────────────
# Lazily imported so the API still starts when prometheus_client is not installed
# (e.g. local dev without full requirements).  All four variables are None until
# the first call to prom_init().

_prom_requests: Any = None   # Counter  — rag_requests_total
_prom_fallbacks: Any = None  # Counter  — rag_fallbacks_total
_prom_duration: Any = None   # Histogram — rag_request_duration_seconds
_prom_ready: bool = False


def prom_init() -> bool:
    """Try to import prometheus_client and create metrics; return True on success."""
    global _prom_requests, _prom_fallbacks, _prom_duration, _prom_ready
    if _prom_ready:
        return True
    try:
        from prometheus_client import Counter, Histogram  # noqa: PLC0415

        _prom_requests = Counter(
            "rag_requests_total",
            "Total RAG requests processed",
            ["route", "system", "relevance"],
        )
        _prom_fallbacks = Counter(
            "rag_fallbacks_total",
            "RAG requests that returned a fallback response",
            ["route", "system"],
        )
        _prom_duration = Histogram(
            "rag_request_duration_seconds",
            "End-to-end RAG request duration in seconds",
            ["route"],
            buckets=[0.5, 1, 2, 5, 10, 20, 30, 60],
        )
        _prom_ready = True
    except ImportError:
        logging.getLogger(__name__).debug(
            "prometheus_client not installed — Prometheus metrics disabled."
        )
    return _prom_ready


def _prom_record(
    *,
    route: str,
    system: str | None,
    relevance: str | None,
    fallback: bool,
    duration_ms: float,
) -> None:
    """Increment Prometheus counters for one completed request (no-op if unavailable)."""
    if not _prom_ready and not prom_init():
        return
    sys_label = system or "none"
    rel_label = relevance or "unknown"
    try:
        _prom_requests.labels(route=route, system=sys_label, relevance=rel_label).inc()
        if fallback:
            _prom_fallbacks.labels(route=route, system=sys_label).inc()
        _prom_duration.labels(route=route).observe(duration_ms / 1000.0)
    except Exception:  # noqa: BLE001 — never crash request logging
        pass
