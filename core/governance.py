"""
Input guardrails and lightweight governance for the RAG API.

Goals:
- Reject oversized payloads (DoS / cost control).
- Block obvious prompt-injection scaffolding (English + Arabic heuristics; conservative).
- Redact injection-shaped spans from **untrusted** text before it is placed in LLM prompts
  (retrieved chunks, thread history, survey-derived lines). This reduces but cannot
  mathematically eliminate all jailbreaks — combine with grounded prompts and output checks.
- Optional substring blocklist from the environment (compliance / local policy).
- Optional audit log of blocked attempts (no full question body by default — prefix + hash only).
- Optional per-client rate limit (Redis preferred; in-process fallback for single-worker dev).

Disabled with GOVERNANCE_ENABLED=false (length checks still apply when MAX > 0).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

from core.env_utils import env_bool, env_default_int
from core.text_ar import language_hint_from_text, normalize_arabic_question

load_dotenv()

logger = logging.getLogger(__name__)

GOVERNANCE_ENABLED = env_bool("GOVERNANCE_ENABLED", True)
try:
    MAX_QUESTION_CHARS = max(256, env_default_int("GOVERNANCE_MAX_QUESTION_CHARS", 12000))
except ValueError:
    msg = "Invalid GOVERNANCE_MAX_QUESTION_CHARS - using 12000"
    logger.warning(msg)
    print(f"[env] {msg}", flush=True)
    MAX_QUESTION_CHARS = max(256, 12000)
AUDIT_LOG_PATH = os.getenv("GOVERNANCE_AUDIT_LOG", "").strip()
# Default 60 req/min protects the API out of the box; set 0 to disable for local dev.
try:
    RATE_LIMIT_PER_MINUTE = max(0, env_default_int("GOVERNANCE_RATE_LIMIT_PER_MINUTE", 60))
except ValueError:
    msg = "Invalid GOVERNANCE_RATE_LIMIT_PER_MINUTE - using 60"
    logger.warning(msg)
    print(f"[env] {msg}", flush=True)
    RATE_LIMIT_PER_MINUTE = 60

# Comma-separated substrings (case-insensitive). Empty = none.
_BLOCK_SUBSTRINGS_RAW = os.getenv("GOVERNANCE_BLOCK_SUBSTRINGS", "").strip()
_BLOCK_FILE = os.getenv("GOVERNANCE_BLOCKLIST_FILE", "").strip()

# Compiled patterns: obvious instruction-override / exfil scaffolding (English).
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE | re.MULTILINE)
    for p in (
        r"\bignore\s+(all\s+)?(previous|prior|above)\s+instructions\b",
        r"\bdisregard\s+(the\s+)?(system|developer)\s+message\b",
        r"\byou\s+are\s+now\s+(DAN|evil|unrestricted)\b",
        r"\b(system\s+prompt|developer\s+message)\s*:",
        r"<\s*/?\s*system\s*>",
        r"\boverride\s+(safety|policy|rules)\b",
        r"\b(jailbreak|prompt\s+injection)\b",
        r"\b(print|reveal|dump)\s+(the\s+)?(full\s+)?(prompt|system\s+prompt)\b",
        r"\b(api[_-]?key|sk-ant-|authorization:\s*bearer)\b",
        r"\bforget\s+(everything|all)\s+(above|before|prior)\b",
        r"\bnew\s+(system\s+)?instructions\s*:",
        r"\b(end\s+of|start\s+of)\s+(system|user)\s+message\b",
        r"<\s*\|?\s*(imstart|system|/s)\s*\|?\s*>",
        r"\[\s*INST\s*\]",
        r"<<\s*SYS\s*>>",
        r"\b(roleplay|rp\s+mode)\s*:\s*you\s+are\b",
    )
)

# Arabic injection / override scaffolding (Unicode script; no IGNORECASE — Arabic is caseless).
_INJECTION_PATTERNS_AR: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.MULTILINE)
    for p in (
        # "Ignore previous / all instructions" variants common in Arabic prompts.
        r"(تجاهل|تجاهلي)\s+(كل\s+)?(التعليمات|التعليمات\s+السابقة|ما\s+سبق|ما\s+سبق\s+من)",
        r"(لا\s+تتبع|لا\s+تتبعي)\s+(التعليمات|القواعد)",
        r"(أنت|انت)\s+الآن\s+(DAN|مخترق|بدون\s+قيود)",
        r"(اعرض|أعرض|اذكر|اكشف)\s+(البرومبت|برومبت|رسالة\s+النظام|تعليمات\s+النظام)",
        r"(انسخ|أرسل)\s+(مفتاح|توكن|التوكن|api)",
        r"(تجاوز|تجاوزي)\s+(السلامة|السياسة|القواعد)",
        r"(انسخ|أرسل)\s+(الرسالة\s+السرية|محتوى\s+النظام)",
        r"(تجاهل|تجاهلي)\s+(هذه\s+الرسالة|الرسالة\s+السابقة)",
    )
)

# Stores (raw_casefolded, normalized_casefolded) pairs — built once, reused per request.
_blocklist_cache: Optional[list[tuple[str, str]]] = None
_rate_lock = threading.Lock()
_rate_memory: dict[tuple[str, int], int] = {}


@dataclass(frozen=True)
class GovernanceOutcome:
    """Result of evaluating a user question before any LLM calls."""

    allowed: bool
    reason_codes: tuple[str, ...]
    sanitized_question: str


def _load_block_substrings() -> list[tuple[str, str]]:
    """
    Merge env CSV and optional newline-separated file into pre-normalized pairs.

    Each entry is stored as (raw_casefolded, normalized_casefolded) so that
    ``evaluate_question`` can do two cheap ``in`` lookups per phrase without
    re-running ``normalize_arabic_question`` on every request.
    """
    global _blocklist_cache
    if _blocklist_cache is not None:
        return _blocklist_cache
    raw_phrases: list[str] = []
    if _BLOCK_SUBSTRINGS_RAW:
        for part in _BLOCK_SUBSTRINGS_RAW.split(","):
            p = part.strip()
            if p:
                raw_phrases.append(p)
    if _BLOCK_FILE:
        try:
            path = Path(_BLOCK_FILE)
            if path.is_file():
                for line in path.read_text(encoding="utf-8").splitlines():
                    s = line.strip()
                    if s and not s.startswith("#"):
                        raw_phrases.append(s)
        except OSError as exc:
            logger.warning("Could not read GOVERNANCE_BLOCKLIST_FILE: %s", exc)
    _blocklist_cache = [
        (p.casefold(), normalize_arabic_question(p).casefold()) for p in raw_phrases
    ]
    return _blocklist_cache


def sanitize_question(text: str) -> str:
    """Strip control characters that can break logs or downstream parsers."""
    # Remove NUL and other C0 controls except common whitespace.
    cleaned = "".join(ch for ch in text if ch == "\n" or ch == "\r" or ch == "\t" or ord(ch) >= 32)
    return cleaned.strip()


_REDACTION_PLACEHOLDER = " [redacted] "


def redact_prompt_injection_spans(text: str) -> str:
    """
    Best-effort removal of injection-shaped substrings before text is sent to an LLM.

    Applies Unicode NFKC, English + Arabic regex redaction (matched spans become
    `` [redacted] ``), then Arabic governance normalization. Use for retrieved chunks, survey-derived lines, and conversation
    history — not for replacing the stored API ``question`` field (that stays raw;
    redact only in prompt assembly paths).
    """
    if not text:
        return text
    t = sanitize_question(unicodedata.normalize("NFKC", text))
    for pat in _INJECTION_PATTERNS:
        t = pat.sub(_REDACTION_PLACEHOLDER, t)
    t_ar = normalize_arabic_question(t)
    for pat in _INJECTION_PATTERNS_AR:
        t_ar = pat.sub(_REDACTION_PLACEHOLDER, t_ar)
    return t_ar


def _question_hits_injection_patterns(text: str, normalized_scan: str) -> bool:
    """True when any heuristic injection pattern matches (raw + NFKC + Arabic-normalized)."""
    nfkc = unicodedata.normalize("NFKC", text)
    nfkc_norm = normalize_arabic_question(nfkc)
    for pat in _INJECTION_PATTERNS:
        if pat.search(text) or pat.search(nfkc):
            return True
    for pat in _INJECTION_PATTERNS_AR:
        if (
            pat.search(normalized_scan)
            or pat.search(text)
            or pat.search(nfkc)
            or pat.search(nfkc_norm)
        ):
            return True
    return False


def evaluate_question(raw: str) -> GovernanceOutcome:
    """
    Validate and lightly sanitize the user question.

    Returns allowed=False with stable reason_codes for HTTP 403 / audit when policy hits.
    """
    text = sanitize_question(raw)
    if not text:
        return GovernanceOutcome(False, ("empty_question",), "")

    if len(text) > MAX_QUESTION_CHARS:
        return GovernanceOutcome(False, ("question_too_long",), text[:MAX_QUESTION_CHARS])

    if not GOVERNANCE_ENABLED:
        return GovernanceOutcome(True, (), text)

    # Normalized copy catches Alef variants / Tashkeel tricks without changing stored question text.
    normalized_scan = normalize_arabic_question(text)
    lowered = text.casefold()
    lowered_norm = normalized_scan.casefold()

    for raw_cf, norm_cf in _load_block_substrings():
        if raw_cf in lowered or norm_cf in lowered_norm:
            return GovernanceOutcome(False, ("blocklist_match",), text)

    if _question_hits_injection_patterns(text, normalized_scan):
        return GovernanceOutcome(False, ("prompt_injection_suspected",), text)

    return GovernanceOutcome(True, (), text)


def refusal_message_for_outcome(
    outcome: GovernanceOutcome,
    language_hint: str = "en",
    *,
    question_text: Optional[str] = None,
) -> str:
    """
    Short user-facing refusal when the request is blocked before the graph runs.

    When question_text is set, we infer ar vs en from Arabic script ratio so refusals match
    the user's language before any LLM language-detection node runs.
    """
    hint = language_hint_from_text(question_text) if question_text else language_hint

    if hint == "ar":
        return (
            "لا يمكن معالجة هذا الطلب وفق سياسات الاستخدام. "
            "يُرجى طرح أسئلة متعلقة بمنصة الخوارزمي للاستبيانات الإحصائية فقط."
        )
    return (
        "This request cannot be processed under the assistant usage policy. "
        "Please ask only about the Al-Khwarizmi survey platform and product documentation."
    )


def log_blocked_attempt(
    *,
    reason_codes: tuple[str, ...],
    thread_id: str,
    request_id: str,
    question_full: str,
) -> None:
    """
    Append one audit line for blocked content (governance).

    Stores a short preview plus a truncated hash of the full question (privacy vs forensics tradeoff).
    """
    if not AUDIT_LOG_PATH:
        return
    try:
        path = Path(AUDIT_LOG_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        preview = question_full[:160]
        digest = hashlib.sha256(question_full.encode("utf-8", errors="replace")).hexdigest()[:16]
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": "governance_block",
            "reason_codes": list(reason_codes),
            "thread_id": thread_id,
            "request_id": request_id,
            "question_len": len(question_full),
            "question_preview": preview,
            "question_sha16": digest,
        }
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("Governance audit log write failed: %s", exc)


def _client_rate_key(request: Any) -> str:
    """Derive a coarse client bucket for rate limiting."""
    # Prefer stable API key material when caller authenticated with Bearer.
    auth = ""
    try:
        auth = request.headers.get("authorization") or ""
    except Exception:  # noqa: BLE001
        auth = ""
    if auth.lower().startswith("bearer ") and len(auth) > 20:
        return hashlib.sha256(auth[7:].strip().encode("utf-8")).hexdigest()[:32]
    xff = request.headers.get("x-forwarded-for") or ""
    if xff:
        return xff.split(",")[0].strip()[:128] or "unknown"
    try:
        return (request.client.host if request.client else "unknown")[:128]
    except Exception:  # noqa: BLE001
        return "unknown"


def enforce_rate_limit(request: Any) -> None:
    """
    Enforce GOVERNANCE_RATE_LIMIT_PER_MINUTE when > 0.

    Raises fastapi.HTTPException 429 on exceed. Uses Redis when available; otherwise memory
    (single-process only — use Redis or a gateway in production).
    """
    if RATE_LIMIT_PER_MINUTE <= 0:
        return

    from fastapi import HTTPException

    client = _client_rate_key(request)
    minute_bucket = int(time.time() // 60)
    key = (client, minute_bucket)

    try:
        from core.thread_memory import redis_client_optional

        r = redis_client_optional()
    except Exception:  # noqa: BLE001
        r = None

    if r is not None:
        try:
            rk = f"rag:gov:rl:{client}:{minute_bucket}"
            n = int(r.incr(rk))
            if n == 1:
                r.expire(rk, 120)
            if n > RATE_LIMIT_PER_MINUTE:
                raise HTTPException(
                    status_code=429,
                    detail={"code": "rate_limited", "message": "Too many requests; try again later."},
                )
            return
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis rate limit failed; falling back to memory: %s", exc)

    with _rate_lock:
        n = _rate_memory.get(key, 0) + 1
        _rate_memory[key] = n
        # Prune old buckets to bound memory.
        if len(_rate_memory) > 50_000:
            cutoff = minute_bucket - 5
            for k in list(_rate_memory.keys()):
                if k[1] < cutoff:
                    del _rate_memory[k]
    if n > RATE_LIMIT_PER_MINUTE:
        raise HTTPException(
            status_code=429,
            detail={"code": "rate_limited", "message": "Too many requests; try again later."},
        )
