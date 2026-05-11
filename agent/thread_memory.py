"""
Thread-scoped conversation memory for cumulative multi-turn chat.

Storage backends (pick one via environment):
- Redis when REDIS_URL is set: durable LIST per thread, atomic append with trim.
- Otherwise JSON files under MEMORY_PATH/threads/ (atomic replace on write).

Empty or missing thread_id from the client yields a new UUID on each first message;
reusing the same thread_id accumulates turns up to MEMORY_MAX_TURNS in prompts.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Client-supplied ids must be safe as Redis keys and filesystem names.
_THREAD_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")

MEMORY_PATH = Path(os.getenv("MEMORY_PATH", "./memory"))
try:
    MEMORY_MAX_TURNS = max(1, int(os.getenv("MEMORY_MAX_TURNS", "10")))
except ValueError:
    MEMORY_MAX_TURNS = 10
try:
    REDIS_MEMORY_TTL = int(os.getenv("REDIS_MEMORY_TTL", "0"))
except ValueError:
    REDIS_MEMORY_TTL = 0
REDIS_URL = os.getenv("REDIS_URL", "").strip()
REDIS_KEY_PREFIX = os.getenv("REDIS_KEY_PREFIX", "rag:thread:").strip() or "rag:thread:"

_redis: Any = None


def _get_redis():
    """Lazy Redis client; None if REDIS_URL unset or connection library missing."""
    global _redis
    if not REDIS_URL:
        return None
    if _redis is False:
        return None
    if _redis is not None:
        return _redis
    try:
        import redis as redis_lib
    except ImportError:
        logger.warning("REDIS_URL is set but redis package is not installed; using file memory.")
        _redis = False
        return None
    try:
        client = redis_lib.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=5.0,
            socket_timeout=5.0,
            health_check_interval=30,
        )
        client.ping()
        _redis = client
        logger.info("Thread memory: using Redis backend.")
        return _redis
    except Exception as exc:  # noqa: BLE001 — startup path: log and fall back
        logger.error("Redis connection failed (%s); falling back to file memory.", exc)
        _redis = False
        return None


def _redis_list_key(thread_id: str) -> str:
    return f"{REDIS_KEY_PREFIX}{thread_id}:turns"


def resolve_thread_id(raw: Optional[str]) -> str:
    """
    Return a validated thread id, or a new UUID when the client omits / blanks it.

    Raises ValueError with a stable message for HTTP 422 when the client id is invalid.
    """
    if raw is None:
        return str(uuid.uuid4())
    tid = raw.strip()
    if not tid:
        return str(uuid.uuid4())
    if ".." in tid or "/" in tid or "\\" in tid:
        raise ValueError("thread_id must not contain path separators or '..'.")
    if not _THREAD_ID_RE.match(tid):
        raise ValueError(
            "thread_id must be 1-128 characters: letters, digits, underscore, dot, colon, hyphen."
        )
    return tid


def _trim_turns(turns: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep only the last MEMORY_MAX_TURNS pairs for prompt context."""
    if len(turns) <= MEMORY_MAX_TURNS:
        return turns
    return turns[-MEMORY_MAX_TURNS:]


def _normalize_turns(raw: list) -> list[dict[str, str]]:
    """
    Coerce a raw list from storage into typed turn dicts.

    Silently drops any entry that is not a dict with both ``user`` and
    ``assistant`` keys — handles partial writes and legacy schema drift.
    """
    out: list[dict[str, str]] = []
    for t in raw:
        if isinstance(t, dict) and "user" in t and "assistant" in t:
            out.append({"user": str(t.get("user", "")), "assistant": str(t.get("assistant", ""))})
    return out


def load_conversation(thread_id: str) -> list[dict[str, str]]:
    """Load prior turns for this thread (empty list on miss or recoverable error)."""
    r = _get_redis()
    if r is not None:
        try:
            raw_items = r.lrange(_redis_list_key(thread_id), 0, -1)
            parsed: list = []
            for item in raw_items:
                try:
                    parsed.append(json.loads(item))
                except json.JSONDecodeError:
                    continue
            return _trim_turns(_normalize_turns(parsed))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Redis load_conversation failed for thread %s: %s", thread_id, exc)
            return []

    path = MEMORY_PATH / "threads" / f"{thread_id}.json"
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        raw = data.get("turns") if isinstance(data, dict) else None
        if not isinstance(raw, list):
            return []
        return _trim_turns(_normalize_turns(raw))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("File load_conversation failed for %s: %s", path, exc)
        return []


def append_turn(thread_id: str, user_message: str, assistant_message: str) -> None:
    """Append one turn after a successful graph run."""
    turn = json.dumps(
        {"user": user_message, "assistant": assistant_message},
        ensure_ascii=False,
    )
    r = _get_redis()
    if r is not None:
        key = _redis_list_key(thread_id)
        try:
            pipe = r.pipeline(transaction=True)
            pipe.rpush(key, turn)
            # Keep at most MEMORY_MAX_TURNS entries in the list.
            pipe.ltrim(key, -MEMORY_MAX_TURNS, -1)
            if REDIS_MEMORY_TTL > 0:
                pipe.expire(key, REDIS_MEMORY_TTL)
            pipe.execute()
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Redis append_turn failed for thread %s (%s); falling back to file.",
                thread_id, exc,
            )
            # Fall through to file backend so the turn is not silently dropped.

    _append_turn_file(thread_id, user_message, assistant_message)


def _append_turn_file(thread_id: str, user_message: str, assistant_message: str) -> None:
    """File backend: read-modify-write with atomic replace."""
    base = MEMORY_PATH / "threads"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{thread_id}.json"
    turns: list[dict[str, str]] = []
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            raw = data.get("turns") if isinstance(data, dict) else None
            if isinstance(raw, list):
                turns = _normalize_turns(raw)
        except (OSError, json.JSONDecodeError):
            turns = []
    turns.append({"user": user_message, "assistant": assistant_message})
    turns = turns[-MEMORY_MAX_TURNS:]
    payload = json.dumps({"turns": turns}, ensure_ascii=False, indent=0)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def redis_client_optional():
    """
    Return the shared Redis client when REDIS_URL is configured, else None.

    Used by optional features (e.g. governance rate limits) without importing Redis at import time.
    """
    return _get_redis()


_HISTORY_ASSISTANT_SUMMARY_LEN = 300


def format_history_for_prompt(turns: list[dict[str, str]]) -> str:
    """
    Render prior turns as a compact, numbered summary for LLM prompts.

    Each turn shows the full user message (usually short) and a truncated
    summary of the assistant answer so the LLM has enough topic context for
    follow-up resolution without being flooded by long previous responses.
    The most recent turn is always rendered in full so the LLM has the
    richest context for the immediately preceding exchange.

    Args:
        turns: Ordered list of ``{"user": ..., "assistant": ...}`` dicts,
               oldest first, as returned by ``load_conversation``.

    Returns:
        A formatted string ready for insertion into a prompt placeholder.
    """
    if not turns:
        return "(none)"

    blocks: list[str] = []
    last_idx = len(turns) - 1

    for i, t in enumerate(turns):
        u = (t.get("user") or "").strip()
        a = (t.get("assistant") or "").strip()
        if not u and not a:
            continue

        # Keep the most recent assistant answer in full; truncate older ones.
        if i == last_idx:
            a_display = a
        elif len(a) > _HISTORY_ASSISTANT_SUMMARY_LEN:
            a_display = a[:_HISTORY_ASSISTANT_SUMMARY_LEN].rstrip() + " …"
        else:
            a_display = a

        blocks.append(f"Turn {i + 1}:\n  User: {u}\n  Assistant: {a_display}")

    return "\n\n".join(blocks) if blocks else "(none)"
