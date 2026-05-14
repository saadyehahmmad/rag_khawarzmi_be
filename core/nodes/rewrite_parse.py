"""
Tolerant JSON parsing for the rewrite+route LLM output.
"""

from __future__ import annotations

import json

from .config import _RE_FENCE_CLOSE, _RE_FENCE_OPEN, _RE_JSON_FENCE, logger


def parse_rewrite_route_json(raw: str, fallback_question: str) -> tuple[str, str]:
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
