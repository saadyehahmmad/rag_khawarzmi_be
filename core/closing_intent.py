"""
Standalone “thanks / goodbye” detection (no LLM).

Short closing-only messages are prescripted so the graph skips retrieval and models.
Must not match a real question that merely contains “thanks” (length + pattern guards).
"""

from __future__ import annotations

import re

from core.text_ar import normalize_arabic_question

# English: whole message is a thank-you / sign-off (optional intensifier / punctuation).
_RE_CLOSING_EN = re.compile(
    r"^(\s*)("
    r"thanks?(\s+you|\s+a\s+lot|\s+so\s+much|\s+again)?"
    r"|thank\s+you(\s+(so\s+much|very\s+much|again))?"
    r"|thx|ty(\s+vm)?"
    r"|much\s+appreciated"
    r"|appreciate\s+it"
    r"|cheers"
    r"|(good)?bye(\s+now)?"
    r"|see\s+you(\s+later|\s+soon)?"
    r"|got\s+it,?\s*thanks?"
    r"|ok(ay)?,?\s*thanks?"
    r"|perfect,?\s*thanks?"
    r")(\s*[.!…]*)$",
    re.I | re.VERBOSE,
)

# Arabic: common closings only (normalized for alef/hamza variants).
_RE_CLOSING_AR = re.compile(
    r"^(\s*)("
    r"شكرا|شكراً|شكرًا"
    r"|مشكور|مشكورة|مشكورين"
    r"|تسلم|تسلمين"
    r"|يسلمو|يسلمونك"
    r"|الله\s*يعطيك\s*العافية|يعطيك\s*العافية|يعافيك"
    r"|الحمد\s*لله"
    r"|مع\s*السلامة|باي|سلام"
    r")(\s*(جزيلا|كتير|كلش|هالمرة)?)(\s*[.!…]*)$",
    re.VERBOSE,
)


def is_standalone_closing_message(text: str) -> bool:
    """
    Return True when *text* is only a thank-you / brief sign-off, not a product question.

    Uses length and full-string regex so “thanks, how do I …?” stays False.
    """
    raw = (text or "").strip()
    if not raw:
        return False
    if len(raw) > 120:
        return False
    q = raw.rstrip("?.!,،؟…").strip()
    if not q:
        return False
    if _RE_CLOSING_EN.match(q):
        return True
    n = normalize_arabic_question(q)
    if _RE_CLOSING_AR.match(n) or _RE_CLOSING_AR.match(q):
        return True
    return False
