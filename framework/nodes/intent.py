"""
Heuristic intent: platform overview questions.

Standalone greetings use ``core.greeting_intent.is_greeting`` (re-exported here for callers).
"""

from __future__ import annotations

import re

from core.greeting_intent import is_greeting

_PLATFORM_NAMES = (
    "خوارزمي",
    "الخوارزمي",
    "al-khawarzmi",
    "alkhawarzmi",
    "al khawarzmi",
    "khawarzmi",
    "al-khwarzmi",
    "al khwarzmi",
    "khwarzmi",
)
_OVERVIEW_QUESTION_WORDS_AR = (
    "شو",
    "ما",
    "ايش",
    "إيش",
    "وش",
    "عرفني",
    "عرّفني",
    "احكيلي",
    "شرح",
    "شرحلي",
    "اخبرني",
    "أخبرني",
    "وضحلي",
    "وضّح",
    "فسرلي",
    "فسّر",
    "what",
    "tell me",
    "explain",
    "describe",
    "about",
)
_SYSTEM_KEYWORDS = (
    "مصمم",
    "designer",
    "builder",
    "بلدر",
    "ادمن",
    "admin",
    "field management",
    "ميداني",
    "callcenter",
    "call center",
    "كول سنتر",
    "مركز الاتصال",
    "runtime",
    "مشغل",
    "رن تايم",
)


def is_platform_overview(question: str) -> bool:
    """Return True when the user is asking what Al-Khawarzmi is (no specific system)."""
    q = question.strip().lower()
    if not any(name in q for name in _PLATFORM_NAMES):
        return False
    if any(kw in q for kw in _SYSTEM_KEYWORDS):
        return False
    words = set(re.split(r"[\s\W]+", q))
    return any(w in words for w in _OVERVIEW_QUESTION_WORDS_AR)
