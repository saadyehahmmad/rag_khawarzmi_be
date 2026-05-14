"""
Standalone greeting detection (no LLM).

Shared by pipeline nodes (rewrite short-circuit, fallbacks) and prescripts
(named-greeting prescript) so rules stay aligned: a greeting prefix plus a real question
(e.g. «هاي الاستمارة شايفها؟») must **not** be treated as a bare greeting.
"""

from __future__ import annotations

_GREETING_PATTERNS_EN = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "howdy",
        "hola",
        "bonjour",
        "guten tag",
        "good morning",
        "good afternoon",
        "good evening",
        "good day",
        "greetings",
        "how are you",
        "how are you doing",
        "how do you do",
        "what's up",
        "whats up",
        "sup",
        "yo",
        "hiya",
    }
)
_GREETING_PATTERNS_AR = frozenset(
    {
        "مرحبا",
        "مرحباً",
        "السلام عليكم",
        "أهلا",
        "أهلاً",
        "اهلا",
        "اهلاً",
        "صباح الخير",
        "مساء الورد",
        "مساء الخير",
        "كيف حالك",
        "شخبارك",
        "مساء النور",
        "كيف الحال",
        "كيفك",
        "شلونك",
        "هلا",
        "هلو",
        "سلام",
    }
)


def is_greeting(question: str) -> bool:
    """Return True only for a short standalone greeting, not greeting + a real question."""
    q = question.strip().rstrip("?.!,،؟").strip().lower()
    if q in _GREETING_PATTERNS_EN or q in _GREETING_PATTERNS_AR:
        return True
    for pat in _GREETING_PATTERNS_EN:
        if q.startswith(pat) and len(q) <= len(pat) + 8:
            return True
    for pat in _GREETING_PATTERNS_AR:
        if q.startswith(pat) and len(q) <= len(pat) + 8:
            return True
    return False
