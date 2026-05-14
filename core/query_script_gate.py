"""
Gate user questions to Latin + Arabic scripts only (no LLM).

Used before spellcheck / routing so unsupported writing systems get a short,
bilingual prescript instead of retrieval or model calls.
"""

from __future__ import annotations

# Friendly bilingual prescript when the question contains disallowed characters.
UNSUPPORTED_SCRIPT_REPLY = (
    "We currently support **English** and **Arabic** only. "
    "Please write in one of those languages (or switch your keyboard) and try again.\n\n"
    "ندعم حاليًا **الإنجليزية** و**العربية** فقط. "
    "يرجى الكتابة بإحدى هاتين اللغتين (أو تغيير لوحة المفاتيح) والمحاولة مرة أخرى."
)

# Bidi / join controls sometimes pasted with Arabic text.
_BIDI_OK = frozenset("\u200c\u200d\u200e\u200f\ufeff")


def _char_supported(ch: str) -> bool:
    """Return True when *ch* is whitespace, basic/Latin-1 text, or Arabic script."""
    if ch.isspace() or ch in _BIDI_OK:
        return True
    o = ord(ch)
    if o < 0x20:  # other C0 controls (except isspace)
        return False
    if o <= 0x7E:  # printable ASCII
        return True
    if 0xA0 <= o <= 0xFF:  # Latin-1 supplement (letters with accents, common symbols)
        return True
    if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F:  # Arabic + supplement
        return True
    if 0x08A0 <= o <= 0x08FF:  # Arabic extended-A
        return True
    if 0xFB50 <= o <= 0xFDFF or 0xFE70 <= o <= 0xFEFF:  # presentation forms
        return True
    return False


def question_uses_only_supported_scripts(question: str) -> bool:
    """
    Return True when every character is allowed for English or Arabic input.

    Empty or whitespace-only input is treated as supported (downstream nodes decide).
    """
    if not question:
        return True
    return all(_char_supported(c) for c in question)
