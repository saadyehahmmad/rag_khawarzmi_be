"""Deterministic query touch-ups before the first LLM (Arabic via ``text_ar.apply_gulf_vocab_bridge``)."""

from __future__ import annotations

import re
from functools import lru_cache

from core.text_ar import normalize_arabic_query_typo_token

_EDGE = frozenset('.,!?;:()[]{}\"\'«»`…،؟')
# Latin that looks like Arabic transliteration — do not English-spellfix.
_ROMAN_AR = re.compile(r"(kh|gh|dh|th|sh|zh|\bqu\b|\bqw\b)", re.I)
# Names / brands — never replace Latin token via spellchecker.
_SKIP_EN: frozenset[str] = frozenset({
    "abdullah", "ahmad", "ahmed", "alkhawarzmi", "khawarzmi", "khwarzmi",
    "mohammad", "mohammed", "muhammad", "qatar", "sadieh",
})


@lru_cache(maxsize=1)
def _spell():
    from spellchecker import SpellChecker

    return SpellChecker(language="en")


def _edges(tok: str) -> tuple[str, str, str]:
    i, j = 0, len(tok)
    while i < j and tok[i] in _EDGE:
        i += 1
    while j > i and tok[j - 1] in _EDGE:
        j -= 1
    return tok[:i], tok[i:j], tok[j:]


def _fix_en(core: str) -> str:
    low = core.lower()
    if len(low) < 4 or not core.isascii() or not core.isalpha():
        return core
    if low in _SKIP_EN or _ROMAN_AR.search(low) or "://" in low or "@" in low:
        return core
    sp = _spell()
    if low in sp:
        return core
    ok = sp.known(sp.edit_distance_1(low))
    if len(ok) != 1:
        return core
    fix = next(iter(ok))
    return fix.upper() if core.isupper() else fix.capitalize() if core[:1].isupper() else fix


def normalize_query_typo(question: str) -> str:
    """Whitespace-preserving token pass: Arabic via ``normalize_arabic_query_typo_token``, then English distance-1 if unambiguous."""
    if not (question and question.strip()):
        return question
    out: list[str] = []
    for m in re.finditer(r"\S+|\s+", question):
        ch = m.group()
        if ch.isspace():
            out.append(ch)
            continue
        a, core, b = _edges(ch)
        if not core:
            out.append(ch)
            continue
        mid = normalize_arabic_query_typo_token(core)
        if mid == core and core.isascii() and core.isalpha():
            mid = _fix_en(core)
        out.append(a + mid + b)
    return "".join(out)
