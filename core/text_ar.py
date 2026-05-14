"""
Arabic text helpers for governance and retrieval.

We keep this dependency-light for governance, but use CAMeL Tools for morphological
lemmatization in BM25 when available. Goals:
- Improve recall for Arabic user questions via proper morphological lemmatization.
- Support Gulf Arabic (Khaleeji) via the dedicated calima-glf-01 database.
- Bridge Gulf dialect vocabulary to MSA so BM25 finds the right corpus chunks.
- Remove Arabic stopwords to improve BM25 precision.
- Give governance a script hint for refusal language before the LLM runs.
- Normalize Arabic query tokens via ``apply_gulf_vocab_bridge`` after Unicode normalize.
"""

from __future__ import annotations

import unicodedata
from typing import Any, Literal

# ─── Gulf Arabic vocabulary bridge ───────────────────────────────────────────
# CAMeL morphological lemmatization handles *inflections* within a dialect, but
# it cannot bridge *vocabulary* gaps between Gulf and MSA (e.g. "وين" lemmatises
# to "وين", not "أين", because they are different lexemes).  These explicit
# mappings cover the most common Gulf words that appear in support questions.
_GULF_VOCAB_BRIDGE: dict[str, str] = {
    # Question words
    "وين":    "أين",
    "فين":    "أين",
    "شو":     "ماذا",
    "إيش":    "ماذا",
    "ايش":    "ماذا",
    "وش":     "ماذا",
    "ليش":    "لماذا",
    "ليه":    "لماذا",
    # Time adverbs
    "الحين":  "الآن",
    "الهين":  "الآن",
    "هسع":    "الآن",
    "هلحين":  "الآن",
    # Negation
    "مو":     "ليس",
    "مب":     "ليس",
    # Modal verbs — want
    "أبغى":   "أريد",
    "ابغى":   "أريد",
    "أبي":    "أريد",
    "ابي":    "أريد",
    "يبغى":   "يريد",
    "يبي":    "يريد",
    "تبغى":   "تريد",
    "نبغى":   "نريد",
    # Same modals after ``normalize_arabic_question`` (ى → ي) or ي keyboard input
    "ابغي":   "أريد",
    "أبغي":   "أريد",
    "تبغي":   "تريد",
    "يبغي":   "يريد",
    "نبغي":   "نريد",
    # Modal verbs — can / able
    "أقدر":   "أستطيع",
    "اقدر":   "أستطيع",
    "تقدر":   "تستطيع",
    "يقدر":   "يستطيع",
    "نقدر":   "نستطيع",
    # Action verbs — do / make
    "أسوي":   "أفعل",
    "اسوي":   "أفعل",
    "يسوي":   "يفعل",
    "نسوي":   "نفعل",
    # Action verbs — see / look
    "أشوف":   "أرى",
    "اشوف":   "أرى",
    "يشوف":   "يرى",
    "شوف":    "انظر",
    # Navigation
    "أدش":    "أدخل",
    "ادش":    "أدخل",
}


def apply_gulf_vocab_bridge(token: str) -> str:
    """
    Map a Gulf Arabic vocabulary token to its MSA equivalent.

    Applied before CAMeL lemmatization so the MSA form — not the Gulf form —
    is what gets lemmatized and scored against the MSA corpus.
    Returns the token unchanged when no mapping exists.
    """
    return _GULF_VOCAB_BRIDGE.get(token, token)


# ─── Arabic stopwords ─────────────────────────────────────────────────────────
# High-frequency function words that carry no retrieval signal.  Removing them
# raises BM25 precision by preventing common words from dominating term scores.
_ARABIC_STOPWORDS: frozenset[str] = frozenset({
    # Prepositions
    "في", "من", "إلى", "الى", "على", "عن", "مع", "عند", "حتى",
    "بعد", "قبل", "خلال", "عبر", "ضمن", "حول", "بين", "فوق", "تحت",
    # Conjunctions
    "و", "أو", "او", "ثم", "بل", "لكن", "فإن", "إذ",
    # Demonstratives
    "هذا", "هذه", "هؤلاء", "ذلك", "تلك", "ذا", "هاذا",
    # Pronouns
    "هو", "هي", "هم", "هن", "نحن", "أنا", "انا", "أنت", "انت",
    # Relative pronouns
    "الذي", "التي", "الذين", "اللاتي", "اللواتي",
    # Common auxiliaries / copula
    "كان", "كانت", "كانوا", "يكون", "تكون", "يكن",
    # Question particles (as standalone tokens they add no corpus signal)
    "هل",
    # Common filler verbs in technical documentation
    "يتم", "يمكن",
    # Short particles
    "أي", "أ", "إن", "أن", "ان", "لو",
})


def is_arabic_stopword(token: str) -> bool:
    """Return True when the token is a pure Arabic function word with no retrieval value."""
    return token in _ARABIC_STOPWORDS


# ─── CAMeL Tools lazy loaders ────────────────────────────────────────────────
# Analyzers are initialised on first use to avoid import-time delays.
# None = not yet tried; False = tried and failed (no retry on subsequent calls).
_camel_glf: Any = None  # calima-glf-01  (Gulf Arabic morphology)
_camel_msa: Any = None  # calima-msa-r13 (MSA morphology — fallback)


def _load_analyzer(db_name: str) -> Any:
    """
    Load a CAMeL Tools MorphologyDB analyzer by name.

    Returns the Analyzer instance on success, or None on any failure (library not
    installed, data not downloaded, version mismatch, etc.). Errors are silently
    swallowed so the rest of the pipeline continues without lemmatization.
    """
    try:
        from camel_tools.morphology.database import MorphologyDB
        from camel_tools.morphology.analyzer import Analyzer

        db = MorphologyDB.builtin_db(db_name)
        # ADD_PROP backoff: unknown words are treated as proper nouns and returned
        # as-is rather than producing no analysis.
        return Analyzer(db, backoff="ADD_PROP", cache_size=2000)
    except Exception:
        return None


def _glf_analyzer() -> Any:
    """Lazy-load the Gulf Arabic analyzer; return None if unavailable."""
    global _camel_glf
    if _camel_glf is None:
        _camel_glf = _load_analyzer("calima-glf-01") or False
    return _camel_glf if _camel_glf is not False else None


def _msa_analyzer() -> Any:
    """Lazy-load the MSA analyzer; return None if unavailable."""
    global _camel_msa
    if _camel_msa is None:
        _camel_msa = _load_analyzer("calima-msa-r13") or False
    return _camel_msa if _camel_msa is not False else None


def _strip_diacritics(token: str) -> str:
    """
    Remove Tashkeel (diacritics) and normalize Alef variants before morphological
    analysis.  CAMeL's DEFAULT_NORMALIZE_MAP handles some of this internally, but
    stripping diacritics up-front significantly improves analyzer hit rates for
    user input that may carry vocalization marks.
    """
    nfkd = unicodedata.normalize("NFKD", token)
    no_marks = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    out: list[str] = []
    for ch in no_marks:
        if ch in ("\u0622", "\u0623", "\u0625", "\u0671"):  # Alef variants
            out.append("\u0627")
        elif ch == "\u0640":  # Tatweel (kashida)
            continue
        else:
            out.append(ch)
    return "".join(out)


def camel_get_lemmas(token: str) -> list[str]:
    """
    Return all unique morphological lemmas for an Arabic token (up to 3).

    Collects lemmas from all analyses returned by both the Gulf and MSA
    analyzers, deduplicating across databases.  Multiple lemmas per token
    enable soft query expansion: the token matches any of its possible
    base forms in the corpus, improving recall without adding noise.

    Returns ``[token]`` unchanged when no analyzer is available or no
    lemma can be extracted, so BM25 still operates as a fallback.
    """
    normalised = _strip_diacritics(token)
    seen: set[str] = set()
    result: list[str] = []

    for get_az in (_glf_analyzer, _msa_analyzer):
        az = get_az()
        if az is None:
            continue
        try:
            analyses = az.analyze(normalised)
            for analysis in analyses:
                lem = analysis.get("lem", "").strip()
                if lem and len(lem) >= 2 and lem not in seen:
                    seen.add(lem)
                    result.append(lem)
                    if len(result) >= 3:
                        return result
        except Exception:
            continue

    return result if result else [token]


# ─── Script / language helpers ───────────────────────────────────────────────

def arabic_script_ratio(text: str) -> float:
    """Share of characters in Arabic script blocks (rough language hint)."""
    if not text:
        return 0.0
    n = sum(1 for ch in text if "\u0600" <= ch <= "\u06FF" or "\u0750" <= ch <= "\u077F")
    return n / max(len(text), 1)


def language_hint_from_text(text: str) -> str:
    """Return 'ar' or 'en' for short user-facing strings (governance refusals)."""
    return "ar" if arabic_script_ratio(text) >= 0.12 else "en"


def detect_language_for_rag(text: str) -> Literal["ar", "en", "mixed"]:
    """
    Heuristic language label for prompts and fallbacks (no LLM call).

    Uses Arabic script density only; good enough for answer language and retrieval hints.
    """
    r = arabic_script_ratio(text)
    if r >= 0.32:
        return "ar"
    if r >= 0.06:
        return "mixed"
    return "en"


def normalize_arabic_question(text: str) -> str:
    """
    Normalize common Arabic Unicode variants before substring/pattern checks.

    Used by governance.py for injection-pattern matching. Delegates shared
    Tashkeel/Alef/Tatweel normalization to ``_strip_diacritics``, then adds
    the two governance-specific transforms:
    - Alef Maqsura (ى) → Ya (ي) for Gulf/MSA consistency.
    - Teh Marbuta (ة) → Heh (ه) for suffix matching.

    Does NOT change meaning; reduces trivial bypass variants.
    """
    if not text:
        return text
    return _strip_diacritics(text).replace("\u0649", "\u064A").replace("\u0629", "\u0647")


def normalize_arabic_query_typo_token(token: str) -> str:
    """
    Normalize one Arabic user-query token before prescripts / routing.

    Applies ``normalize_arabic_question`` then ``apply_gulf_vocab_bridge`` so all
    Gulf→MSA lexical fixes use ``_GULF_VOCAB_BRIDGE`` (single source of truth).

    Args:
        token: A single token (no surrounding whitespace); punctuation should be stripped by caller.

    Returns:
        The same token when it is not mostly Arabic script, otherwise the bridged form.
    """
    if not token or arabic_script_ratio(token) < 0.35:
        return token
    n = normalize_arabic_question(token)
    return apply_gulf_vocab_bridge(n)


def normalize_arabic_answer_text(text: str) -> str:
    """
    Apply minimal, targeted Arabic spelling fixes to the model's final answer.

    This is intentionally conservative (no broad rewriting) to avoid changing meaning.
    It only corrects a small set of observed recurring typos in this domain.
    """
    if not text:
        return text
    # Common model typo: "قواعس" (wrong root) instead of "قواعد" (rules). Apply longer phrases first.
    fixes = (
        ("بالقواعس", "بالقواعد"),
        ("للقواعس", "للقواعد"),
        ("والقواعس", "والقواعد"),
        ("القواعس", "القواعد"),
        ("قواعس", "قواعد"),
    )
    out = text
    for wrong, right in fixes:
        out = out.replace(wrong, right)
    return out
