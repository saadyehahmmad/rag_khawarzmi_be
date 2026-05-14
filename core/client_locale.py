"""
Client-driven locale for prescripted chat text (no LLM).

Single place for:
- **UI-driven** copy: ``ui_reply_language`` / ``say`` (``system_language`` first, else
  detected ``language``).
- **Prompt-driven** copy: ``prompt_reply_language`` / ``say_prompt`` — same language as
  the user's message (after ``language_detect``), for answers that must mirror how they asked.
- Raw question text from state.
"""

from __future__ import annotations

from typing import Literal

from core.text_ar import language_hint_from_text
from framework.state import AgentState


def ui_reply_language(state: AgentState) -> Literal["en", "ar"]:
    """
    Return ``\"ar\"`` or ``\"en\"`` for prescripted / UI-facing strings.

    Prefer ``state[\"system_language\"]`` when it is ``ar`` or ``en``; otherwise use
    ``state[\"language\"]`` (``ar`` only when that equals ``\"ar\"``).
    """
    raw = (state.get("system_language") or "").strip().lower()
    if raw == "ar":
        return "ar"
    if raw == "en":
        return "en"
    return "ar" if state.get("language") == "ar" else "en"


def prompt_reply_language(state: AgentState) -> Literal["en", "ar"]:
    """
    Language for prescripted replies that should match how the user wrote (prompt language).

    Uses ``state[\"language\"]`` from ``language_detect`` (``ar`` / ``en`` / ``mixed``).
    For ``mixed``, uses Arabic script density on the current question as a tie-breaker.

    Args:
        state: Graph state after ``language_detect_node`` has run.

    Returns:
        ``\"ar\"`` or ``\"en\"`` for choosing template strings.
    """
    raw = state.get("language") or "en"
    if raw == "ar":
        return "ar"
    if raw == "en":
        return "en"
    return "ar" if language_hint_from_text(current_question(state)) == "ar" else "en"


def say(state: AgentState, en: str, ar: str) -> str:
    """Pick ``en`` or ``ar`` text using ``ui_reply_language``."""
    return ar if ui_reply_language(state) == "ar" else en


def say_prompt(state: AgentState, en: str, ar: str) -> str:
    """Pick ``en`` or ``ar`` text using ``prompt_reply_language`` (question language)."""
    return ar if prompt_reply_language(state) == "ar" else en


def current_question(state: AgentState) -> str:
    """Latest user message from graph state."""
    return (state.get("question") or "").strip()


def question_with_rewrite(state: AgentState, *, sep: str = "\n") -> str:
    """
    Original question plus ``rewritten_question`` as one blob for intent / regex checks.

    Args:
        state: Graph state.
        sep: Joiner between the two segments (default newline, good for separate-line patterns).

    Returns:
        Stripped question only when rewrite is empty; otherwise ``question`` ``sep`` ``rewrite``.
    """
    q = (state.get("question") or "").strip()
    rw = (state.get("rewritten_question") or "").strip()
    if not rw:
        return q
    return f"{q}{sep}{rw}"
