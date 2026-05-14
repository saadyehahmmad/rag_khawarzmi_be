"""
Deterministic prescripted replies from the HTTP payload (no LLM).

Priority (first match wins): where-am-I → navigation → thanks/closing → named greeting.
(Language / UI locale questions are answered by normal RAG + LLM, not prescripted.)
Prescript wording follows **prompt language** (how the user wrote), not UI locale;
see ``say_prompt`` in ``core.client_locale``.
Designer routes: ``alkawarzmi.designer.prescripts``.
"""

from __future__ import annotations

import re

from alkawarzmi.designer.prescripts import (
    navigation_target_tail,
    reply_designer_navigation,
    reply_designer_where_am_i,
)
from alkawarzmi.closing_reply import closing_thanks_reply, is_closing_thanks_turn
from alkawarzmi.greeting_reply import is_named_greeting_turn, named_greeting_reply
from core.client_locale import (
    current_question,
    question_with_rewrite,
)
from framework.state import AgentState

# --- Intent patterns -----------------------------------------------------------

_WHERE = re.compile(
    r"\b(where am i|which page|what (page|screen)|current page|where am i now)\b"
    r"فين انا|انا فين|في أي صفحة|في اي صفحة|وين انا|أين أنا|اين انا|ما اسم الصفحة|ايش الصفحة",
    re.I,
)


# --- Replies -------------------------------------------------------------------


def _prescript_from_payload(state: AgentState) -> str | None:
    q = current_question(state)
    combined = question_with_rewrite(state, sep="\n")

    if _WHERE.search(combined):
        return reply_designer_where_am_i(state)

    tail = navigation_target_tail(q)
    if tail is not None:
        nav = reply_designer_navigation(state, tail)
        if nav:
            return nav

    if is_closing_thanks_turn(state):
        return closing_thanks_reply(state)

    if is_named_greeting_turn(state):
        return named_greeting_reply(state)

    return None


def payload_context_step(state: AgentState) -> dict[str, str]:
    """Return ``prescripted_answer`` when payload alone determines the reply."""
    if state.get("prescripted_answer"):
        return {}
    text = _prescript_from_payload(state)
    return {"prescripted_answer": text} if text else {}
