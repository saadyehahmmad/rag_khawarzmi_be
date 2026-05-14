"""
Deterministic Designer-related prescripts (where-am-I, in-app navigation hints).

Used by ``alkawarzmi.payload_context``; no LLM. Route data comes from ``page_map``.
"""

from __future__ import annotations

import re

from core.client_locale import prompt_reply_language, say_prompt
from framework.state import AgentState

from .page_map import (
    describe_designer_location_for_user,
    resolve_designer_page,
    resolved_designer_route_is_known,
)

_NAV_VERB = re.compile(
    r"\b(go to|open|access|navigate to|get to|launch|how (do i|to) (open|get to|access|go to))\b",
    re.I,
)
_NAV_VERB_AR = re.compile(r"كيف [اأ]دخل|فتح|الذهاب [إل]|كيف [اأ]ذهب")
_ARTICLES = re.compile(r"^(the|a|an)\s+", re.I)
_IS_DESIGNER_APP = re.compile(r"\bdesigner\b|مصمّم الاستبيانات|مصمم", re.I)


def navigation_target_tail(question: str) -> str | None:
    """
    Return text after a navigation verb (e.g. ``open`` … ``builder``), or ``None``.

    Args:
        question: User message (trimmed by caller as needed).

    Returns:
        The tail segment to resolve as a Designer ``page_id`` hint, or ``None``.
    """
    m = _NAV_VERB.search(question) or _NAV_VERB_AR.search(question)
    if not m:
        return None
    tail = _ARTICLES.sub("", question[m.end() :].strip(" ?.؟")).strip()
    return tail or None


def reply_designer_where_am_i(state: AgentState) -> str:
    """
    Build a short “where am I” reply using ``page_id`` and the Designer route map.

    Args:
        state: Graph state with ``page_id`` and locale fields.

    Returns:
        Markdown-safe user message in the user's prompt language.
    """
    head = describe_designer_location_for_user(
        state.get("page_id"),
        lang=prompt_reply_language(state),
    )
    tail = say_prompt(
        state,
        en=(
            "Tell me what you want to do next — for example: **skip logic**, **validation**, "
            "**publishing**, **sample upload**, or a specific button or tab on this screen."
        ),
        ar=(
            "قل لي ماذا تريد أن تفعل بعد ذلك — مثلاً: **التخطي والمنطق**، **التحقق من الإجابات**، "
            "**النشر**، **رفع العينة**، أو أي زر أو تبويب في هذه الشاشة."
        ),
    )
    return f"{head}\n\n{tail}"


def reply_designer_navigation(state: AgentState, target_tail: str) -> str | None:
    """
    Short navigation hint for known Designer routes, or ``None`` for unknown targets.

    Handles “already in Designer” and “already on this screen” without LLM.

    Args:
        state: Graph state (``page_id``, locale).
        target_tail: Segment after the nav verb from ``navigation_target_tail``.

    Returns:
        Reply string, or ``None`` to let the graph fall through to RAG.
    """
    cur = resolve_designer_page(state.get("page_id"))

    if _IS_DESIGNER_APP.search(target_tail):
        return say_prompt(
            state,
            en=(
                "You're **already inside Survey Designer**.\n"
                f"Current screen: **{cur['title_en']}**."
            ),
            ar=(
                "أنت **داخل تطبيق مصمّم الاستبيانات** بالفعل.\n"
                f"شاشتك الحالية: **{cur['title_ar']}**."
            ),
        )

    dest = resolve_designer_page(target_tail)
    if not resolved_designer_route_is_known(dest):
        return None

    if dest["canonical"] == cur["canonical"]:
        return say_prompt(
            state,
            en=f"You're **already on {dest['title_en']}**.",
            ar=f"أنت **بالفعل في {dest['title_ar']}**.",
        )

    return say_prompt(
        state,
        en=(
            f"To get to **{dest['title_en']}**: {dest['desc_en']}\n\n"
            f"You're currently on: **{cur['title_en']}**."
        ),
        ar=(
            f"للانتقال إلى **{dest['title_ar']}**: {dest['desc_ar']}\n\n"
            f"أنت حالياً في: **{cur['title_ar']}**."
        ),
    )
