"""
Prescripted replies for short “thanks / goodbye” turns (no LLM).

Uses ``say_prompt`` so wording follows **prompt language**, same as greetings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.client_locale import current_question, say_prompt
from core.closing_intent import is_standalone_closing_message

if TYPE_CHECKING:
    from framework.state import AgentState


def is_closing_thanks_turn(state: "AgentState") -> bool:
    """True when the current question is only a thank-you / brief sign-off."""
    return is_standalone_closing_message(current_question(state))


def closing_thanks_reply(state: "AgentState") -> str:
    """
    Friendly closing reply; invites further product questions without running RAG.

    Args:
        state: Graph state (for ``say_prompt`` locale).

    Returns:
        Markdown-safe bilingual message per ``say_prompt`` rules.
    """
    return say_prompt(
        state,
        en=(
            "You're welcome — glad that helped.\n\n"
            "If anything else comes up about **Designer**, **Field Management**, "
            "**Call Center**, or **Survey Runtime**, just ask."
        ),
        ar=(
            "العفو — يسعدنا أن نكون عند حسن ظنك.\n\n"
            "إذا احتجت لاحقاً أي توضيح عن **المصمم** أو **إدارة العمل الميداني** أو "
            "**مركز الاتصال** أو **مشغّل الاستمارة**، تفضل بسؤالك."
        ),
    )
