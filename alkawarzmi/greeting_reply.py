"""
Standalone greeting replies (no LLM): one source of truth for generic vs named copy.

Used by ``payload_context`` (prescript) and ``fallback_text`` (post-route) so wording,
bullets, and **prompt language** (``say_prompt``) never diverge.
"""

from __future__ import annotations

from core.client_locale import current_question, say_prompt
from core.greeting_intent import is_greeting
from framework.state import AgentState

# Shared “what I can help with” block — identical for generic and named greetings.
_GREETING_SCOPE_AR = """يمكنني الإجابة على أسئلتك المتعلقة بـ:
• **المصمم** — إنشاء الاستمارات وتصميمها
• **إدارة العمل الميداني** — توزيع الأدوار والمهام الميدانية
• **مركز الاتصال** — إدارة المقابلات الهاتفية
• **مشغّل الاستمارة** — تعبئة الاستبيانات ميدانياً"""

_GREETING_SCOPE_EN = """I can answer questions about:
• **Designer** — creating and designing surveys
• **Field Management (Admin)** — assigning roles and field tasks
• **Call Center** — managing CATI interviews
• **Survey Runtime** — filling surveys in the field"""

_INTRO_GENERIC_AR = (
    "أهلاً وسهلاً! أنا المساعد الذكي لمنصة الخوارزمي للمسوح الإحصائية. كيف يمكنني مساعدتك اليوم؟"
)
_INTRO_GENERIC_EN = (
    "Hello! I'm the AI assistant for the Al-Khwarzmi statistical survey platform. How can I help you today?"
)

_NAMED_BODY_AR = _INTRO_GENERIC_AR.removeprefix("أهلاً وسهلاً! ").strip()
_NAMED_BODY_EN = _INTRO_GENERIC_EN.removeprefix("Hello! ").strip()

# Public constants (stable greeting copy for API / tests).
GREETING_MESSAGE_AR = f"{_INTRO_GENERIC_AR}\n\n{_GREETING_SCOPE_AR}"
GREETING_MESSAGE_EN = f"{_INTRO_GENERIC_EN}\n\n{_GREETING_SCOPE_EN}"


def user_has_display_name(state: AgentState) -> bool:
    """True when the client sent at least one non-empty display name field."""
    return bool((state.get("user_name_en") or "").strip() or (state.get("user_name_ar") or "").strip())


def is_named_greeting_turn(state: AgentState) -> bool:
    """Standalone greeting (``is_greeting``) and client provided a name — prescript path."""
    q = current_question(state).rstrip("?.!,،؟").strip()
    if not q or not is_greeting(q):
        return False
    return user_has_display_name(state)


def generic_greeting_reply(state: AgentState) -> str:
    """Anonymous user: same intro + scope as historical ``GREETING_MESSAGE_*``, via ``say_prompt``."""
    return say_prompt(
        state,
        en=f"{_INTRO_GENERIC_EN}\n\n{_GREETING_SCOPE_EN}",
        ar=f"{_INTRO_GENERIC_AR}\n\n{_GREETING_SCOPE_AR}",
    )


def named_greeting_reply(state: AgentState) -> str:
    """
    Logged-in user: personalized salutation, same middle line as generic (minus redundant hello),
    then the **same** scope block as generic (via ``say_prompt``).
    """
    en = (state.get("user_name_en") or "").strip()
    ar = (state.get("user_name_ar") or "").strip()
    return say_prompt(
        state,
        en=(
            f"Hello **{en or ar}**!\n\n"
            f"{_NAMED_BODY_EN}\n\n"
            f"{_GREETING_SCOPE_EN}"
        ),
        ar=(
            f"أهلاً وسهلاً **{ar or en}**!\n\n"
            f"{_NAMED_BODY_AR}\n\n"
            f"{_GREETING_SCOPE_AR}"
        ),
    )


def standalone_greeting_reply(state: AgentState) -> str:
    """
    Pick named vs generic greeting using the same rules as the payload prescript.

    Call from fallback when ``is_greeting`` already holds for the current question.
    """
    if user_has_display_name(state):
        return named_greeting_reply(state)
    return generic_greeting_reply(state)
