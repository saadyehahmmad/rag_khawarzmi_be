"""
Static fallback copy when retrieval is not usable (sync + SSE).
"""

from __future__ import annotations

from alkawarzmi.designer.survey_listing_intent import wants_survey_structure_help
from alkawarzmi.greeting_reply import standalone_greeting_reply
from core.client_locale import current_question, say_prompt
from core.greeting_intent import is_greeting
from alkawarzmi.prompt_templates import (
    FALLBACK_MESSAGE_AR,
    FALLBACK_MESSAGE_EN,
    PLATFORM_OVERVIEW_MESSAGE_AR,
    PLATFORM_OVERVIEW_MESSAGE_EN,
    SURVEY_INGESTING_WAIT_AR,
    SURVEY_INGESTING_WAIT_EN,
    SURVEY_NOT_INGESTED_AR,
    SURVEY_NOT_INGESTED_EN,
)
from framework.state import AgentState

from .intent import is_platform_overview


def fallback_answer_text(state: AgentState) -> str:
    """Static bilingual message when retrieval is not usable (sync + SSE paths)."""
    lang = state.get("language", "en")
    original_q = state.get("question", "")

    if state.get("survey_ingesting"):
        return say_prompt(
            state,
            en=SURVEY_INGESTING_WAIT_EN,
            ar=SURVEY_INGESTING_WAIT_AR,
        )

    if state.get("survey_index_absent") and wants_survey_structure_help(state):
        return say_prompt(
            state,
            en=SURVEY_NOT_INGESTED_EN,
            ar=SURVEY_NOT_INGESTED_AR,
        )

    if is_greeting(current_question(state)):
        return standalone_greeting_reply(state)
    if is_platform_overview(original_q):
        return PLATFORM_OVERVIEW_MESSAGE_AR if lang == "ar" else PLATFORM_OVERVIEW_MESSAGE_EN
    return FALLBACK_MESSAGE_AR if lang == "ar" else FALLBACK_MESSAGE_EN
