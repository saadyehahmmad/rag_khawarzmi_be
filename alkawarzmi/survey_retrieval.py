"""
Al-Khawarzmi implementation of :class:`framework.profile.SurveyRetrievalHooks`.

Delegates to designer layout helpers and semantic image selection. Kept in Layer 3
so the retrieval node stays free of direct ``alkawarzmi.designer`` imports.
"""

from __future__ import annotations

from typing import Any

from alkawarzmi.designer import (
    fetch_survey_all_rules,
    fetch_survey_overview_and_pages,
    fetch_survey_overview_only,
    wants_survey_question_catalog,
    wants_survey_rules_catalog,
    wants_survey_overview,
)
from alkawarzmi.image_selection import select_images_for_question
from framework.state import AgentState


class AlKhawarzmiSurveyRetrievalHooks:
    """Survey catalog, layout fetches, and screenshot selection for the Designer + survey RAG path."""

    def wants_survey_rules_catalog(self, state: AgentState) -> bool:
        return wants_survey_rules_catalog(state)

    def wants_survey_question_catalog(self, state: AgentState) -> bool:
        return wants_survey_question_catalog(state)

    def wants_survey_overview(self, state: AgentState) -> bool:
        return wants_survey_overview(state)

    def fetch_survey_overview_only(self, survey_store: Any, *, limit: int) -> list[Any]:
        return fetch_survey_overview_only(survey_store, limit=limit)

    def fetch_survey_all_rules(self, survey_store: Any, *, limit: int) -> list[Any]:
        return fetch_survey_all_rules(survey_store, limit=limit)

    def fetch_survey_overview_and_pages(self, survey_store: Any, *, limit: int) -> list[Any]:
        return fetch_survey_overview_and_pages(survey_store, limit=limit)

    def select_images_for_question(self, question: str, image_system: str, *, language: str) -> list[str]:
        return select_images_for_question(question, image_system, language=language)
