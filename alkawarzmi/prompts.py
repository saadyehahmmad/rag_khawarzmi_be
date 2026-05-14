"""
Al-Khawarzmi prompt provider — implements ``framework.profile.PromptProvider``.

Wraps :mod:`alkawarzmi.prompt_templates` so they can be composed into an
``RAGProfile`` without touching the underlying templates.
"""

from __future__ import annotations

from alkawarzmi.prompt_templates import ANSWER_PROMPT, REWRITE_AND_ROUTE_JSON_PROMPT


class AlKhawarzmiPrompts:
    """Provides the Al-Khawarzmi-branded prompt templates."""

    def rewrite_and_route(self) -> str:
        """Return the rewrite-and-route JSON prompt (includes platform name, system list)."""
        return REWRITE_AND_ROUTE_JSON_PROMPT

    def answer(self) -> str:
        """Return the final answer system prompt (includes platform branding, workflow knowledge)."""
        return ANSWER_PROMPT
