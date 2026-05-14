"""
Al-Khawarzmi fallback provider — implements ``framework.profile.FallbackProvider``.

Wraps :func:`framework.nodes.fallback_text.fallback_answer_text` so it can be composed into an
``RAGProfile``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from framework.nodes.fallback_text import fallback_answer_text as _fallback_answer_text

if TYPE_CHECKING:
    from framework.state import AgentState


class AlKhawarzmiFallback:
    """Provides bilingual fallback messages for the Al-Khawarzmi platform."""

    def get(self, state: "AgentState") -> str:
        """Return the appropriate bilingual fallback string for the given state."""
        return _fallback_answer_text(state)
