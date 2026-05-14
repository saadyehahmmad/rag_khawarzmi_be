"""
Al-Khawarzmi intent detector — implements ``framework.profile.IntentDetector``.

Wraps :func:`framework.nodes.intent.is_platform_overview` so it can be composed into an
``RAGProfile`` without changing the underlying implementation.
"""

from __future__ import annotations

from framework.nodes.intent import is_platform_overview as _is_platform_overview


class AlKhawarzmiIntentDetector:
    """Detects Al-Khawarzmi-specific intents (platform overview, system keywords)."""

    def is_platform_overview(self, question: str) -> bool:
        """Return True when the user is asking what Al-Khawarzmi is (no specific system)."""
        return _is_platform_overview(question)
